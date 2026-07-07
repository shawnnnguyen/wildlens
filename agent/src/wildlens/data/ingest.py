"""
Data ingestion CLI for the Safari Guide.

Usage
─────
# Fetch EOL + IUCN text, push to Supabase + Pinecone
python -m safari_guide.data.ingest --text

# Download images from ALL sources (LILA BC + Ultralytics), upload to Supabase Storage
python -m safari_guide.data.ingest --images

# Run only LILA BC image ingestion
python -m safari_guide.data.ingest --lila-only

# Run only Ultralytics African Wildlife Dataset ingestion
python -m safari_guide.data.ingest --ultralytics-only

# Everything: text + all image sources
python -m safari_guide.data.ingest --all

# Re-fetch everything even if cache exists
python -m safari_guide.data.ingest --all --force

# Show what would be done, no writes
python -m safari_guide.data.ingest --dry-run

Image sources
─────────────
EOL (Encyclopedia of Life)    — up to 10 images per species, stored as remote URLs + license
LILA BC / Snapshot Serengeti  — 48 species, camera-trap photos, uploaded to Supabase Storage
Ultralytics African Wildlife  — 4 species (buffalo, elephant, rhino, zebra), ~248 clean labeled images

Text sources
────────────
EOL        — Encyclopedia of Life: species overview, habitat, behavior, diet
Wikipedia  — Full article extract parsed into sections; fills gaps EOL misses
API Ninjas — Structured animal facts: speed, weight, lifespan, prey, gestation, etc.
IUCN       — Red List: conservation status, population trend, threats

Environment variables required
───────────────────────────────
PINECONE_API_KEY            — Pinecone API key
PINECONE_INDEX_NAME         — Pinecone index name (default: safari-guide)
SUPABASE_URL                — Supabase project URL
SUPABASE_INGEST_KEY         — Supabase service-role key (falls back to SUPABASE_KEY)
IUCN_API_KEY                — IUCN Red List API key (optional; uses mock if absent)
API_NINJAS_KEY              — API Ninjas key (optional; skips characteristics chunk if absent)
LILA_IMAGES_PER_SPECIES     — how many LILA BC images per species (default: 20)
EOL_IMAGES_PER_SPECIES      — how many EOL images per species (default: 10)
EOL_COMMERCIAL_ONLY         — set to 'true' to skip NC-licensed EOL images
ULTRALYTICS_DATASET_URL     — override Ultralytics zip URL (optional)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("safari_guide.data.ingest")

_SPECIES_LIST_PATH = Path(__file__).parent / "species_list.json"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_species_list() -> list[dict]:
    with open(_SPECIES_LIST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _init_pinecone():
    """Return a Pinecone Index object for upserting text embeddings."""
    try:
        from pinecone import Pinecone
    except ImportError:
        log.error("pinecone package not installed. Run: pip install pinecone")
        sys.exit(1)

    api_key    = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "safari-guide")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY is not set.")

    pc    = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    log.info(f"[Pinecone] Connected to index '{index_name}'")
    return index


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings with all-MiniLM-L6-v2 (384-dim)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

    embedder = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedder.embed_documents(texts)


def _pinecone_vector_id(common_name: str, section: str, source: str) -> str:
    """Deterministic Pinecone vector ID — allows idempotent upserts."""
    slug = f"{common_name}::{section}::{source}".lower().replace(" ", "_")
    return slug


# ══════════════════════════════════════════════════════════════════════════════
# TEXT INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def run_text_ingest(
    species_list: list[dict],
    force:        bool = False,
    dry_run:      bool = False,
) -> None:
    """
    For each species, fetch from all four text sources then push to Supabase + Pinecone:
      1. EOL        — rich species text, section-level (highest priority)
      2. Wikipedia  — full article parsed into sections (fills EOL gaps)
      3. API Ninjas — structured facts: speed, weight, prey, gestation, etc.
      4. IUCN       — conservation status, population trend, threats (own chunk)
      5. Handcrafted safety notes from species_list.json (always present)
    """
    from .fetcher import APINinjasClient, EOLClient, IUCNClient, WikipediaClient, build_species_chunks
    from .supabase_store import SupabaseStore

    store    = SupabaseStore(role="ingest") if not dry_run else None
    pc_index = _init_pinecone() if not dry_run else None

    # Clear cache dirs if --force
    if force:
        import shutil
        from pathlib import Path
        repo_root  = Path(__file__).resolve().parents[4]
        cache_root = repo_root / "data" / "cache"
        if cache_root.exists():
            shutil.rmtree(cache_root)
            log.info("[Ingest] Cache cleared (--force)")

    total_chunks = 0

    with EOLClient() as eol, WikipediaClient() as wiki, APINinjasClient() as ninjas, IUCNClient() as iucn:
        for entry in species_list:
            common_name     = entry["common_name"]
            scientific_name = entry.get("scientific_name", common_name)
            log.info(f"\n── {common_name} ({scientific_name}) ──")

            eol_sections     = eol.fetch(common_name, scientific_name)
            wiki_sections    = wiki.fetch(common_name, scientific_name)
            api_ninjas_data  = ninjas.fetch(common_name)
            iucn_data        = iucn.fetch(scientific_name)
            chunks           = build_species_chunks(
                entry, eol_sections, iucn_data, wiki_sections, api_ninjas_data
            )

            sources_summary = (
                f"EOL={len(eol_sections)} sections | "
                f"Wiki={len(wiki_sections)} sections | "
                f"Ninjas={'✓' if api_ninjas_data else '–'} | "
                f"IUCN={iucn_data.get('category', 'N/A')} | "
                f"chunks={len(chunks)}"
            )
            log.info(f"  {sources_summary}")

            if dry_run:
                for c in chunks:
                    print(f"    [{c['section']} / {c['source']}] {c['content'][:80]}…")
                continue

            # Supabase: upsert species row + all document chunks
            species_id = store.upsert_species(entry)
            for chunk in chunks:
                store.upsert_document(
                    species_id=species_id,
                    section=chunk["section"],
                    content=chunk["content"],
                    source=chunk["source"],
                )

            # Pinecone: embed all chunks and upsert to 'text' namespace
            texts   = [c["content"] for c in chunks]
            vectors = _embed_texts(texts)

            pinecone_records = []
            for chunk, vector in zip(chunks, vectors):
                pinecone_records.append({
                    "id":     _pinecone_vector_id(common_name, chunk["section"], chunk["source"]),
                    "values": vector,
                    "metadata": {
                        "species":      common_name,
                        "section":      chunk["section"],
                        "source":       chunk["source"],
                        "threat_level": entry.get("threat_level", "low"),
                        "text":         chunk["content"][:1000],
                    },
                })

            pc_index.upsert(vectors=pinecone_records, namespace="text")
            total_chunks += len(chunks)
            log.info(f"  → {len(chunks)} chunks upserted to Supabase + Pinecone")

    if not dry_run:
        log.info(f"\n[Ingest] Text ingestion complete — {total_chunks} total chunks")
        log.info(f"  Supabase: {store.species_count()} species, {store.document_count()} documents")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE INGESTION — LILA BC
# ══════════════════════════════════════════════════════════════════════════════

def run_lila_ingest(
    species_list:  list[dict],
    n_per_species: int  = 20,
    dry_run:       bool = False,
) -> None:
    """
    Download LILA BC / Snapshot Serengeti sample images per species
    and upload to Supabase Storage.
    """
    from .lila import LILADownloader
    from .supabase_store import SupabaseStore

    store      = SupabaseStore(role="ingest") if not dry_run else None
    downloader = LILADownloader()

    results = downloader.run(
        species_list=species_list,
        supabase_store=store,
        n_per_species=n_per_species,
        dry_run=dry_run,
    )

    total = sum(results.values())
    log.info(f"\n[LILA BC] Complete — {total} images uploaded across {len(results)} species")
    for name, count in results.items():
        log.info(f"  {name}: {count} images")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE INGESTION — ULTRALYTICS AFRICAN WILDLIFE DATASET
# ══════════════════════════════════════════════════════════════════════════════

def run_eol_image_ingest(
    species_list:          list[dict],
    images_per_species:    int  = 10,
    allow_commercial_only: bool = False,
    dry_run:               bool = False,
) -> None:
    """
    Fetch EOL image URLs per species and store them as remote URL records
    in the Supabase image_files table.

    Images are NOT downloaded or re-hosted — only the CDN URL and license
    are stored. This respects Creative Commons per-image licensing and keeps
    storage costs at zero for this source.

    The license field in the database allows the app to:
      - Display proper attribution to the photographer
      - Filter out NC (Non-Commercial) images if the app is monetized

    Note: EOL deprecated API keys — no key is needed. The API is fully public.
    """
    from .fetcher import EOLImageFetcher
    from .supabase_store import SupabaseStore

    store   = SupabaseStore(role="ingest") if not dry_run else None
    fetcher = EOLImageFetcher(
        images_per_species=images_per_species,
        allow_commercial_only=allow_commercial_only,
    )

    total = 0
    with fetcher:
        for entry in species_list:
            common_name     = entry["common_name"]
            scientific_name = entry.get("scientific_name", common_name)

            images = fetcher.fetch(common_name, scientific_name)
            log.info(f"  {common_name}: {len(images)} images from EOL")

            if dry_run:
                for img in images:
                    print(f"    {img['eol_media_url'][:70]}… [{img['license']}]")
                continue

            species_id = store.get_species_id(common_name)
            if not species_id:
                log.warning(f"  [EOL Images] Species not in Supabase: {common_name!r} — run --text first")
                continue

            for img in images:
                store.upsert_remote_image_record(
                    species_id=species_id,
                    remote_url=img["eol_media_url"],
                    license=img["license"],
                    source="eol",
                )
                total += 1

    log.info(f"\n[EOL Images] Complete — {total} image URLs stored across {len(species_list)} species")


def run_ultralytics_ingest(
    dry_run: bool = False,
) -> None:
    """
    Download the Ultralytics African Wildlife Dataset (buffalo, elephant,
    rhino, zebra — ~248 labeled images) and upload to Supabase Storage.

    Note: run --text first so species rows exist in Supabase before images
    are uploaded (the uploader looks up species_id by common_name).
    """
    from .ultralytics_dl import UltralyticsDownloader
    from .supabase_store import SupabaseStore

    store      = SupabaseStore(role="ingest") if not dry_run else None
    downloader = UltralyticsDownloader()

    results = downloader.run(supabase_store=store, dry_run=dry_run)

    total = sum(results.values())
    log.info(f"\n[Ultralytics] Complete — {total} images uploaded across {len(results)} species")
    for name, count in results.items():
        log.info(f"  {name}: {count} images")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m safari_guide.data.ingest",
        description="Ingest wildlife data into Pinecone (vectors) and Supabase (documents + images).",
    )

    parser.add_argument("--text",             action="store_true", help="EOL + Wikipedia + API Ninjas + IUCN text → Supabase + Pinecone")
    parser.add_argument("--images",           action="store_true", help="All image sources (EOL URLs + LILA BC + Ultralytics)")
    parser.add_argument("--eol-images-only",  action="store_true", help="EOL image URLs only → Supabase (no download)")
    parser.add_argument("--lila-only",        action="store_true", help="LILA BC images only → Supabase Storage")
    parser.add_argument("--ultralytics-only", action="store_true", help="Ultralytics images only → Supabase Storage")
    parser.add_argument("--all",              action="store_true", help="Run everything: text + all image sources")

    parser.add_argument("--force",              action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--dry-run",            action="store_true", help="Show what would be done; no writes")
    parser.add_argument("--commercial-only",    action="store_true", help="Skip NC-licensed EOL images (use if monetizing)")
    parser.add_argument(
        "--n-images", type=int,
        default=int(os.getenv("LILA_IMAGES_PER_SPECIES", "20")),
        help="Images per species from LILA BC (default: 20)",
    )
    parser.add_argument(
        "--eol-images", type=int,
        default=int(os.getenv("EOL_IMAGES_PER_SPECIES", "10")),
        help="Images per species from EOL (default: 10)",
    )

    args = parser.parse_args()

    run_text        = args.text or args.all
    run_eol_images  = args.images or args.eol_images_only or args.all
    run_lila        = args.images or args.lila_only or args.all
    run_ultralytics = args.images or args.ultralytics_only or args.all

    if not any([run_text, run_eol_images, run_lila, run_ultralytics]):
        parser.print_help()
        sys.exit(0)

    if args.dry_run:
        log.info("[Ingest] DRY RUN — no writes will be made")

    species_list = _load_species_list()
    log.info(f"[Ingest] Loaded {len(species_list)} species from species_list.json")

    if run_text:
        log.info("\n[Ingest] ── TEXT: EOL + Wikipedia + API Ninjas + IUCN ───")
        run_text_ingest(species_list, force=args.force, dry_run=args.dry_run)

    if run_eol_images:
        log.info("\n[Ingest] ── IMAGES: EOL (remote URLs, no download) ──────")
        commercial_only = args.commercial_only or os.getenv("EOL_COMMERCIAL_ONLY", "").lower() == "true"
        run_eol_image_ingest(
            species_list,
            images_per_species=args.eol_images,
            allow_commercial_only=commercial_only,
            dry_run=args.dry_run,
        )

    if run_lila:
        log.info("\n[Ingest] ── IMAGES: LILA BC / Snapshot Serengeti ────────")
        run_lila_ingest(species_list, n_per_species=args.n_images, dry_run=args.dry_run)

    if run_ultralytics:
        log.info("\n[Ingest] ── IMAGES: Ultralytics African Wildlife ─────────")
        run_ultralytics_ingest(dry_run=args.dry_run)

    log.info("\n[Ingest] Done.")


if __name__ == "__main__":
    main()
