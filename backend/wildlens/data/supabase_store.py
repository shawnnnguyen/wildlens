"""
Supabase read/write adapter for the WildLens data layer.

Responsibilities
────────────────
Write path (used by ingest.py):
  - upsert_species()        insert/update row in `species` table
  - upsert_document()       insert one document chunk into `documents` table
  - upload_image()          upload image bytes to `wildlife-images` Storage bucket
  - upsert_image_record()   insert metadata row into `image_files` table

Read path (used by rag.py at app startup):
  - load_all_documents()    SELECT all chunks → list[langchain Document]
                            Used to rebuild BM25 retriever in-memory
  - list_image_files()      SELECT image_files joined to species → golden-set
                            image source for eval/seed_dataset.py (role="ingest")
  - public_image_url()      module-level helper, no client needed — public
                            Storage URL for a given storage_path

Prerequisites (run once in Supabase dashboard SQL editor):

    create table species (
      id              serial primary key,
      common_name     text not null unique,
      scientific_name text,
      threat_level    text check (threat_level in ('low','medium','high')),
      safety_notes    text,
      created_at      timestamptz default now()
    );

    create table documents (
      id          serial primary key,
      species_id  int references species(id) on delete cascade,
      section     text,
      content     text not null,
      source      text,
      created_at  timestamptz default now()
    );

    create table image_files (
      id            serial primary key,
      species_id    int references species(id) on delete cascade,
      source        text,
      storage_path  text not null,
      created_at    timestamptz default now()
    );

Storage bucket: create a public bucket named 'wildlife-images' in the
Supabase dashboard (Storage → New bucket).
"""
from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path

from langchain_core.documents import Document

log = logging.getLogger("safari_guide.data.supabase_store")

_BUCKET = "wildlife-images"


class SupabaseStore:
    """
    Thin wrapper around the Supabase Python client.

    Two callers, two very different privilege needs — split by the `role`
    param rather than sharing one service-role key everywhere:

      role="ingest"  (data/ingest.py, offline/CLI only) — needs full
          read+write on species, documents, image_files, and the
          wildlife-images Storage bucket. Uses SUPABASE_INGEST_KEY.

      role="runtime" (default; rag/factory.py's BM25 corpus load, and
          ranking.py's enrich_async() write-back cache) — only ever needs
          SELECT on species/documents and INSERT/UPDATE on documents (never
          image_files or Storage, never DELETE). Uses SUPABASE_RUNTIME_KEY.
          Point this at a Supabase Postgres role scoped to exactly that via
          RLS, e.g.:
              create policy "runtime_read_species" on species
                for select using (true);
              create policy "runtime_read_documents" on documents
                for select using (true);
              create policy "runtime_write_documents" on documents
                for insert, update using (true) with check (true);
          (No policy on image_files/storage.objects for this role — RLS
          default-denies anything without a matching policy.)

    Both fall back to SUPABASE_KEY (a single service-role key for both roles)
    if the role-specific var isn't set, so today's single-key setup keeps
    working unconfigured — this is additive, not a breaking change.

    Env vars:
      SUPABASE_URL          — https://<project-ref>.supabase.co
      SUPABASE_INGEST_KEY   — service-role key, ingest.py only
      SUPABASE_RUNTIME_KEY  — least-privilege key, the running agent
      SUPABASE_KEY          — fallback used for either role if the
                               role-specific var above isn't set
    """

    def __init__(self, role: str = "runtime"):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv(f"SUPABASE_{role.upper()}_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise EnvironmentError(
                f"SUPABASE_URL and SUPABASE_{role.upper()}_KEY (or SUPABASE_KEY) must be set.\n"
                "Get them from your Supabase project dashboard → Settings → API."
            )
        from supabase import create_client
        self._sb = create_client(url, key)

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE PATH
    # ══════════════════════════════════════════════════════════════════════════

    def upsert_species(self, species_entry: dict) -> int:
        """
        Insert or update a species row.
        Returns the species row id.
        """
        payload = {
            "common_name":     species_entry["common_name"],
            "scientific_name": species_entry.get("scientific_name", ""),
            "threat_level":    species_entry.get("threat_level", "low"),
            "safety_notes":    species_entry.get("safety_notes", ""),
        }
        resp = (
            self._sb.table("species")
            .upsert(payload, on_conflict="common_name")
            .execute()
        )
        row = resp.data[0] if resp.data else {}
        species_id = row.get("id")
        if not species_id:
            # Fetch after upsert in case upsert did not return id
            fetch = (
                self._sb.table("species")
                .select("id")
                .eq("common_name", species_entry["common_name"])
                .single()
                .execute()
            )
            species_id = fetch.data["id"]

        log.debug(f"  [Supabase] Upserted species '{species_entry['common_name']}' → id={species_id}")
        return species_id

    def upsert_document(
        self,
        species_id: int,
        section:    str,
        content:    str,
        source:     str,
    ) -> None:
        """
        Insert a document chunk.
        Uses (species_id, section, source) as a logical composite key —
        if the same section already exists for the species, it is replaced.
        """
        # Delete existing chunk for this (species_id, section, source) first
        (
            self._sb.table("documents")
            .delete()
            .eq("species_id", species_id)
            .eq("section", section)
            .eq("source", source)
            .execute()
        )
        payload = {
            "species_id": species_id,
            "section":    section,
            "content":    content,
            "source":     source,
        }
        self._sb.table("documents").insert(payload).execute()
        log.debug(f"  [Supabase] Inserted document: species_id={species_id} section={section}")

    def upload_image(self, local_path: Path, common_name: str) -> str | None:
        """
        Upload an image file to the 'wildlife-images' Storage bucket.

        Storage path: wildlife/{common_name_slug}/{filename}
        Returns the storage_path string, or None on failure.
        """
        slug     = common_name.lower().replace(" ", "_")
        filename = local_path.name
        dest     = f"wildlife/{slug}/{filename}"

        mime, _ = mimetypes.guess_type(str(local_path))
        mime     = mime or "image/jpeg"

        try:
            with open(local_path, "rb") as fh:
                image_bytes = fh.read()
            self._sb.storage.from_(_BUCKET).upload(
                path=dest,
                file=image_bytes,
                file_options={"content-type": mime},
            )
            log.debug(f"  [Supabase] Uploaded image: {dest}")
            return dest
        except Exception as exc:
            # Duplicate upload raises an error — treat as success if file already exists
            if "already exists" in str(exc).lower() or "duplicate" in str(exc).lower():
                log.debug(f"  [Supabase] Image already exists: {dest}")
                return dest
            log.warning(f"  [Supabase] Image upload failed: {dest} — {exc}")
            return None

    def upsert_image_record(
        self,
        species_id:   int,
        storage_path: str,
        source:       str,
    ) -> None:
        """Insert a row into image_files for a Supabase Storage upload; skip if already recorded."""
        existing = (
            self._sb.table("image_files")
            .select("id")
            .eq("storage_path", storage_path)
            .execute()
        )
        if existing.data:
            return
        payload = {
            "species_id":   species_id,
            "storage_path": storage_path,
            "source":       source,
        }
        self._sb.table("image_files").insert(payload).execute()

    def upsert_remote_image_record(
        self,
        species_id:  int,
        remote_url:  str,
        license:     str,
        source:      str = "eol",
    ) -> None:
        """
        Insert a row into image_files for a remote URL (e.g. EOL CDN image).
        Images are not re-hosted — only the URL and license are stored.
        Skips silently if the remote_url is already recorded.
        """
        existing = (
            self._sb.table("image_files")
            .select("id")
            .eq("remote_url", remote_url)
            .execute()
        )
        if existing.data:
            return
        payload = {
            "species_id": species_id,
            "remote_url": remote_url,
            "license":    license,
            "source":     source,
        }
        self._sb.table("image_files").insert(payload).execute()
        log.debug(f"  [Supabase] Remote image record: {remote_url[:60]}… license={license}")

    # ══════════════════════════════════════════════════════════════════════════
    # READ PATH
    # ══════════════════════════════════════════════════════════════════════════

    def get_species_id(self, common_name: str) -> int | None:
        """Return the id of a species row by common_name, or None if not found."""
        resp = (
            self._sb.table("species")
            .select("id")
            .eq("common_name", common_name)
            .maybe_single()
            .execute()
        )
        return resp.data["id"] if resp.data else None

    def load_all_documents(self) -> list[Document]:
        """
        Fetch all document chunks from Supabase and return as LangChain Documents.

        Used by rag.py at startup to rebuild the BM25 retriever in-memory.
        Each document carries metadata so node_retrieve_information's
        [Source: species] formatting continues to work correctly.
        """
        log.info("[Supabase] Loading all documents for BM25 rebuild …")
        resp = (
            self._sb.table("documents")
            .select("content, section, source, species(common_name, threat_level)")
            .execute()
        )
        rows = resp.data or []
        documents: list[Document] = []
        for row in rows:
            species_info = row.get("species") or {}
            documents.append(Document(
                page_content=row["content"],
                metadata={
                    "species":      species_info.get("common_name", "Unknown"),
                    "section":      row.get("section", ""),
                    "source":       row.get("source", ""),
                    "threat_level": species_info.get("threat_level", "low"),
                },
            ))
        log.info(f"[Supabase] Loaded {len(documents)} document chunks")
        return documents

    def document_count(self) -> int:
        """Return total number of document chunks stored."""
        resp = self._sb.table("documents").select("id", count="exact").execute()
        return resp.count or 0

    def species_count(self) -> int:
        """Return total number of species rows."""
        resp = self._sb.table("species").select("id", count="exact").execute()
        return resp.count or 0

    def list_image_files(self, limit_per_species: int | None = None) -> list[dict]:
        """
        Return [{"common_name": ..., "storage_path": ...}, ...] for every
        Storage-backed image (used by eval/seed_dataset.py to build the golden
        eval set from images already ingested by data/lila.py).

        Only rows with a storage_path are returned — upsert_remote_image_record
        inserts EOL-URL rows into this same table with no storage_path, which
        aren't usable here (they aren't in the wildlife-images bucket at all).

        Requires role="ingest" — per this class's docstring, the runtime role
        has no RLS policy on image_files and will get an empty/denied result.
        """
        resp = (
            self._sb.table("image_files")
            .select("storage_path, species(common_name)")
            .not_.is_("storage_path", "null")
            .execute()
        )
        by_species: dict[str, list[str]] = {}
        for row in resp.data or []:
            common_name = (row.get("species") or {}).get("common_name")
            storage_path = row.get("storage_path")
            if not common_name or not storage_path:
                continue
            by_species.setdefault(common_name, []).append(storage_path)

        out: list[dict] = []
        for common_name, paths in by_species.items():
            selected = paths if limit_per_species is None else paths[:limit_per_species]
            out.extend({"common_name": common_name, "storage_path": p} for p in selected)
        return out


def public_image_url(storage_path: str) -> str:
    """
    Public URL for a wildlife-images Storage object — the bucket is public
    (see module docstring), so fetching an image needs only SUPABASE_URL, not
    a Supabase client/credentials. Used by the eval runner to download dataset
    item images: neither SUPABASE_INGEST_KEY nor SUPABASE_RUNTIME_KEY has read
    access to Storage (see SupabaseStore's role docstring), so a keyless
    public-URL fetch is the only path that works for both roles.
    """
    base = os.environ["SUPABASE_URL"].rstrip("/")
    return f"{base}/storage/v1/object/public/{_BUCKET}/{storage_path}"
