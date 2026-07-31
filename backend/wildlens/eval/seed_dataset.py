"""
Seeds the Langfuse golden dataset "wildlens-golden-v1" from images already
uploaded to the wildlife-images Supabase bucket during LILA ingestion
(data/lila.py), paired with expected species/threat_level from
species_list.json.

Idempotent: dataset items use a deterministic id (hash of storage_path), so
re-running this script upserts existing items rather than duplicating them
(see langfuse's create_dataset_item — upserts when `id` already exists).

Coverage of the full species_list.json roster by LILA's Snapshot Serengeti
sampling isn't guaranteed — this prints a per-species image count and flags
any species with zero images rather than silently seeding a lopsided set.

Usage:
    python -m wildlens.eval.seed_dataset [--limit-per-species N]

Env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, SUPABASE_URL,
SUPABASE_INGEST_KEY (or SUPABASE_KEY) — see SupabaseStore's role docstring;
the runtime key has no read access to image_files/Storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from ..data.species_lookup import lookup_species
from ..data.supabase_store import SupabaseStore
from ..logging_config import configure_logging
from ..observability import init_langfuse

log = logging.getLogger("wildlens.eval.seed_dataset")

DATASET_NAME = "wildlens-golden-v1"


def _item_id(storage_path: str) -> str:
    return "img-" + hashlib.sha256(storage_path.encode("utf-8")).hexdigest()[:16]


def _all_species_common_names() -> list[str]:
    path = Path(__file__).parent.parent / "data" / "species_list.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [entry["common_name"] for entry in entries]


def seed(limit_per_species: int | None = None) -> None:
    from langfuse import get_client

    if init_langfuse() is None:
        raise EnvironmentError(
            "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY must be set to seed the eval dataset."
        )
    client = get_client()

    store = SupabaseStore(role="ingest")
    images = store.list_image_files(limit_per_species=limit_per_species)

    counts: dict[str, int] = {}
    for row in images:
        counts[row["common_name"]] = counts.get(row["common_name"], 0) + 1
    log.info("Found %d images across %d species", len(images), len(counts))

    zero_coverage = [name for name in _all_species_common_names() if name not in counts]
    if zero_coverage:
        log.warning(
            "Zero image coverage for %d of %d species_list.json entries: %s",
            len(zero_coverage), len(_all_species_common_names()), ", ".join(sorted(zero_coverage)),
        )

    client.create_dataset(
        name=DATASET_NAME,
        description="Golden set: LILA-sourced photos paired with species_list.json ground truth.",
    )

    created = 0
    skipped = 0
    for row in images:
        entry = lookup_species(row["common_name"])
        if entry is None:
            skipped += 1
            continue
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=_item_id(row["storage_path"]),
            input=row["storage_path"],
            expected_output={"species": entry["common_name"], "threat_level": entry["threat_level"]},
            metadata={"common_name": entry["common_name"]},
        )
        created += 1

    client.flush()
    log.info("Seeded/updated %d dataset items in '%s' (%d skipped, no species_list.json match)",
              created, DATASET_NAME, skipped)


def main() -> None:
    load_dotenv()
    configure_logging()
    parser = argparse.ArgumentParser(description="Seed the Langfuse golden-dataset eval set.")
    parser.add_argument("--limit-per-species", type=int, default=None)
    args = parser.parse_args()
    seed(limit_per_species=args.limit_per_species)


if __name__ == "__main__":
    main()
