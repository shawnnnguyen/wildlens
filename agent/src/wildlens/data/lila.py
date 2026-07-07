"""
LILA BC image downloader for Snapshot Serengeti dataset.

Downloads representative camera-trap images per species to populate the
Supabase Storage 'wildlife-images' bucket.

Strategy
────────
1. Stream-parse the Snapshot Serengeti COCO metadata JSON with ijson
   (avoids loading ~500 MB into RAM all at once).
2. Build a mapping of species_name → list[image_url].
3. Sample N images per species spread across different annotation sequences
   for visual diversity (different lighting, angles, distances).
4. Download sampled images to a temp directory.
5. Upload each image to Supabase Storage via SupabaseStore.
6. Delete temp files after upload.

Usage (via ingest.py):
    downloader = LILADownloader()
    downloader.run(species_list, n_per_species=20)
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Generator

import httpx

log = logging.getLogger("safari_guide.data.lila")

# Snapshot Serengeti v2.1 — COCO-format metadata
# LILA BC migrated off lilablobssc.blob.core.windows.net (now 403s); dataset is
# hosted on Google Cloud Storage as of this writing. The combined metadata file
# is now zip-compressed (unzipped it's ~500 MB of JSON).
SNAPSHOT_SERENGETI_METADATA_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2_1.json.zip"
)
SNAPSHOT_SERENGETI_IMAGE_BASE = (
    "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped/"
)

# iNaturalist Camera Traps (broader species coverage)
INAT_METADATA_URL = (
    "https://lilablobssc.blob.core.windows.net/"
    "inat-camera-traps/inat_camera_traps_eccv_2020.json"
)


class LILADownloader:
    """
    Downloads representative LILA BC images per species.

    Args:
        metadata_url:     COCO metadata JSON URL (defaults to Snapshot Serengeti).
        image_base_url:   Base URL prepended to relative image file_names in metadata.
        cache_dir:        Local directory for temp image storage during upload.
    """

    def __init__(
        self,
        metadata_url:   str = SNAPSHOT_SERENGETI_METADATA_URL,
        image_base_url: str = SNAPSHOT_SERENGETI_IMAGE_BASE,
        cache_dir:      Path | None = None,
    ):
        self._metadata_url   = os.getenv("LILA_METADATA_URL", metadata_url)
        self._image_base_url = image_base_url
        self._cache_dir      = cache_dir or Path(tempfile.gettempdir()) / "lila_images"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(timeout=60, follow_redirects=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        species_list:     list[dict],
        supabase_store,                # SupabaseStore — avoid circular import
        n_per_species:    int = 20,
        dry_run:          bool = False,
    ) -> dict[str, int]:
        """
        Full pipeline: metadata → sample → download → upload → cleanup.

        Returns dict of {common_name: images_uploaded}.
        """
        target_names = {
            s["scientific_name"].lower(): s["common_name"]
            for s in species_list
            if s.get("scientific_name")
        }
        log.info(f"[LILA] Targeting {len(target_names)} species from {self._metadata_url}")

        # Build species → image_url mapping by stream-parsing metadata
        species_images = self._build_species_image_map(target_names)
        log.info(f"[LILA] Found images for {len(species_images)} / {len(target_names)} target species")

        results: dict[str, int] = {}
        for sci_name_lower, image_urls in species_images.items():
            common_name = target_names[sci_name_lower]
            sample      = self._sample(image_urls, n_per_species)
            log.info(f"[LILA] {common_name}: {len(sample)} images sampled from {len(image_urls)} total")

            if dry_run:
                results[common_name] = len(sample)
                continue

            uploaded = 0
            for url in sample:
                local_path = self._download_image(url)
                if local_path is None:
                    continue
                try:
                    storage_path = supabase_store.upload_image(local_path, common_name)
                    species_id   = supabase_store.get_species_id(common_name)
                    if species_id and storage_path:
                        supabase_store.upsert_image_record(
                            species_id=species_id,
                            storage_path=storage_path,
                            source="lila_bc",
                        )
                        uploaded += 1
                except Exception as exc:
                    log.warning(f"  [LILA] Upload failed for {url}: {exc}")
                finally:
                    if local_path and local_path.exists():
                        local_path.unlink()

            results[common_name] = uploaded
            log.info(f"[LILA] {common_name}: {uploaded} images uploaded")

        return results

    def sample_urls(self, scientific_name: str, n: int = 20) -> list[str]:
        """
        Return up to n image URLs for a single species without downloading.
        Useful for inspection or dry-run previews.
        """
        target = {scientific_name.lower(): scientific_name}
        mapping = self._build_species_image_map(target)
        urls = mapping.get(scientific_name.lower(), [])
        return self._sample(urls, n)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_species_image_map(
        self, target_names: dict[str, str]
    ) -> dict[str, list[str]]:
        """
        Stream-parse COCO metadata JSON with ijson to avoid loading ~500 MB at once.

        COCO format structure:
          {
            "images":      [{"id": int, "file_name": str, ...}, ...],
            "annotations": [{"image_id": int, "category_id": int, ...}, ...],
            "categories":  [{"id": int, "name": str, ...}, ...]
          }

        Returns {scientific_name_lower: [url, ...]}
        """
        try:
            import ijson
        except ImportError:
            log.error("[LILA] ijson is required for stream parsing. Install with: pip install ijson")
            return {}

        log.info(f"[LILA] Stream-parsing metadata from {self._metadata_url} …")

        # Phase 1: collect categories that match target species
        category_map: dict[int, str] = {}   # category_id → sci_name_lower
        image_url_map: dict[int, str] = {}  # image_id → full URL
        image_categories: dict[int, int] = {}  # image_id → category_id

        try:
            with self._client.stream("GET", self._metadata_url) as response:
                response.raise_for_status()
                raw_bytes = response.read()
        except httpx.HTTPError as exc:
            log.error(f"[LILA] Failed to fetch metadata: {exc}")
            return {}

        import io

        if self._metadata_url.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                raw_bytes = zf.read(zf.namelist()[0])

        # The dataset embeds non-standard NaN/Infinity literals (e.g. annotation
        # "count"/"standing" fields) which are invalid JSON and reject ijson —
        # neutralize them since we never read those fields.
        import re
        raw_bytes = re.sub(rb"\b(NaN|-?Infinity)\b", b"null", raw_bytes)

        # Pass 1: categories
        for item in ijson.items(io.BytesIO(raw_bytes), "categories.item"):
            name_lower = item.get("name", "").lower()
            for sci_lower in target_names:
                if sci_lower in name_lower or name_lower in sci_lower:
                    category_map[item["id"]] = sci_lower
                    break

        log.info(f"[LILA] Matched {len(category_map)} categories")
        if not category_map:
            return {}

        # Pass 2: images
        for item in ijson.items(io.BytesIO(raw_bytes), "images.item"):
            file_name = item.get("file_name", "")
            if file_name:
                url = (
                    file_name if file_name.startswith("http")
                    else self._image_base_url.rstrip("/") + "/" + file_name
                )
                image_url_map[item["id"]] = url

        # Pass 3: annotations → map image_id → category_id
        for item in ijson.items(io.BytesIO(raw_bytes), "annotations.item"):
            cat_id = item.get("category_id")
            img_id = item.get("image_id")
            if cat_id in category_map and img_id is not None:
                image_categories[img_id] = cat_id

        # Build result
        result: dict[str, list[str]] = {}
        for img_id, cat_id in image_categories.items():
            sci_lower = category_map[cat_id]
            url = image_url_map.get(img_id)
            if url:
                result.setdefault(sci_lower, []).append(url)

        return result

    def _sample(self, urls: list[str], n: int) -> list[str]:
        """
        Select n URLs spread evenly across the full list for diversity.
        If len(urls) <= n, return all.
        """
        if len(urls) <= n:
            return urls
        step = len(urls) // n
        return [urls[i * step] for i in range(n)]

    def _download_image(self, url: str) -> Path | None:
        """Download a single image to the cache dir. Returns local path or None on failure."""
        filename = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        dest     = self._cache_dir / filename

        if dest.exists():
            return dest

        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            time.sleep(0.1)  # polite delay
            return dest
        except httpx.HTTPError as exc:
            log.warning(f"  [LILA] Download failed: {url} — {exc}")
            return None

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
