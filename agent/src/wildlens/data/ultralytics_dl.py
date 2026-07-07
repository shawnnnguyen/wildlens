"""
Ultralytics African Wildlife Dataset downloader.

Dataset: https://docs.ultralytics.com/datasets/detect/african-wildlife/
License: AGPL-3.0
Classes: buffalo, elephant, rhino, zebra (4 classes, ~248 labeled images)

Download URL: https://ultralytics.com/assets/african-wildlife.zip
Format: YOLO (images/ + labels/ directories, .txt annotation per image)

Strategy
────────
1. Download the zip (~XX MB) to a temp directory.
2. Extract images from train/val/test splits.
3. Map Ultralytics class names → species_list.json common names.
4. Upload each image to Supabase Storage under wildlife/{species_slug}/.
5. Record in the image_files table with source='ultralytics'.
6. Delete the local temp files after upload.

The label .txt files are not used — we already know the species from the
directory structure (YOLO datasets group images by split, labels are in a
parallel labels/ folder with the same filename stem).  We use the annotation
file to read the class id and map it to the species name.
"""
from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path

import httpx

log = logging.getLogger("safari_guide.data.ultralytics_dl")

DATASET_URL = "https://ultralytics.com/assets/african-wildlife.zip"

# Ultralytics class id → our species_list.json common_name
# Class order defined in the dataset YAML:
#   0: buffalo, 1: elephant, 2: rhino, 3: zebra
_CLASS_MAP: dict[int, str] = {
    0: "African Buffalo",
    1: "African Elephant",
    2: "White Rhinoceros",   # dataset labels both species as 'rhino'
    3: "Plains Zebra",
}

# Reverse map: ultralytics folder/label name → common_name
_NAME_MAP: dict[str, str] = {
    "buffalo":  "African Buffalo",
    "elephant": "African Elephant",
    "rhino":    "White Rhinoceros",
    "zebra":    "Plains Zebra",
}


class UltralyticsDownloader:
    """
    Downloads and uploads the Ultralytics African Wildlife Dataset.

    Args:
        dataset_url: Override the default download URL.
        work_dir:    Directory for the downloaded zip + extracted files.
                     Defaults to a system temp directory.
    """

    def __init__(
        self,
        dataset_url: str = DATASET_URL,
        work_dir:    Path | None = None,
    ):
        self._url      = os.getenv("ULTRALYTICS_DATASET_URL", dataset_url)
        self._work_dir = work_dir or Path(tempfile.gettempdir()) / "ultralytics_african_wildlife"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._client   = httpx.Client(timeout=120, follow_redirects=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        supabase_store,          # SupabaseStore — avoid circular import
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Full pipeline: download → extract → upload → cleanup.

        Returns dict of {common_name: images_uploaded}.
        """
        zip_path = self._work_dir / "african-wildlife.zip"

        if not zip_path.exists():
            log.info(f"[Ultralytics] Downloading dataset from {self._url} …")
            self._download(zip_path)
        else:
            log.info(f"[Ultralytics] Using cached zip at {zip_path}")

        log.info("[Ultralytics] Extracting …")
        # The zip has no wrapping "african-wildlife/" directory — it extracts
        # images/ and labels/ straight into work_dir.
        extract_dir = self._work_dir
        if not (extract_dir / "images").exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        results: dict[str, int] = {}

        # Collect all images grouped by species using annotation labels
        image_species_pairs = self._collect_images(extract_dir)
        log.info(f"[Ultralytics] Found {len(image_species_pairs)} labeled images across {len(_CLASS_MAP)} species")

        if dry_run:
            counts: dict[str, int] = {}
            for _, common_name in image_species_pairs:
                counts[common_name] = counts.get(common_name, 0) + 1
            for name, count in counts.items():
                log.info(f"  [DRY RUN] {name}: {count} images would be uploaded")
            return counts

        for image_path, common_name in image_species_pairs:
            species_id = supabase_store.get_species_id(common_name)
            if not species_id:
                log.warning(f"  [Ultralytics] Species not in Supabase yet: {common_name!r} — skipping. Run --text first.")
                continue

            storage_path = supabase_store.upload_image(image_path, common_name)
            if storage_path:
                supabase_store.upsert_image_record(
                    species_id=species_id,
                    storage_path=storage_path,
                    source="ultralytics",
                )
                results[common_name] = results.get(common_name, 0) + 1

        # Cleanup extracted images/labels (zip kept for re-runs)
        import shutil
        for sub in ("images", "labels"):
            sub_dir = extract_dir / sub
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
        log.info("[Ultralytics] Extracted files cleaned up")

        for name, count in results.items():
            log.info(f"[Ultralytics] {name}: {count} images uploaded")

        return results

    def list_classes(self) -> dict[int, str]:
        """Return the class id → common_name mapping for inspection."""
        return dict(_CLASS_MAP)

    # ── Private ───────────────────────────────────────────────────────────────

    def _download(self, dest: Path) -> None:
        """Stream-download the dataset zip to dest."""
        with self._client.stream("GET", self._url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        log.info(f"  [Ultralytics] Download progress: {pct:.0f}%")
        log.info(f"[Ultralytics] Downloaded to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    def _collect_images(self, extract_dir: Path) -> list[tuple[Path, str]]:
        """
        Walk the extracted dataset and pair each image with its species name.

        YOLO dataset structure:
          african-wildlife/
            train/
              images/  ← .jpg files
              labels/  ← .txt files (one annotation per line: class_id cx cy w h)
            val/
              images/
              labels/
            test/
              images/
              labels/

        We read the label file for each image to get the class_id, then map to
        common_name via _CLASS_MAP. Images with multiple classes (rare) use the
        first annotation line.
        """
        pairs: list[tuple[Path, str]] = []

        for split in ("train", "val", "test"):
            images_dir = extract_dir / split / "images"
            labels_dir = extract_dir / split / "labels"

            if not images_dir.exists():
                # Some zips use a flat structure — try alternative layout
                images_dir = extract_dir / "images" / split
                labels_dir = extract_dir / "labels" / split

            if not images_dir.exists():
                log.debug(f"  [Ultralytics] Split directory not found: {split} — skipping")
                continue

            for img_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
                label_path = labels_dir / (img_path.stem + ".txt")

                common_name = self._read_class_from_label(label_path)
                if common_name:
                    pairs.append((img_path, common_name))

        return pairs

    def _read_class_from_label(self, label_path: Path) -> str | None:
        """
        Read the first annotation from a YOLO label file.
        Returns the common_name for the class, or None if unreadable.

        YOLO label format (one object per line):
          class_id center_x center_y width height
        """
        if not label_path.exists():
            return None
        try:
            first_line = label_path.read_text(encoding="utf-8").strip().splitlines()[0]
            class_id   = int(first_line.split()[0])
            return _CLASS_MAP.get(class_id)
        except (IndexError, ValueError, OSError):
            return None

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
