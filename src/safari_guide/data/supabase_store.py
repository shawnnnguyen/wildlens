"""
Supabase read/write adapter for the Safari Guide data layer.

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

    Env vars required:
      SUPABASE_URL  — https://<project-ref>.supabase.co
      SUPABASE_KEY  — anon/service key (service key for server-side ingestion)
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_KEY must be set.\n"
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
