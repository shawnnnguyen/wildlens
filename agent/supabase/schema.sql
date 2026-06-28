create table if not exists species (
  id              serial primary key,
  common_name     text not null unique,
  scientific_name text,
  threat_level    text check (threat_level in ('low', 'medium', 'high')),
  safety_notes    text,
  created_at      timestamptz default now()
);

create table if not exists documents (
  id          serial primary key,
  species_id  int references species(id) on delete cascade,
  section     text,
  content     text not null,
  source      text,
  created_at  timestamptz default now()
);

create index if not exists idx_documents_species_id on documents(species_id);

-- image_files stores both:
--   * uploaded files  (LILA BC, Ultralytics) — storage_path is set, remote_url is null
--   * remote URLs     (EOL images)           — remote_url is set, storage_path is null
-- At least one of storage_path / remote_url must be non-null (enforced by check constraint).
create table if not exists image_files (
  id            serial primary key,
  species_id    int references species(id) on delete cascade,
  source        text,                   -- 'lila_bc' | 'ultralytics' | 'eol'
  storage_path  text unique,            -- path inside 'wildlife-images' Supabase Storage bucket
  remote_url    text,                   -- direct image URL (EOL CDN link)
  license       text,                   -- e.g. 'http://creativecommons.org/licenses/by/4.0/'
  created_at    timestamptz default now(),
  constraint image_has_location check (
    storage_path is not null or remote_url is not null
  )
);

create index if not exists idx_image_files_species_id on image_files(species_id);
create index if not exists idx_image_files_source     on image_files(source);

-- Grant service_role full access (required for server-side ingestion via service key)
grant select, insert, update, delete on public.species     to service_role;
grant select, insert, update, delete on public.documents   to service_role;
grant select, insert, update, delete on public.image_files to service_role;
grant usage, select on all sequences in schema public to service_role;
