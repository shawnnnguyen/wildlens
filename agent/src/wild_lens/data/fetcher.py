"""
API clients for fetching real wildlife data from open-source sources.

EOLClient       — Encyclopedia of Life text (https://eol.org)
                  No auth required — fully public API, no key needed.
                  Rate limit ~150 req/min. Returns section-level text chunks.

EOLImageFetcher — Encyclopedia of Life images (https://eol.org)
                  Same public API, no key needed.
                  Workflow:
                    1. Search by name → page_id
                    2. Pages API with images_per_page → dataObjects
                    3. Filter dataType = StillImage
                    4. Return list of {eol_media_url, license, scientific_name}
                  EOL images are linked (URL stored), not re-hosted, to respect
                  the per-image Creative Commons licenses.

WikipediaClient  — Wikipedia article extracts (https://en.wikipedia.org)
                   No auth required. Rate limit generous (~200 req/s).
                   Fetches full plain-text article extract via MediaWiki API,
                   parses == Section == headers into a section dict.
                   Used to fill gaps where EOL returns no text for a section.
                   Priority in chunk builder: EOL > Wikipedia > fallback stub.

APINinjasClient  — API Ninjas Animals API (https://api.api-ninjas.com/v1/animals)
                   Requires API_NINJAS_KEY env var (free at api-ninjas.com).
                   Returns structured animal characteristics: weight, speed,
                   lifespan, diet type, prey, gestation, group behavior, etc.
                   Built into a dedicated 'characteristics' chunk per species.
                   Falls back gracefully (returns empty dict) if key absent.

IUCNClient       — IUCN Red List API v3 (https://apiv3.iucnredlist.org)
                  Requires free API key (IUCN_API_KEY env var).
                  Falls back to a mock stub if key is absent or API is
                  unreachable, so ingestion is never blocked on key registration.

All clients cache raw API responses as JSON files under data/cache/ so that
re-runs skip the network entirely for already-fetched species.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("safari_guide.data.fetcher")

# ── Cache directories (repo-root data/cache/) ────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parents[4]
_CACHE_EOL        = _REPO_ROOT / "data" / "cache" / "eol"
_CACHE_IUCN       = _REPO_ROOT / "data" / "cache" / "iucn"
_CACHE_WIKIPEDIA  = _REPO_ROOT / "data" / "cache" / "wikipedia"
_CACHE_API_NINJAS = _REPO_ROOT / "data" / "cache" / "api_ninjas"

# EOL TDWG subject URI → friendly section name
_EOL_SUBJECT_MAP: dict[str, str] = {
    "http://rs.tdwg.org/ontology/voc/SPMInfoItems#Description":       "overview",
    "http://rs.tdwg.org/ontology/voc/SPMInfoItems#Habitat":           "habitat",
    "http://rs.tdwg.org/ontology/voc/SPMInfoItems#Behavior":          "behavior",
    "http://rs.tdwg.org/ontology/voc/SPMInfoItems#TrophicStrategy":   "diet",
    "http://rs.tdwg.org/ontology/voc/SPMInfoItems#Evolution":         "evolution",
    "http://rs.tdwg.org/ontology/voc/SPMInfoItems#ConservationStatus":"conservation",
    "http://rs.tdwg.org/ontology/voc/SPMInfoItems#Associations":      "associations",
    # Abbreviated forms EOL sometimes returns
    "description":        "overview",
    "habitat":            "habitat",
    "behavior":           "behavior",
    "conservationstatus": "conservation",
}

class EOLClient:
    """
    Fetches species text from the Encyclopedia of Life API.

    Flow per species:
      1. Search by scientific name → resolve page_id
      2. Fetch page detail with text=true → parse dataObjects
      3. Group by TDWG subject URI → return dict[section → text]
    """

    SEARCH_URL = "https://eol.org/api/search/1.0.json"
    PAGE_URL   = "https://eol.org/api/pages/1.0/{page_id}.json"
    _DELAY     = 0.5  # seconds between requests — stay well under 150 req/min

    def __init__(self, api_key: str | None = None):
        self._key    = api_key or os.getenv("EOL_API_KEY", "")
        self._client = httpx.Client(timeout=30)
        _CACHE_EOL.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(self, common_name: str, scientific_name: str) -> dict[str, str]:
        """
        Return dict of {section: text} for the species.
        Returns an empty dict if the species cannot be found.
        Reads from cache if available.
        """
        cache_key  = scientific_name.lower().replace(" ", "_")
        cache_file = _CACHE_EOL / f"{cache_key}.json"

        if cache_file.exists():
            log.info(f"  [EOL] Cache hit: {scientific_name}")
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            log.info(f"  [EOL] Fetching: {scientific_name}")
            raw = self._fetch_raw(scientific_name)
            cache_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(self._DELAY)

        return self._parse_sections(raw)

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_raw(self, scientific_name: str) -> dict:
        """Search for species and return the raw page detail JSON."""
        # Step 1: search
        params: dict[str, Any] = {"q": scientific_name, "exact": False, "page": 1, "per_page": 1}
        if self._key:
            params["key"] = self._key

        try:
            resp = self._client.get(self.SEARCH_URL, params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except httpx.HTTPError as exc:
            log.warning(f"  [EOL] Search failed for {scientific_name!r}: {exc}")
            return {}

        if not results:
            log.warning(f"  [EOL] No results for {scientific_name!r}")
            return {}

        page_id = results[0].get("id")
        if not page_id:
            return {}

        # Step 2: page detail
        time.sleep(self._DELAY)
        page_params: dict[str, Any] = {
            "batch":         False,
            "id":            page_id,
            "text":          True,
            "images_per_page": 0,
            "videos_per_page": 0,
            "details":       True,
            "common_names":  False,
            "synonyms":      False,
            "references":    False,
        }
        if self._key:
            page_params["key"] = self._key

        try:
            url  = self.PAGE_URL.format(page_id=page_id)
            resp = self._client.get(url, params=page_params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            log.warning(f"  [EOL] Page fetch failed for page_id={page_id}: {exc}")
            return {}

    def _parse_sections(self, raw: dict) -> dict[str, str]:
        """
        Extract text from EOL dataObjects, grouped by TDWG subject URI.
        Only English, non-empty text objects are used.
        Multiple objects for the same section are joined with a space.
        """
        sections: dict[str, list[str]] = {}

        for data_obj in raw.get("dataObjects", []):
            if data_obj.get("dataType", "").lower() != "text":
                continue
            if data_obj.get("language", "en") not in ("en", "eng", ""):
                continue

            text = data_obj.get("description", "").strip()
            if not text:
                continue

            # Resolve subject → section name
            subject_uri  = data_obj.get("subject", "")
            section_name = _EOL_SUBJECT_MAP.get(subject_uri)
            if not section_name:
                # Try last path segment lowercased
                last = subject_uri.split("#")[-1].lower()
                section_name = _EOL_SUBJECT_MAP.get(last, last or "overview")

            sections.setdefault(section_name, []).append(text)

        # Join duplicates, strip HTML tags simply
        result: dict[str, str] = {}
        for section, texts in sections.items():
            joined = " ".join(texts)
            # Minimal HTML strip — remove <tag> patterns
            import re
            cleaned = re.sub(r"<[^>]+>", " ", joined)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                result[section] = cleaned

        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

# Characteristics fields we care about, in display order
_NINJAS_FIELDS: list[tuple[str, str]] = [
    ("top_speed",               "Top speed"),
    ("weight",                  "Weight"),
    ("height",                  "Height"),
    ("length",                  "Length"),
    ("lifespan",                "Lifespan"),
    ("color",                   "Color"),
    ("skin_type",               "Skin type"),
    ("diet",                    "Diet type"),
    ("prey",                    "Prey"),
    ("group_behavior",          "Group behavior"),
    ("lifestyle",               "Lifestyle"),
    ("habitat",                 "Habitat"),
    ("location",                "Location"),
    ("gestation_period",        "Gestation period"),
    ("average_litter_size",     "Average litter size"),
    ("age_of_sexual_maturity",  "Age of sexual maturity"),
    ("estimated_population_size","Estimated population"),
    ("biggest_threat",          "Biggest threat"),
    ("most_distinctive_feature","Most distinctive feature"),
    ("name_of_young",           "Young called"),
]


class APINinjasClient:
    """
    Fetches structured animal facts from the API Ninjas Animals API.

    Endpoint: GET https://api.api-ninjas.com/v1/animals?name={name}
    Auth:     X-Api-Key header (free key at https://api-ninjas.com)
    Free tier: 10,000 requests/month, 1 req/sec

    Returns a flat dict of characteristics that gets built into a dedicated
    'characteristics' document chunk per species. This chunk is distinct from
    EOL/Wikipedia sections and adds precise factual data (speed, weight,
    lifespan, prey list, gestation period) that narrative sources often omit.

    Falls back silently (returns {}) if API_NINJAS_KEY is not set.
    """

    API_URL = "https://api.api-ninjas.com/v1/animals"
    _DELAY  = 1.1  # stay under 1 req/sec free-tier limit

    def __init__(self):
        self._key    = os.getenv("API_NINJAS_KEY", "")
        self._client = httpx.Client(timeout=30)
        _CACHE_API_NINJAS.mkdir(parents=True, exist_ok=True)

        if not self._key:
            log.warning(
                "[API Ninjas] API_NINJAS_KEY not set — characteristics data will be skipped. "
                "Get a free key at https://api-ninjas.com"
            )

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(self, common_name: str) -> dict:
        """
        Return a characteristics dict for the animal, e.g.:
          {
            "top_speed": "80 km/h",
            "weight": "120-249 kg",
            "lifespan": "10-14 years",
            "diet": "Carnivore",
            "prey": "Zebra, Wildebeest, Antelope",
            "group_behavior": "Social",
            "gestation_period": "110 days",
            "biggest_threat": "Habitat loss",
            ...
          }
        Returns {} if key not set, animal not found, or API fails.
        """
        if not self._key:
            return {}

        cache_key  = common_name.lower().replace(" ", "_")
        cache_file = _CACHE_API_NINJAS / f"{cache_key}.json"

        if cache_file.exists():
            log.info(f"  [API Ninjas] Cache hit: {common_name}")
            return json.loads(cache_file.read_text(encoding="utf-8"))

        log.info(f"  [API Ninjas] Fetching: {common_name}")
        result = self._fetch_live(common_name)
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(self._DELAY)
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_live(self, common_name: str) -> dict:
        """Query the API and return the best-matching animal's characteristics."""
        try:
            resp = self._client.get(
                self.API_URL,
                params={"name": common_name},
                headers={"X-Api-Key": self._key},
            )
            resp.raise_for_status()
            animals = resp.json()
        except httpx.HTTPError as exc:
            log.warning(f"  [API Ninjas] Request failed for {common_name!r}: {exc}")
            return {}

        if not animals:
            log.warning(f"  [API Ninjas] No results for {common_name!r}")
            return {}

        # Pick the closest name match
        target = common_name.lower()
        best   = min(
            animals,
            key=lambda a: abs(len(a.get("name", "").lower()) - len(target)),
        )

        characteristics = best.get("characteristics", {})
        taxonomy        = best.get("taxonomy", {})
        locations       = best.get("locations", [])

        # Flatten into a clean dict keeping only non-empty values
        result: dict = {}
        for field, _ in _NINJAS_FIELDS:
            val = characteristics.get(field, "").strip()
            if val:
                result[field] = val

        if locations:
            result["locations"] = ", ".join(locations)
        if taxonomy.get("scientific_name"):
            result["scientific_name"] = taxonomy["scientific_name"]

        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

class IUCNClient:
    """
    Fetches conservation data from the IUCN Red List API v3.

    Requires IUCN_API_KEY environment variable (free registration).
    Falls back to a static mock stub when:
      - IUCN_API_KEY is not set, OR
      - the API is unreachable (network error), OR
      - the species is not found

    This ensures text ingestion is never blocked on IUCN key availability.
    """

    BASE = "https://apiv3.iucnredlist.org/api/v3"
    _DELAY = 0.1  # well under 30 req/sec limit

    def __init__(self):
        self._key    = os.getenv("IUCN_API_KEY", "")
        self._client = httpx.Client(timeout=30)
        _CACHE_IUCN.mkdir(parents=True, exist_ok=True)

        if not self._key:
            log.warning(
                "[IUCN] IUCN_API_KEY not set — using mock stub for all species. "
                "Register at https://apiv3.iucnredlist.org/api/v3/token for real data."
            )

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(self, scientific_name: str) -> dict:
        """
        Return conservation dict:
          {
            'category': 'VU',
            'population_trend': 'Decreasing',
            'year_assessed': 2023,
            'threats': ['Habitat loss', ...],
            'habitats': ['Savanna', ...],
          }
        Returns mock data if key absent or API fails.
        """
        if not self._key:
            return self._mock_stub(scientific_name)

        cache_key  = scientific_name.lower().replace(" ", "_")
        cache_file = _CACHE_IUCN / f"{cache_key}.json"

        if cache_file.exists():
            log.info(f"  [IUCN] Cache hit: {scientific_name}")
            return json.loads(cache_file.read_text(encoding="utf-8"))

        log.info(f"  [IUCN] Fetching: {scientific_name}")
        result = self._fetch_live(scientific_name)
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_live(self, scientific_name: str) -> dict:
        name_enc = scientific_name.replace(" ", "%20")
        token    = self._key
        result   = {"category": "NE", "population_trend": "Unknown",
                    "year_assessed": None, "threats": [], "habitats": []}

        # Main assessment
        try:
            url  = f"{self.BASE}/species/{name_enc}?token={token}"
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
            assessments = data.get("result", [])
            if assessments:
                a = assessments[0]
                result["category"]         = a.get("category", "NE")
                result["population_trend"] = a.get("population_trend", "Unknown")
                result["year_assessed"]    = a.get("assessment_date", "")
            time.sleep(self._DELAY)
        except httpx.HTTPError as exc:
            log.warning(f"  [IUCN] Assessment fetch failed for {scientific_name!r}: {exc}")
            return result

        # Threats
        try:
            url  = f"{self.BASE}/threats/species/name/{name_enc}?token={token}"
            resp = self._client.get(url)
            resp.raise_for_status()
            result["threats"] = [
                t.get("title", "") for t in resp.json().get("result", [])
                if t.get("title")
            ]
            time.sleep(self._DELAY)
        except httpx.HTTPError as exc:
            log.warning(f"  [IUCN] Threats fetch failed for {scientific_name!r}: {exc}")

        # Habitats
        try:
            url  = f"{self.BASE}/habitats/species/name/{name_enc}?token={token}"
            resp = self._client.get(url)
            resp.raise_for_status()
            result["habitats"] = [
                h.get("habitat", "") for h in resp.json().get("result", [])
                if h.get("habitat")
            ]
            time.sleep(self._DELAY)
        except httpx.HTTPError as exc:
            log.warning(f"  [IUCN] Habitats fetch failed for {scientific_name!r}: {exc}")

        return result

    def _mock_stub(self, scientific_name: str) -> dict:
        """Static fallback used when IUCN_API_KEY is absent."""
        return {
            "category":         "NE",
            "population_trend": "Unknown",
            "year_assessed":    None,
            "threats":          [],
            "habitats":         [],
            "_mock":            True,
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

# Wikipedia section header → our section name
_WIKI_SECTION_MAP: dict[str, str] = {
    "description":            "overview",
    "appearance":             "overview",
    "physical description":   "overview",
    "characteristics":        "overview",
    "morphology":             "overview",
    "taxonomy":               "overview",
    "habitat":                "habitat",
    "distribution":           "habitat",
    "range":                  "habitat",
    "distribution and habitat": "habitat",
    "ecology":                "habitat",
    "behavior":               "behavior",
    "behaviour":              "behavior",
    "social behavior":        "behavior",
    "social structure":       "behavior",
    "communication":          "behavior",
    "diet":                   "diet",
    "feeding":                "diet",
    "food":                   "diet",
    "predation":              "diet",
    "reproduction":           "behavior",
    "breeding":               "behavior",
    "conservation":           "conservation",
    "conservation status":    "conservation",
    "status":                 "conservation",
    "threats":                "conservation",
    "human interaction":      "conservation",
    "relationship with humans": "conservation",
}


class WikipediaClient:
    """
    Fetches species article text from the Wikipedia MediaWiki API.

    No authentication required. No rate-limit key needed.

    Workflow per species:
      1. Try scientific name first; fall back to common name if no article found.
      2. Fetch full plain-text extract via MediaWiki action=query&prop=extracts.
      3. Parse == Section == headers to split the article into named sections.
      4. Map section headings → our standard section names via _WIKI_SECTION_MAP.
      5. Return dict[section → text], same shape as EOLClient.fetch().

    Used to fill gaps where EOL returned no text for a section.
    Priority in chunk builder: EOL > Wikipedia > handcrafted stub.
    """

    API_URL = "https://en.wikipedia.org/w/api.php"
    _DELAY  = 0.3

    def __init__(self):
        self._client = httpx.Client(timeout=30)
        _CACHE_WIKIPEDIA.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(self, common_name: str, scientific_name: str) -> dict[str, str]:
        """
        Return dict of {section: text} for the species.
        Tries scientific name first, falls back to common name.
        Returns {} if article not found or API fails.
        Uses disk cache to skip network on re-runs.
        """
        cache_key  = scientific_name.lower().replace(" ", "_")
        cache_file = _CACHE_WIKIPEDIA / f"{cache_key}.json"

        if cache_file.exists():
            log.info(f"  [Wikipedia] Cache hit: {scientific_name}")
            return json.loads(cache_file.read_text(encoding="utf-8"))

        log.info(f"  [Wikipedia] Fetching: {scientific_name}")
        result = self._fetch_sections(scientific_name) or self._fetch_sections(common_name)

        if not result:
            log.warning(f"  [Wikipedia] No article found for {scientific_name!r} or {common_name!r}")
            result = {}

        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(self._DELAY)
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_sections(self, title: str) -> dict[str, str] | None:
        """
        Fetch the full plain-text article extract for title.
        Returns parsed section dict, or None if article not found.
        """
        params = {
            "action":       "query",
            "prop":         "extracts",
            "titles":       title,
            "format":       "json",
            "explaintext":  True,       # plain text, no HTML
            "exsectionformat": "plain", # section headers as == Header ==
            "redirects":    True,       # follow redirects (e.g. "Lion" → "Lion (animal)")
        }
        try:
            resp = self._client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data  = resp.json()
            pages = data.get("query", {}).get("pages", {})
        except httpx.HTTPError as exc:
            log.warning(f"  [Wikipedia] Request failed for {title!r}: {exc}")
            return None

        # MediaWiki returns a dict keyed by page id; -1 = not found
        for page_id, page in pages.items():
            if page_id == "-1" or "missing" in page:
                return None
            extract = page.get("extract", "").strip()
            if not extract:
                return None
            return self._parse_sections(extract)

        return None

    def _parse_sections(self, extract: str) -> dict[str, str]:
        """
        Split a MediaWiki plain-text extract into named sections.

        Format produced by exsectionformat=plain:
          Lead paragraph text (no header)

          == Section Name ==
          Section content...

          === Sub-section ===
          Sub-section content...

        Sub-sections (===) are merged into their parent section (==).
        The lead text (before the first ==) becomes the 'overview' section.
        Only sections matching _WIKI_SECTION_MAP are kept.
        """
        import re

        sections: dict[str, list[str]] = {}
        current_mapped = "overview"

        # Split on any == ... == header (level 2 or 3)
        parts = re.split(r"={2,3}\s*(.+?)\s*={2,3}", extract)

        # parts alternates: [text_before_first_header, header1, text1, header2, text2, ...]
        # First element is the lead paragraph
        if parts:
            lead = parts[0].strip()
            # Trim excessive whitespace runs
            lead = re.sub(r"\n{3,}", "\n\n", lead)
            if lead:
                sections.setdefault("overview", []).append(lead)

        it = iter(parts[1:])
        for header, text in zip(it, it):
            mapped = _WIKI_SECTION_MAP.get(header.strip().lower())
            if mapped:
                current_mapped = mapped
            # Always append to last known mapped section so sub-sections
            # (===) that don't match a key still attach to their parent
            content = re.sub(r"\n{3,}", "\n\n", text.strip())
            if content and mapped:
                sections.setdefault(mapped, []).append(content)

        return {k: "\n\n".join(v) for k, v in sections.items() if v}

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

# EOL dataType URI for still images
_EOL_STILL_IMAGE_TYPE = "http://purl.org/dc/dcmitype/StillImage"

# Licenses that allow commercial use — filter these in when allow_commercial=True
_COMMERCIAL_OK_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by/2.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "http://creativecommons.org/licenses/by-sa/3.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/publicdomain/mark/1.0/",
    "public domain",
}

_CACHE_EOL_IMAGES = _REPO_ROOT / "data" / "cache" / "eol_images"


class EOLImageFetcher:
    """
    Fetches image URLs from the Encyclopedia of Life API.

    No API key required — EOL deprecated key-based auth; the API is fully public.

    Workflow per species:
      1. Search by scientific name → resolve page_id (same as EOLClient)
      2. Pages API: GET /pages/1.0/{page_id}.json?images_per_page=N&details=true
      3. Filter dataObjects where dataType == StillImage URI
      4. Return list of image dicts with URL, license, and scientific name

    Images are stored as remote URLs in Supabase (not re-hosted) to respect
    individual Creative Commons licenses. The license field lets the app
    display attribution and filter out non-commercial-only images if needed.
    """

    SEARCH_URL = "https://eol.org/api/search/1.0.json"
    PAGE_URL   = "https://eol.org/api/pages/1.0/{page_id}.json"
    _DELAY     = 0.5

    def __init__(self, images_per_species: int = 10, allow_commercial_only: bool = False):
        """
        Args:
            images_per_species:    Max images to fetch per species from EOL.
            allow_commercial_only: If True, skip NC (Non-Commercial) licensed images.
                                   Set True if you plan to monetize the app.
        """
        self._n               = images_per_species
        self._commercial_only = allow_commercial_only
        self._client          = httpx.Client(timeout=30)
        _CACHE_EOL_IMAGES.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def fetch(self, common_name: str, scientific_name: str) -> list[dict]:
        """
        Return a list of image dicts for the species:
          [
            {
              "eol_media_url":   "https://content.eol.org/data/media/…/…jpg",
              "license":         "http://creativecommons.org/licenses/by/4.0/",
              "scientific_name": "Panthera leo",
              "common_name":     "African Lion",
              "source":          "eol",
            },
            ...
          ]
        Returns empty list if no images found or API fails.
        Uses cache to avoid repeat API calls on re-runs.
        """
        cache_key  = scientific_name.lower().replace(" ", "_")
        cache_file = _CACHE_EOL_IMAGES / f"{cache_key}.json"

        if cache_file.exists():
            log.info(f"  [EOL Images] Cache hit: {scientific_name}")
            return json.loads(cache_file.read_text(encoding="utf-8"))

        log.info(f"  [EOL Images] Fetching images: {scientific_name}")
        results = self._fetch_images(common_name, scientific_name)
        cache_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_images(self, common_name: str, scientific_name: str) -> list[dict]:
        """Search EOL for the species, then fetch image dataObjects."""
        # Step 1: search → page_id
        try:
            resp = self._client.get(
                self.SEARCH_URL,
                params={"q": scientific_name, "exact": False, "page": 1, "per_page": 1},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except httpx.HTTPError as exc:
            log.warning(f"  [EOL Images] Search failed for {scientific_name!r}: {exc}")
            return []

        if not results:
            log.warning(f"  [EOL Images] No results for {scientific_name!r}")
            return []

        page_id = results[0].get("id")
        if not page_id:
            return []

        time.sleep(self._DELAY)

        # Step 2: pages API with images_per_page
        try:
            url  = self.PAGE_URL.format(page_id=page_id)
            resp = self._client.get(url, params={
                "id":              page_id,
                "images_per_page": self._n,
                "images_page":     1,
                "details":         True,
                "text":            False,
                "videos_per_page": 0,
                "common_names":    False,
                "synonyms":        False,
                "references":      False,
                "batch":           False,
            })
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            log.warning(f"  [EOL Images] Pages fetch failed for {scientific_name!r}: {exc}")
            return []

        time.sleep(self._DELAY)

        # Step 3: filter StillImage dataObjects
        images: list[dict] = []
        for obj in data.get("dataObjects", []):
            data_type = obj.get("dataType", "")
            if data_type != _EOL_STILL_IMAGE_TYPE:
                continue

            media_url = obj.get("eolMediaURL") or obj.get("mediaURL", "")
            if not media_url:
                continue

            license_url = obj.get("license", "").lower()

            # Skip NC licenses if commercial-only mode
            if self._commercial_only and "nc" in license_url:
                log.debug(f"  [EOL Images] Skipping NC license: {license_url}")
                continue

            images.append({
                "eol_media_url":   media_url,
                "license":         obj.get("license", "unknown"),
                "scientific_name": obj.get("scientificName", scientific_name),
                "common_name":     common_name,
                "source":          "eol",
            })

            if len(images) >= self._n:
                break

        log.info(f"  [EOL Images] {len(images)} images found for {scientific_name!r}")
        return images

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

def build_species_chunks(
    species_entry:     dict,
    eol_sections:      dict[str, str],
    iucn_data:         dict,
    wiki_sections:     dict[str, str] | None = None,
    api_ninjas_data:   dict | None = None,
) -> list[dict]:
    """
    Merge EOL + Wikipedia + API Ninjas + IUCN + handcrafted safety notes into
    document chunks ready for Supabase insertion and Pinecone embedding.

    Source priority per section (highest wins):
      EOL > Wikipedia > handcrafted stub

    Dedicated chunks (always their own entry, never merged):
      API Ninjas → 'characteristics' chunk with structured animal facts
      IUCN       → 'conservation' chunk (authoritative Red List data)
      Safety     → 'safety' chunk (handcrafted, always present)

    Each chunk dict:
      {
        'section': str,   # 'overview' | 'habitat' | 'behavior' | 'diet' |
                          # 'characteristics' | 'conservation' | 'safety' | ...
        'content': str,
        'source':  str,   # 'eol' | 'wikipedia' | 'api_ninjas' | 'iucn' | 'handcrafted'
      }
    """
    common_name     = species_entry["common_name"]
    scientific_name = species_entry.get("scientific_name", "")
    safety_notes    = species_entry.get("safety_notes", "")
    wiki_sections   = wiki_sections or {}
    api_ninjas_data = api_ninjas_data or {}
    chunks: list[dict] = []

    section_priority = [
        "overview", "habitat", "behavior", "diet",
        "conservation", "associations", "evolution",
    ]

    # ── EOL sections (highest priority) ──────────────────────────────────────
    written_sections: set[str] = set()

    for section in section_priority:
        text = eol_sections.get(section, "").strip()
        if text:
            chunks.append({"section": section, "content": text, "source": "eol"})
            written_sections.add(section)

    # Remaining EOL sections not in priority list
    for section, text in eol_sections.items():
        if section not in written_sections and text.strip():
            chunks.append({"section": section, "content": text.strip(), "source": "eol"})
            written_sections.add(section)

    # ── Wikipedia sections (fill gaps where EOL returned nothing) ─────────────
    for section in section_priority:
        if section in written_sections:
            continue   # EOL already covered this section
        text = wiki_sections.get(section, "").strip()
        if text:
            # Truncate to 1200 chars — Wikipedia articles can be very long
            if len(text) > 1200:
                text = text[:1200].rsplit(" ", 1)[0] + " …"
            chunks.append({"section": section, "content": text, "source": "wikipedia"})
            written_sections.add(section)

    # Remaining Wikipedia sections also not covered by EOL
    for section, text in wiki_sections.items():
        if section in written_sections or not text.strip():
            continue
        text = text.strip()
        if len(text) > 1200:
            text = text[:1200].rsplit(" ", 1)[0] + " …"
        chunks.append({"section": section, "content": text, "source": "wikipedia"})
        written_sections.add(section)

    # ── API Ninjas characteristics chunk ─────────────────────────────────────
    if api_ninjas_data:
        lines: list[str] = [f"{common_name} ({scientific_name}) — Animal Facts:"]
        for field, label in _NINJAS_FIELDS:
            val = api_ninjas_data.get(field, "")
            if val:
                lines.append(f"{label}: {val}.")
        if api_ninjas_data.get("locations"):
            lines.append(f"Found in: {api_ninjas_data['locations']}.")
        if len(lines) > 1:   # only write if we got actual fields, not just the header
            chunks.append({
                "section": "characteristics",
                "content": " ".join(lines),
                "source":  "api_ninjas",
            })

    # ── IUCN conservation chunk (always its own chunk — authoritative source) ─
    if iucn_data.get("category") not in ("NE", None) or iucn_data.get("threats"):
        cat      = iucn_data.get("category", "NE")
        trend    = iucn_data.get("population_trend", "Unknown")
        year     = iucn_data.get("year_assessed") or ""
        threats  = "; ".join(iucn_data.get("threats", [])[:5])
        habitats = "; ".join(iucn_data.get("habitats", [])[:5])

        iucn_text = (
            f"{common_name} ({scientific_name}) — IUCN Red List Category: {cat}. "
            f"Population trend: {trend}."
        )
        if year:
            iucn_text += f" Last assessed: {year}."
        if threats:
            iucn_text += f" Key threats: {threats}."
        if habitats:
            iucn_text += f" Habitats: {habitats}."

        chunks.append({"section": "conservation", "content": iucn_text, "source": "iucn"})

    # ── Handcrafted safety chunk (always present) ─────────────────────────────
    if safety_notes:
        chunks.append({
            "section": "safety",
            "content": f"SAFETY — {common_name}: {safety_notes}",
            "source":  "handcrafted",
        })

    # ── Fallback stub if all sources returned nothing ─────────────────────────
    if not chunks or all(c["section"] in ("safety",) for c in chunks):
        stub = (
            f"{common_name} ({scientific_name}) is an African wildlife species. "
            f"Threat level: {species_entry.get('threat_level', 'unknown')}."
        )
        chunks.insert(0, {"section": "overview", "content": stub, "source": "handcrafted"})

    return chunks
