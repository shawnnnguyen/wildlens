// Mirrors backend/schemas.py 1:1.

export type ThreatLevel = "low" | "medium" | "high";

export interface WildlifeIdentification {
  species: string;
  confidence_score: number;
  visual_traits: string[];
  threat_level: ThreatLevel;
  habitat_context: string;
}

export interface ChatResponse {
  thread_id: string;
  session_secret: string | null; // only set on the turn that creates thread_id
  final_script: string;
  audio_url: string | null;
  identification: WildlifeIdentification | null;
  fallback_triggered: boolean;
  retrieved_facts: string[];
  error_message: string | null;
}

export type MessageRole = "human" | "ai";

export interface ChatMessageOut {
  role: MessageRole;
  content: string;
}

export interface SessionHistoryResponse {
  thread_id: string;
  messages: ChatMessageOut[];
  conversation_summary: string | null;
  identification_history: WildlifeIdentification[];
  total_turns: number;
}

export interface ErrorDetail {
  code: string;
  message: string;
  field?: string | null;
}

export interface ErrorResponse {
  error: ErrorDetail;
  thread_id?: string | null;
}

export interface AudioSynthesizeResponse {
  audio_url: string;
}

// ── Frontend-only UI state (not part of the backend contract) ────────────────

// `identification.species` arrives as "Common name (Scientific name)" — split
// once here so the UI never re-parses it.
export interface SpeciesCard {
  common: string;
  scientific: string;
  confidenceScore: number;
  threatLevel: ThreatLevel;
  habitatContext: string;
  visualTraits: string[];
  description: string;
}

export type UiMessage =
  | { id: string; kind: "image"; role: "human"; imageUrl: string }
  | { id: string; kind: "text"; role: "human" | "ai" | "error"; text: string; audioUrl?: string }
  | { id: string; kind: "card"; role: "ai"; card: SpeciesCard; audioUrl?: string };

export interface Session {
  id: string; // doubles as the backend thread_id, once identification has started
  secret: string; // capability token returned once at session creation; "" until then
  title: string;
  subtitle: string;
  thumbnail: string;
  species: SpeciesCard | null;
  messages: UiMessage[];
}
