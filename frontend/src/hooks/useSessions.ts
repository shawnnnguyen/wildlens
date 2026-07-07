import { useCallback, useState } from "react";
import { ApiError, postChat } from "../api/client";
import type { Session, SpeciesCard, UiMessage, WildlifeIdentification } from "../types";

const NEW_SESSION_ID = "new";

function uid(): string {
  return crypto.randomUUID();
}

function emptySession(): Session {
  return { id: NEW_SESSION_ID, secret: "", title: "New identification", subtitle: "", thumbnail: "", species: null, messages: [] };
}

function toSpeciesCard(id: WildlifeIdentification, description: string): SpeciesCard {
  const match = id.species.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  return {
    common: match ? match[1].trim() : id.species,
    scientific: match ? match[2].trim() : "",
    confidenceScore: id.confidence_score,
    threatLevel: id.threat_level,
    habitatContext: id.habitat_context,
    visualTraits: id.visual_traits,
    description,
  };
}

// No backend persistence yet (MemorySaver is in-process only), so sessions
// live only in memory for the life of the tab — nothing is restored on reload.
export function useSessions() {
  const [sessions, setSessions] = useState<Session[]>([emptySession()]);
  const [activeId, setActiveId] = useState<string>(NEW_SESSION_ID);
  const [draft, setDraft] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  const active = sessions.find((s) => s.id === activeId) ?? sessions[0];

  const updateSession = useCallback((id: string, updater: (s: Session) => Session) => {
    setSessions((prev) => prev.map((s) => (s.id === id ? updater(s) : s)));
  }, []);

  const onNewSession = useCallback(() => {
    setSessions((prev) => (prev.some((s) => s.id === NEW_SESSION_ID) ? prev : [emptySession(), ...prev]));
    setActiveId(NEW_SESSION_ID);
    setDraft("");
  }, []);

  const onSelectSession = useCallback((id: string) => {
    setActiveId(id);
    setDraft("");
  }, []);

  const startIdentification = useCallback(
    async (file: File) => {
      const sid = "sess_" + uid();
      const imageUrl = URL.createObjectURL(file);

      setSessions((prev) => [
        {
          id: sid,
          secret: "",
          title: "Identifying…",
          subtitle: "Just now",
          thumbnail: imageUrl,
          species: null,
          messages: [{ id: uid(), kind: "image", role: "human", imageUrl }],
        },
        ...prev.filter((s) => s.id !== NEW_SESSION_ID),
      ]);
      setActiveId(sid);
      setDraft("");
      setIsThinking(true);

      try {
        const response = await postChat({ threadId: sid, image: file });
        // First call for this thread_id — the backend hands back a capability
        // token that every later call on this session must present (see
        // client.ts/postChat). Not accounts/login — the app stays single-use
        // and anonymous; this just stops anyone who learns the thread_id from
        // reading or continuing someone else's session.
        const secret = response.session_secret ?? "";

        if (response.identification) {
          const card = toSpeciesCard(response.identification, response.final_script);
          updateSession(sid, (s) => ({
            ...s,
            secret,
            title: card.common,
            subtitle: card.scientific || "Just now",
            species: card,
            messages: [...s.messages, { id: uid(), kind: "card", role: "ai", card }],
          }));
        } else {
          const isFallback = response.fallback_triggered;
          const text =
            response.error_message ??
            response.final_script ??
            "I couldn't get a clear look at that — try another photo.";
          updateSession(sid, (s) => ({
            ...s,
            secret,
            title: isFallback ? "Unclear photo" : "Identification failed",
            subtitle: isFallback ? "Try another image" : "Something went wrong",
            messages: [
              ...s.messages,
              { id: uid(), kind: "text", role: isFallback ? "ai" : "error", text },
            ],
          }));
        }
      } catch (err) {
        const text = err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
        updateSession(sid, (s) => ({
          ...s,
          title: "Identification failed",
          subtitle: "Something went wrong",
          messages: [...s.messages, { id: uid(), kind: "text", role: "error", text }],
        }));
      } finally {
        setIsThinking(false);
      }
    },
    [updateSession],
  );

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      const session = active;
      if (!trimmed || !session || session.id === NEW_SESSION_ID) return;
      const sid = session.id;

      updateSession(sid, (s) => ({
        ...s,
        messages: [...s.messages, { id: uid(), kind: "text", role: "human", text: trimmed }],
      }));
      setDraft("");
      setIsThinking(true);

      try {
        const response = await postChat({ threadId: sid, message: trimmed, sessionSecret: session.secret });
        const replyText = response.error_message ?? response.final_script;
        const role: UiMessage["role"] = response.error_message ? "error" : "ai";
        updateSession(sid, (s) => ({
          ...s,
          messages: [...s.messages, { id: uid(), kind: "text", role, text: replyText }],
        }));
      } catch (err) {
        const msg = err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
        updateSession(sid, (s) => ({
          ...s,
          messages: [...s.messages, { id: uid(), kind: "text", role: "error", text: msg }],
        }));
      } finally {
        setIsThinking(false);
      }
    },
    [active, updateSession],
  );

  return {
    sessions,
    active,
    activeId,
    draft,
    setDraft,
    isDragging,
    setIsDragging,
    isThinking,
    onNewSession,
    onSelectSession,
    startIdentification,
    send,
  };
}
