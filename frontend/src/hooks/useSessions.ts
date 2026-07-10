import { useCallback, useState } from "react";
import { ApiError, postChat, postFeedback } from "../api/client";
import type { ChatResponse, FeedbackRating, MessageFeedback, Session, SpeciesCard, UiMessage, WildlifeIdentification } from "../types";

const NEW_SESSION_ID = "new";

function uid(): string {
  return crypto.randomUUID();
}

// AI-authored replies are only scoreable when the backend actually traced the
// turn (Langfuse enabled) — everything downstream (FeedbackButtons) treats a
// missing traceId as "don't render feedback UI for this message" rather than
// as an error, since tracing is an optional, degrade-silently feature here.
function traceFields(response: Pick<ChatResponse, "trace_id">): { traceId?: string; feedback?: MessageFeedback } {
  if (!response.trace_id) return {};
  return { traceId: response.trace_id, feedback: { rating: null, comment: "" } };
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

  const setMessageAudioUrl = useCallback(
    (sessionId: string, messageId: string, audioUrl: string) => {
      updateSession(sessionId, (s) => ({
        ...s,
        messages: s.messages.map((m) =>
          m.id === messageId && (m.kind === "text" || m.kind === "card") ? { ...m, audioUrl } : m,
        ),
      }));
    },
    [updateSession],
  );

  // Optimistically records the visitor's rating/note, then fires the request.
  // Best-effort: on failure, revert to whatever was last confirmed (not just
  // cleared) rather than surface an error bubble — feedback is never
  // mandatory and must not disrupt the chat.
  const submitFeedback = useCallback(
    async (sessionId: string, messageId: string, traceId: string, rating: FeedbackRating, comment?: string) => {
      const session = sessions.find((s) => s.id === sessionId);
      if (!session) return;
      const target = session.messages.find((m) => m.id === messageId);
      const previous: MessageFeedback =
        (target?.kind === "text" || target?.kind === "card") && target.feedback
          ? target.feedback
          : { rating: null, comment: "" };

      updateSession(sessionId, (s) => ({
        ...s,
        messages: s.messages.map((m) =>
          m.id === messageId && (m.kind === "text" || m.kind === "card")
            ? { ...m, feedback: { rating, comment: comment ?? m.feedback?.comment ?? "" } }
            : m,
        ),
      }));

      try {
        await postFeedback({ threadId: sessionId, traceId, sessionSecret: session.secret, rating, comment });
      } catch {
        updateSession(sessionId, (s) => ({
          ...s,
          messages: s.messages.map((m) =>
            m.id === messageId && (m.kind === "text" || m.kind === "card") ? { ...m, feedback: previous } : m,
          ),
        }));
      }
    },
    [sessions, updateSession],
  );

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
            messages: [...s.messages, { id: uid(), kind: "card", role: "ai", card, ...traceFields(response) }],
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
              {
                id: uid(),
                kind: "text",
                role: isFallback ? "ai" : "error",
                text,
                ...(isFallback ? traceFields(response) : {}),
              },
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
          messages: [
            ...s.messages,
            { id: uid(), kind: "text", role, text: replyText, ...(role === "ai" ? traceFields(response) : {}) },
          ],
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
    setMessageAudioUrl,
    submitFeedback,
  };
}
