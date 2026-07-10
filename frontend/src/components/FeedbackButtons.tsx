import { useState, type CSSProperties } from "react";
import type { FeedbackRating, MessageFeedback } from "../types";

const ACCENT = "#5a7250";

// Thumbs up/down + an optional free-text note on one AI response — entirely
// optional at every step: clicking a thumb alone is a complete submission,
// the note is an additional detail a visitor may or may not bother adding.
export default function FeedbackButtons({
  feedback,
  onSubmit,
}: {
  feedback: MessageFeedback;
  onSubmit: (rating: FeedbackRating, comment?: string) => void;
}) {
  const [noteOpen, setNoteOpen] = useState(false);
  const [draft, setDraft] = useState(feedback.comment);

  const pick = (rating: FeedbackRating) => onSubmit(rating, feedback.comment || undefined);

  const sendNote = () => {
    if (!feedback.rating) return;
    onSubmit(feedback.rating, draft.trim() || undefined);
    setNoteOpen(false);
  };

  const thumbStyle = (active: boolean): CSSProperties => ({
    width: 26,
    height: 26,
    borderRadius: "50%",
    border: "none",
    background: active ? ACCENT : "#f0f0ed",
    color: active ? "#fff" : "#8a8a85",
    fontSize: 12,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <button
          onClick={() => pick("up")}
          title="Good response"
          aria-label="Good response"
          aria-pressed={feedback.rating === "up"}
          style={thumbStyle(feedback.rating === "up")}
        >
          👍
        </button>
        <button
          onClick={() => pick("down")}
          title="Poor response"
          aria-label="Poor response"
          aria-pressed={feedback.rating === "down"}
          style={thumbStyle(feedback.rating === "down")}
        >
          👎
        </button>
        {feedback.rating && !noteOpen && (
          <button
            onClick={() => setNoteOpen(true)}
            style={{ border: "none", background: "none", color: "#a0a09a", fontSize: 12, cursor: "pointer", padding: "0 2px" }}
          >
            {feedback.comment ? "Edit note" : "Add a note"}
          </button>
        )}
      </div>

      {noteOpen && (
        <div style={{ display: "flex", gap: 6, maxWidth: 420 }}>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="What made this response good or bad? (optional)"
            rows={2}
            style={{
              flex: 1,
              fontSize: 13,
              padding: "6px 9px",
              border: "1px solid #e6e6e1",
              borderRadius: 8,
              resize: "vertical",
              fontFamily: "inherit",
              background: "#fbfbfa",
            }}
          />
          <button onClick={sendNote} className="wg-chip" style={{ fontSize: 12.5, color: "#33332f", background: "#f4f4f2", border: "1px solid #e8e8e4", borderRadius: 999, padding: "7px 13px", cursor: "pointer", alignSelf: "flex-end" }}>
            Send
          </button>
        </div>
      )}
    </div>
  );
}
