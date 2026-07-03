import { useEffect, useRef, type CSSProperties, type DragEvent, type KeyboardEvent } from "react";
import { useSessions } from "../hooks/useSessions";
import type { Session, SpeciesCard, UiMessage } from "../types";

const ACCENT = "#5a7250";

const FOLLOW_UP_SUGGESTIONS = ["What does it eat?", "Where does it live?", "Is it endangered?"];

const EXAMPLE_SPECIES = ["Lions", "Elephants", "Leopards", "Zebra", "Giraffe"];

const THREAT_LABEL: Record<SpeciesCard["threatLevel"], string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
};

function fileFromDataTransfer(e: DragEvent<HTMLElement>): File | null {
  return e.dataTransfer?.files?.[0] ?? null;
}

export default function AnimalIdChat() {
  const {
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
  } = useSessions();

  const fileRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [active?.messages.length, isThinking]);

  const isEmpty = !active || active.messages.length === 0;

  const onPick = () => fileRef.current?.click();

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) startIdentification(f);
    e.target.value = "";
  };

  const onDragOver = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };
  const onDragLeave = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    if (e.currentTarget === e.target) setIsDragging(false);
  };
  const onDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const f = fileFromDataTransfer(e);
    if (f) startIdentification(f);
  };

  const onDraftChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDraft(e.target.value);
    const el = inputRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };
  const onSend = () => {
    send(draft);
    const el = inputRef.current;
    if (el) el.style.height = "auto";
  };

  const headerTitle = active?.species ? active.species.common : active?.id === "new" ? "New identification" : active?.title ?? "New identification";
  const headerSub = active?.species?.scientific ?? "";

  return (
    <div style={{ display: "flex", height: "100vh", width: "100%", background: "#ffffff", color: "#1a1a19", overflow: "hidden" }}>
      <Sidebar sessions={sessions} activeId={activeId} onNewSession={onNewSession} onSelectSession={onSelectSession} />

      <main
        style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, position: "relative" }}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <header style={{ height: 54, flex: "0 0 54px", borderBottom: "1px solid #f0f0ee", display: "flex", alignItems: "center", padding: "0 24px", gap: 12 }}>
          <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: "-0.01em" }}>{headerTitle}</span>
          {headerSub && (
            <span style={{ fontSize: 12.5, color: "#a0a09a", fontStyle: "italic", fontFamily: "Newsreader, serif" }}>{headerSub}</span>
          )}
        </header>

        {isEmpty ? (
          <EmptyState onPick={onPick} />
        ) : (
          <>
            <div ref={scrollRef} style={{ flex: 1, overflowY: "auto" }}>
              <div style={{ maxWidth: 720, margin: "0 auto", padding: "28px 24px 8px" }}>
                {active!.messages.map((m) => (
                  <MessageRow key={m.id} message={m} onSuggestionClick={send} />
                ))}
                {isThinking && <ThinkingDots />}
              </div>
            </div>

            <div style={{ padding: "10px 24px 22px" }}>
              <div style={{ maxWidth: 720, margin: "0 auto" }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-end",
                    gap: 8,
                    background: "#f7f7f5",
                    border: "1px solid #e6e6e1",
                    borderRadius: 18,
                    padding: "6px 8px 6px 6px",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
                  }}
                >
                  <textarea
                    ref={inputRef}
                    value={draft}
                    onChange={onDraftChange}
                    onKeyDown={onKeyDown}
                    rows={1}
                    placeholder="Ask a follow-up about this animal…"
                    style={textareaStyle}
                  />
                  <button onClick={onSend} className="wg-send" style={sendStyle}>
                    ↑
                  </button>
                </div>
                <div style={{ textAlign: "center", fontSize: 11, color: "#b4b4ad", marginTop: 9 }}>
                  Safari Guide can make mistakes — verify critical identifications.
                </div>
              </div>
            </div>
          </>
        )}

        <input ref={fileRef} type="file" accept="image/*" onChange={onFileChange} style={{ display: "none" }} />

        {isDragging && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "rgba(255,255,255,0.82)",
              backdropFilter: "blur(2px)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 20,
              pointerEvents: "none",
            }}
          >
            <div style={{ border: `2px dashed ${ACCENT}`, borderRadius: 18, padding: "48px 72px", textAlign: "center" }}>
              <div style={{ fontSize: 34, marginBottom: 8 }}>📥</div>
              <div style={{ fontSize: 15, fontWeight: 600 }}>Drop to identify</div>
            </div>
          </div>
        )}
      </main>

      <button
        onClick={onPick}
        disabled={isThinking}
        title={isThinking ? "Please wait for the current response" : "Upload a photo"}
        className="wg-fab"
        style={{ ...fabStyle, opacity: isThinking ? 0.5 : 1, cursor: isThinking ? "not-allowed" : "pointer" }}
      >
        ＋
      </button>
    </div>
  );
}

function Sidebar({
  sessions,
  activeId,
  onNewSession,
  onSelectSession,
}: {
  sessions: Session[];
  activeId: string;
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
}) {
  return (
    <aside style={{ width: 264, flex: "0 0 264px", background: "#f7f7f5", borderRight: "1px solid #ececea", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "16px 14px 10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "4px 6px 14px" }}>
          <div style={{ width: 26, height: 26, borderRadius: 7, background: ACCENT, display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 15 }}>
            🐾
          </div>
          <span style={{ fontSize: 14.5, fontWeight: 600, letterSpacing: "-0.01em" }}>Safari Guide</span>
        </div>
        <button onClick={onNewSession} className="wg-newbtn" style={newBtnStyle}>
          <span style={{ fontSize: 16, lineHeight: 0, color: "#8a8a85" }}>＋</span>
          New identification
        </button>
      </div>

      <div style={{ padding: "14px 10px 6px", fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", color: "#a0a09a" }}>
        History
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "0 8px 12px" }}>
        {sessions.map((s) => {
          const isActive = s.id === activeId;
          return (
            <div
              key={s.id}
              onClick={() => onSelectSession(s.id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: 8,
                borderRadius: 9,
                cursor: "pointer",
                marginBottom: 2,
                background: isActive ? "#ececea" : "transparent",
              }}
            >
              <div style={{ width: 34, height: 34, flex: "0 0 34px", borderRadius: 7, overflow: "hidden", background: "#e6e6e1", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13 }}>
                {s.thumbnail ? <img src={s.thumbnail} style={{ width: "100%", height: "100%", objectFit: "cover" }} alt="" /> : "🐾"}
              </div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 500, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.title}</div>
                <div style={{ fontSize: 11.5, color: "#a0a09a", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.subtitle}</div>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ padding: "12px 16px", borderTop: "1px solid #ececea", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#e6e6e1", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 600, color: "#6a6a63" }}>
          SR
        </div>
        <div style={{ fontSize: 12.5, color: "#6a6a63" }}>Safari Ranger</div>
      </div>
    </aside>
  );
}

function EmptyState({ onPick }: { onPick: () => void }) {
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div style={{ maxWidth: 520, width: "100%", textAlign: "center" }}>
        <h1 style={{ fontFamily: "Newsreader, serif", fontWeight: 400, fontSize: 32, lineHeight: 1.15, margin: "0 0 10px", letterSpacing: "-0.01em" }}>
          Identify any wildlife
        </h1>
        <p style={{ fontSize: 14.5, color: "#7a7a74", margin: "0 0 28px", lineHeight: 1.5 }}>
          Drop a photo of an animal and I'll tell you the species, then answer anything you'd like to know about it.
        </p>
        <div onClick={onPick} style={{ border: "1.5px dashed #d6d6d0", borderRadius: 16, padding: "40px 24px", cursor: "pointer", background: "#fbfbfa" }}>
          <div style={{ fontSize: 30, marginBottom: 10 }}>📷</div>
          <div style={{ fontSize: 14.5, fontWeight: 500, marginBottom: 4 }}>Drop an image here, or click to upload</div>
          <div style={{ fontSize: 12.5, color: "#a0a09a" }}>JPG or PNG · a clear photo works best</div>
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap", marginTop: 20 }}>
          {EXAMPLE_SPECIES.map((ex) => (
            <span key={ex} style={{ fontSize: 12, color: "#9a9a93", padding: "6px 12px", border: "1px solid #ececea", borderRadius: 999 }}>
              {ex}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function MessageRow({ message, onSuggestionClick }: { message: UiMessage; onSuggestionClick: (text: string) => void }) {
  const isCard = message.kind === "card";
  const isUser = message.role === "human";
  const wrapStyle: CSSProperties = {
    display: "flex",
    marginBottom: isCard ? 28 : 20,
    justifyContent: isUser ? "flex-end" : "flex-start",
    flexDirection: isCard ? "column" : undefined,
  };

  if (message.kind === "image") {
    return (
      <div style={wrapStyle}>
        <img src={message.imageUrl} style={{ display: "block", maxWidth: 280, width: "100%", borderRadius: 12 }} alt="Uploaded animal" />
      </div>
    );
  }

  if (message.kind === "card") {
    return (
      <div style={wrapStyle}>
        <SpeciesCardView card={message.card} onSuggestionClick={onSuggestionClick} />
      </div>
    );
  }

  const isError = message.role === "error";
  const bubbleStyle: CSSProperties = isUser
    ? { display: "inline-block", background: "#f0f0ed", padding: "10px 15px", borderRadius: "16px 16px 4px 16px", fontSize: 14.5, lineHeight: 1.55, maxWidth: "78%", textAlign: "left", whiteSpace: "pre-wrap" }
    : {
        display: "block",
        fontSize: 14.5,
        lineHeight: 1.65,
        color: isError ? "#a13d2f" : "#26261f",
        maxWidth: 640,
        whiteSpace: "pre-wrap",
      };

  return (
    <div style={wrapStyle}>
      <div style={bubbleStyle}>{message.text}</div>
    </div>
  );
}

function SpeciesCardView({ card, onSuggestionClick }: { card: SpeciesCard; onSuggestionClick: (text: string) => void }) {
  const details = [
    { label: "Habitat", value: card.habitatContext },
    { label: "Threat level", value: THREAT_LABEL[card.threatLevel] },
    { label: "Key traits", value: card.visualTraits.join(", ") || "—" },
  ];

  return (
    <div style={{ width: "100%", animation: "acfade .4s ease both" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
        <span style={{ fontFamily: "Newsreader, serif", fontSize: 25, letterSpacing: "-0.01em" }}>{card.common}</span>
        {card.scientific && (
          <span style={{ fontFamily: "Newsreader, serif", fontStyle: "italic", fontSize: 15, color: "#9a9a93" }}>{card.scientific}</span>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "12px 0 4px" }}>
        <div style={{ height: 6, flex: 1, maxWidth: 180, background: "#eeeeeb", borderRadius: 999, overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${Math.round(card.confidenceScore * 100)}%`, background: ACCENT, borderRadius: 999 }} />
        </div>
        <span style={{ fontSize: 12, color: "#8a8a85" }}>{Math.round(card.confidenceScore * 100)}% match</span>
      </div>

      <p style={{ fontSize: 14.5, lineHeight: 1.6, color: "#33332f", margin: "14px 0 16px", maxWidth: 600, whiteSpace: "pre-wrap" }}>
        {card.description}
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 1, background: "#ececea", border: "1px solid #ececea", borderRadius: 12, overflow: "hidden" }}>
        {details.map((d) => (
          <div key={d.label} style={{ background: "#fbfbfa", padding: "12px 14px" }}>
            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em", color: "#a0a09a", marginBottom: 3 }}>{d.label}</div>
            <div style={{ fontSize: 13.5, fontWeight: 500 }}>{d.value}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 16 }}>
        {FOLLOW_UP_SUGGESTIONS.map((label) => (
          <button key={label} onClick={() => onSuggestionClick(label)} className="wg-chip" style={chipStyle}>
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

function ThinkingDots() {
  return (
    <div style={{ display: "flex", gap: 5, padding: "8px 0 20px" }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#c0c0b9", animation: "acblink 1.2s infinite both" }} />
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#c0c0b9", animation: "acblink 1.2s infinite both .2s" }} />
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#c0c0b9", animation: "acblink 1.2s infinite both .4s" }} />
    </div>
  );
}

const newBtnStyle: CSSProperties = {
  width: "100%",
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "10px 12px",
  background: "#ffffff",
  border: "1px solid #e4e4e0",
  borderRadius: 10,
  fontSize: 13.5,
  fontWeight: 500,
  color: "#1a1a19",
  cursor: "pointer",
  boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
};

const fabStyle: CSSProperties = {
  position: "fixed",
  right: 28,
  bottom: 28,
  width: 52,
  height: 52,
  borderRadius: "50%",
  border: "none",
  background: ACCENT,
  color: "#fff",
  fontSize: 22,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  boxShadow: "0 4px 14px rgba(0,0,0,0.18)",
  zIndex: 30,
};

const sendStyle: CSSProperties = {
  flex: "0 0 auto",
  width: 34,
  height: 34,
  borderRadius: 9,
  border: "none",
  background: ACCENT,
  color: "#fff",
  fontSize: 17,
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const chipStyle: CSSProperties = {
  fontSize: 12.5,
  color: "#33332f",
  background: "#f4f4f2",
  border: "1px solid #e8e8e4",
  borderRadius: 999,
  padding: "7px 13px",
  cursor: "pointer",
};

const textareaStyle: CSSProperties = {
  flex: 1,
  border: "none",
  outline: "none",
  resize: "none",
  background: "transparent",
  fontSize: 14.5,
  lineHeight: 1.5,
  color: "#1a1a19",
  maxHeight: 160,
  padding: "7px 0",
};
