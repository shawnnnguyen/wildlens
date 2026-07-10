import { useEffect, useRef, useState } from "react";
import { postSynthesizeAudio, resolveAudioUrl } from "../api/client";

const ACCENT = "#5a7250";
const ERROR_BG = "#c7c7c0";

type Status = "idle" | "loading" | "playing" | "error";

export default function AudioPlayButton({
  text,
  audioUrl,
  threadId,
  sessionSecret,
  onAudioReady,
}: {
  text: string;
  audioUrl?: string;
  threadId: string;
  sessionSecret: string;
  onAudioReady: (url: string) => void;
}) {
  const [status, setStatus] = useState<Status>("idle");
  const audioRef = useRef<HTMLAudioElement>(null);
  // Set right before onAudioReady lifts a freshly-synthesized URL into message
  // state — the <audio> element's src only picks it up on the next render, so
  // this flags the effect below to auto-play once that src actually lands.
  const autoplayRef = useRef(false);
  // Guards the in-flight synthesize fetch: if this button unmounts (e.g. the
  // user switches sessions) before the response arrives, skip the state
  // update instead of calling setStatus on an unmounted component.
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

  useEffect(() => {
    if (audioUrl && autoplayRef.current) {
      autoplayRef.current = false;
      audioRef.current?.play().catch(() => setStatus("error"));
    }
  }, [audioUrl]);

  const handleClick = async () => {
    if (status === "loading") return;

    if (audioUrl) {
      const el = audioRef.current;
      if (!el) return;
      if (el.paused) {
        el.play().catch(() => setStatus("error"));
      } else {
        el.pause();
      }
      return;
    }

    setStatus("loading");
    try {
      const response = await postSynthesizeAudio({ threadId, text, sessionSecret });
      if (!mountedRef.current) return;
      autoplayRef.current = true;
      onAudioReady(resolveAudioUrl(response.audio_url));
    } catch {
      if (mountedRef.current) setStatus("error");
    }
  };

  const icon = status === "loading" ? "…" : status === "playing" ? "⏸" : status === "error" ? "↻" : "▶";
  const title =
    status === "error" ? "Couldn't play audio — click to retry" : status === "playing" ? "Pause" : "Listen";

  return (
    <>
      <button
        onClick={handleClick}
        disabled={status === "loading"}
        title={title}
        aria-label={title}
        style={{
          width: 30,
          height: 30,
          flex: "0 0 30px",
          borderRadius: "50%",
          border: "none",
          background: status === "error" ? ERROR_BG : ACCENT,
          color: "#fff",
          fontSize: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: status === "loading" ? "wait" : "pointer",
          marginTop: 8,
        }}
      >
        {icon}
      </button>
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          preload="none"
          onPlay={() => setStatus("playing")}
          onPause={() => setStatus("idle")}
          onEnded={() => setStatus("idle")}
          onError={() => setStatus("error")}
          style={{ display: "none" }}
        />
      )}
    </>
  );
}
