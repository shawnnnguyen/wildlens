import type { AudioSynthesizeResponse, ChatResponse, ErrorResponse, FeedbackRating, FeedbackResponse } from "../types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export function resolveAudioUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export class ApiError extends Error {
  code: string;
  field?: string | null;
  status: number;

  constructor(status: number, error: ErrorResponse["error"]) {
    super(error.message);
    this.name = "ApiError";
    this.code = error.code;
    this.field = error.field;
    this.status = status;
  }
}

async function parseErrorResponse(response: Response): Promise<never> {
  let body: ErrorResponse | null = null;
  try {
    body = await response.json();
  } catch {
    // response body wasn't JSON — fall through to the generic error below
  }
  if (body?.error) {
    throw new ApiError(response.status, body.error);
  }
  throw new ApiError(response.status, {
    code: "UNKNOWN_ERROR",
    message: `Request failed with status ${response.status}`,
  });
}

export interface PostChatParams {
  threadId: string;
  message?: string;
  image?: File;
  sessionSecret?: string; // required once a session already exists — see useSessions.ts
}

export async function postChat({
  threadId,
  message,
  image,
  sessionSecret,
}: PostChatParams): Promise<ChatResponse> {
  const form = new FormData();
  form.set("thread_id", threadId);
  if (message) form.set("message", message);
  if (image) form.set("image", image);

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    body: form,
    headers: sessionSecret ? { "X-Session-Secret": sessionSecret } : undefined,
  });

  if (!response.ok) {
    await parseErrorResponse(response);
  }
  return response.json();
}

export interface PostFeedbackParams {
  threadId: string;
  traceId: string;
  sessionSecret: string;
  rating: FeedbackRating;
  comment?: string;
}

export async function postFeedback({
  threadId,
  traceId,
  sessionSecret,
  rating,
  comment,
}: PostFeedbackParams): Promise<FeedbackResponse> {
  const response = await fetch(`${API_BASE_URL}/api/feedback`, {
    method: "POST",
    body: JSON.stringify({
      thread_id: threadId,
      trace_id: traceId,
      rating,
      comment: comment || null,
    }),
    headers: { "Content-Type": "application/json", "X-Session-Secret": sessionSecret },
  });

  if (!response.ok) {
    await parseErrorResponse(response);
  }
  return response.json();
}

export interface PostSynthesizeAudioParams {
  threadId: string;
  text: string;
  sessionSecret: string;
}

export async function postSynthesizeAudio({
  threadId,
  text,
  sessionSecret,
}: PostSynthesizeAudioParams): Promise<AudioSynthesizeResponse> {
  const form = new FormData();
  form.set("thread_id", threadId);
  form.set("text", text);

  const response = await fetch(`${API_BASE_URL}/api/audio/synthesize`, {
    method: "POST",
    body: form,
    headers: { "X-Session-Secret": sessionSecret },
  });

  if (!response.ok) {
    await parseErrorResponse(response);
  }
  return response.json();
}
