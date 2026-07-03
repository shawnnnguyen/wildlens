import type { ChatResponse, ErrorResponse } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

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
}

export async function postChat({ threadId, message, image }: PostChatParams): Promise<ChatResponse> {
  const form = new FormData();
  form.set("thread_id", threadId);
  if (message) form.set("message", message);
  if (image) form.set("image", image);

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    await parseErrorResponse(response);
  }
  return response.json();
}
