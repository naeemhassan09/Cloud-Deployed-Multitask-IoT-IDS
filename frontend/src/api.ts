import type { PredictFileResponse } from "./types";

export async function uploadCsvAndPredict(file: File): Promise<PredictFileResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/api/predict-file", { method: "POST", body: form });

  if (!res.ok) {
    let detail: any = null;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(msg);
  }

  return res.json();
}