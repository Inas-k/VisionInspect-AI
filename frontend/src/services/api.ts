import axios from "axios";
import type { HistoryRecord, InspectionParameters, InspectionResult, MetricsSummary, SampleImage } from "../types/api";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

// Log backend errors to console for easier debugging
client.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err?.response?.data?.detail ?? err?.message ?? "UNKNOWN ERROR";
    console.error("[VisionInspect API]", err?.response?.status, detail);
    return Promise.reject(new Error(String(detail)));
  }
);

export const mediaUrl = (path?: string) => (path ? `${API_BASE}${path}` : "");

export async function getHealth() {
  const { data } = await client.get("/health");
  return data as { status: string; service: string; vision_engine: string };
}

export async function getSamples() {
  const { data } = await client.get("/api/samples");
  return data.samples as SampleImage[];
}

export async function inspectImage(file: File, params: InspectionParameters, saveResults = true) {
  const form = new FormData();
  form.append("image", file);
  form.append("canny_low", String(params.canny_low));
  form.append("canny_high", String(params.canny_high));
  form.append("min_defect_area", String(params.min_defect_area));
  form.append("pass_fail_threshold", String(params.pass_fail_threshold));
  form.append("blur_kernel", String(params.blur_kernel));
  // FastAPI reads bool Form fields as "true"/"false" strings — must be lowercase
  form.append("save_results", saveResults ? "true" : "false");
  const { data } = await client.post("/api/inspect", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data as InspectionResult;
}

export async function inspectSample(sample: SampleImage, params: InspectionParameters, saveResults = true) {
  // Use the dedicated backend endpoint that reads the file directly from disk
  // instead of downloading + re-uploading through the browser (avoids CORS/blob issues)
  const category = sample.category.toLowerCase(); // "good" | "defective"
  const { data } = await client.get(`/api/sample-inspect/${category}/${sample.filename}`, {
    params: {
      canny_low: params.canny_low,
      canny_high: params.canny_high,
      min_defect_area: params.min_defect_area,
      pass_fail_threshold: params.pass_fail_threshold,
      blur_kernel: params.blur_kernel,
      save_results: saveResults,
    },
  });
  return data as InspectionResult;
}

export async function getHistory() {
  const { data } = await client.get("/api/history");
  return data as HistoryRecord[];
}

export async function getMetrics() {
  const { data } = await client.get("/api/metrics");
  return data as MetricsSummary;
}

export async function resetHistory() {
  await client.post("/api/reset-history");
}

