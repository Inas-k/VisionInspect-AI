export type Verdict = "PASS" | "FAIL" | "UNKNOWN";

export interface InspectionParameters {
  canny_low: number;
  canny_high: number;
  min_defect_area: number;
  pass_fail_threshold: number;
  blur_kernel: number;
}

export interface InspectionMetrics {
  num_defects: number;
  total_defect_area: number;
  image_area: number;
  defect_percentage: number;
  confidence: number;
}

export interface InspectionImages {
  original: string;
  grayscale: string;
  edges: string;
  annotated: string;
}

export interface InspectionResult {
  inspection_id: string;
  filename: string;
  image_width: number;
  image_height: number;
  inspection_timestamp: string;
  inspection_duration_ms: number;
  parameters: InspectionParameters;
  metrics: InspectionMetrics;
  verdict: Verdict;
  images: InspectionImages;
}

export interface SampleImage {
  filename: string;
  category: "GOOD" | "DEFECTIVE";
  preview_url: string;
}

export interface HistoryRecord {
  inspection_id: string;
  filename: string;
  timestamp: string;
  width: number;
  height: number;
  num_defects: number;
  total_defect_area: number;
  image_area: number;
  defect_percentage: number;
  confidence: number;
  verdict: Verdict;
  inspection_duration_ms: number;
  parameters: InspectionParameters;
  images: InspectionImages;
}

export interface MetricsSummary {
  total_inspections: number;
  pass_count: number;
  fail_count: number;
  pass_rate: number;
  average_defect_coverage: number;
  average_inspection_time_ms: number;
}

export type StageKey = "original" | "grayscale" | "edges" | "annotated";

