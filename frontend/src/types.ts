export interface StageBlock {
  status: "done" | "running" | "error";
  ms?: number;
  rows?: number;
  used_rows?: number;
  features_expected?: number;
  features_present?: number;
}

export interface Stages {
  upload: StageBlock;
  validate: StageBlock;
  preprocess: StageBlock;
  predict: StageBlock;
  results: StageBlock;
}

export interface Summary {
  total_rows: number;
  num_benign: number;
  num_attacks: number;
  attack_distribution: Record<string, number>;
  device_distribution: Record<string, number>;
}

export interface Metrics {
  preprocessing_ms: number;
  inference_ms: number;
  total_ms: number;
}

export interface PredictionRow {
  row_index: number;
  device_id: number;
  device_name: string;
  attack_label: string;
  attack_score: number;
  device_confidence: number;
  device_raw_label?: string;
}

export interface PredictFileResponse {
  stages: Stages;
  summary: Summary;
  metrics: Metrics;
  predictions: PredictionRow[];
  eval_summary?: {
    attack_accuracy?: number;
    device_accuracy?: number;
  };
}