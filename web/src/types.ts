export type Role =
  | "administrador"
  | "tecnologo_radiologia"
  | "medico_programa_tb"
  | "medico_radiologo"
  | "coordinador_estrategia_tb";

export interface User {
  id: string;
  username: string;
  display_name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  permissions: string[];
}

export interface RoleInfo {
  value: Role;
  label: string;
  permissions: string[];
}

export interface LoginResponse {
  user: User;
  session_expires_at: string;
}

export type EvaluationStatus =
  | "pendiente_imagen"
  | "imagen_cargada"
  | "lista_para_analisis"
  | "analizada"
  | "revisada";

export interface Radiograph {
  id: string;
  evaluation_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  width: number;
  height: number;
  sha256: string;
  uploaded_at: string;
  content_url: string;
}

export type PredictionClass = "health" | "sick" | "TB";

export type ConcordanceState =
  | "both_alert"
  | "classifier_only_review"
  | "detector_only_review"
  | "neither_alert";

export type ObservationAgreement =
  | "concordante"
  | "discrepante"
  | "requiere_confirmacion"
  | "sin_hallazgos_relevantes";

export interface QualityMetrics {
  width: number;
  height: number;
  min_dimension: number;
  aspect_ratio: number;
  intensity_std: number;
  dynamic_range_p99_p01: number;
  clip_low_ratio: number;
  clip_high_ratio: number;
}

export interface QualityCheck {
  id: string;
  evaluation_id: string;
  radiograph_id: string;
  passed: boolean;
  warnings: string[];
  failures: string[];
  metrics: QualityMetrics;
  checked_by_user_id: string | null;
  checked_at: string;
}

export interface ModelVersion {
  id: string;
  bundle_schema: string;
  contract_schema: string;
  contract_sha256: string;
  data_config_sha256: string;
  classifier_arch: string;
  classifier_display_name: string;
  classifier_run_id: string;
  classifier_sha256: string;
  classifier_epoch: number;
  classifier_threshold: number;
  detector_arch: string;
  detector_run_id: string;
  detector_sha256: string;
  detector_threshold: number;
  concordance_rule: string;
  image_size: number;
  is_active: boolean;
  registered_at: string;
}

export interface Detection {
  id: string;
  box_index: number;
  label: string;
  confidence: number;
  xyxy: number[];
  normalized_xyxy: number[];
}

export interface Prediction {
  id: string;
  evaluation_id: string;
  radiograph_id: string;
  predicted_class: PredictionClass;
  p_health: number;
  p_sick: number;
  p_tb: number;
  classifier_alert: boolean;
  detector_box_count: number;
  detector_max_conf: number;
  detector_alert: boolean;
  concordance_state: ConcordanceState;
  review_required: boolean;
  low_confidence: boolean;
  low_confidence_reasons: string[];
  inference_ms: number;
  explainability_verified: boolean;
  gradcam_url: string;
  rollout_url: string;
  detections: Detection[];
  model_version: ModelVersion;
  created_by_user_id: string | null;
  created_at: string;
}

export interface Observation {
  id: string;
  evaluation_id: string;
  prediction_id: string | null;
  author_user_id: string | null;
  author_display_name: string | null;
  clinical_impression: string;
  agreement_status: ObservationAgreement;
  created_at: string;
}

export interface Evaluation {
  id: string;
  internal_code: string;
  status: EvaluationStatus;
  notes: string | null;
  created_at: string;
  updated_at: string;
  created_by_user_id: string | null;
  created_by_display_name: string | null;
  radiograph: Radiograph | null;
  quality_check: QualityCheck | null;
  has_prediction: boolean;
  concordance_state: ConcordanceState | null;
  review_required: boolean | null;
  observation_count: number;
}

export const ROLE_LABELS: Record<Role, string> = {
  administrador: "Administrador del sistema",
  tecnologo_radiologia: "Tecnólogo médico en radiología",
  medico_programa_tb: "Médico del programa de TB",
  medico_radiologo: "Médico radiólogo",
  coordinador_estrategia_tb: "Coordinador de la Estrategia Sanitaria de TB",
};

export const ROLE_SHORT_LABELS: Record<Role, string> = {
  administrador: "Administración",
  tecnologo_radiologia: "Tecnología médica",
  medico_programa_tb: "Programa de TB",
  medico_radiologo: "Radiología",
  coordinador_estrategia_tb: "Coordinación TB",
};

export const STATUS_LABELS: Record<EvaluationStatus, string> = {
  pendiente_imagen: "Pendiente de imagen",
  imagen_cargada: "Imagen cargada",
  lista_para_analisis: "Lista para análisis",
  analizada: "Analizada",
  revisada: "Revisada",
};

export const CLASS_LABELS: Record<PredictionClass, string> = {
  health: "Normal (health)",
  sick: "Otra alteración no-TB (sick)",
  TB: "Sospecha de tuberculosis (TB)",
};

export const CONCORDANCE_LABELS: Record<ConcordanceState, string> = {
  both_alert: "Ambas alertas (clasificador + detector)",
  classifier_only_review: "Revisión por clasificador (sin cajas)",
  detector_only_review: "Revisión por detector focal",
  neither_alert: "Sin alerta operativa",
};

export const CONCORDANCE_DESCRIPTIONS: Record<ConcordanceState, string> = {
  both_alert:
    "ViT-B/16 superó el umbral operativo (p_TB >= 0,30) y YOLO11n localizó al menos una lesión focal (>= 0,25).",
  classifier_only_review:
    "ViT-B/16 superó p_TB >= 0,30 sin cajas YOLO11n >= 0,25. La regla aditiva conserva la alerta sin suprimirla.",
  detector_only_review:
    "YOLO11n detectó al menos una zona focal (>= 0,25) aunque p_TB fue menor a 0,30. Requiere contraste radiológico.",
  neither_alert:
    "Ni ViT-B/16 (p_TB >= 0,30) ni YOLO11n (>= 0,25) activaron alerta operativa. No descarta criterio clínico.",
};

export const AGREEMENT_LABELS: Record<ObservationAgreement, string> = {
  concordante: "Concordante con la impresión clínica",
  discrepante: "Discrepante con el modelo",
  requiere_confirmacion: "Requiere estudio confirmatorio",
  sin_hallazgos_relevantes: "Sin hallazgos activos de TB",
};
