CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    bundle_schema TEXT NOT NULL,
    contract_schema TEXT NOT NULL,
    contract_sha256 TEXT NOT NULL,
    data_config_sha256 TEXT NOT NULL,
    classifier_arch TEXT NOT NULL,
    classifier_display_name TEXT NOT NULL,
    classifier_run_id TEXT NOT NULL,
    classifier_sha256 TEXT NOT NULL,
    classifier_epoch INTEGER NOT NULL,
    classifier_threshold REAL NOT NULL,
    detector_arch TEXT NOT NULL,
    detector_run_id TEXT NOT NULL,
    detector_sha256 TEXT NOT NULL,
    detector_threshold REAL NOT NULL,
    concordance_rule TEXT NOT NULL,
    image_size INTEGER NOT NULL,
    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
    registered_at TEXT NOT NULL,
    UNIQUE(contract_sha256, classifier_sha256, detector_sha256)
);

CREATE TABLE IF NOT EXISTS radiograph_quality_checks (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL UNIQUE REFERENCES evaluations(id) ON DELETE CASCADE,
    radiograph_id TEXT NOT NULL REFERENCES radiographs(id) ON DELETE CASCADE,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    warnings_json TEXT NOT NULL,
    failures_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    checked_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL UNIQUE REFERENCES evaluations(id) ON DELETE CASCADE,
    radiograph_id TEXT NOT NULL REFERENCES radiographs(id) ON DELETE CASCADE,
    model_version_id TEXT NOT NULL REFERENCES model_versions(id),
    predicted_class TEXT NOT NULL CHECK (predicted_class IN ('health', 'sick', 'TB')),
    p_health REAL NOT NULL,
    p_sick REAL NOT NULL,
    p_tb REAL NOT NULL,
    classifier_alert INTEGER NOT NULL CHECK (classifier_alert IN (0, 1)),
    detector_box_count INTEGER NOT NULL,
    detector_max_conf REAL NOT NULL,
    detector_alert INTEGER NOT NULL CHECK (detector_alert IN (0, 1)),
    concordance_state TEXT NOT NULL CHECK (concordance_state IN (
        'both_alert',
        'classifier_only_review',
        'detector_only_review',
        'neither_alert'
    )),
    review_required INTEGER NOT NULL CHECK (review_required IN (0, 1)),
    low_confidence INTEGER NOT NULL CHECK (low_confidence IN (0, 1)),
    low_confidence_reasons_json TEXT NOT NULL,
    inference_ms REAL NOT NULL,
    explainability_verified INTEGER NOT NULL CHECK (explainability_verified IN (0, 1)),
    gradcam_path TEXT NOT NULL,
    rollout_path TEXT NOT NULL,
    created_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    box_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    norm_x1 REAL NOT NULL,
    norm_y1 REAL NOT NULL,
    norm_x2 REAL NOT NULL,
    norm_y2 REAL NOT NULL,
    letterbox_x1 REAL NOT NULL,
    letterbox_y1 REAL NOT NULL,
    letterbox_x2 REAL NOT NULL,
    letterbox_y2 REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS index_detections_prediction
    ON detections(prediction_id, box_index);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    prediction_id TEXT REFERENCES predictions(id) ON DELETE SET NULL,
    author_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    clinical_impression TEXT NOT NULL,
    agreement_status TEXT NOT NULL CHECK (agreement_status IN (
        'concordante',
        'discrepante',
        'requiere_confirmacion',
        'sin_hallazgos_relevantes'
    )),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS index_observations_evaluation
    ON observations(evaluation_id, created_at DESC);
