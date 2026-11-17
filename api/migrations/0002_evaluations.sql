CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    internal_code TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN (
        'pendiente_imagen',
        'imagen_cargada',
        'lista_para_analisis',
        'analizada',
        'revisada'
    )),
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    created_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS index_evaluations_created_at
    ON evaluations(created_at DESC);

CREATE INDEX IF NOT EXISTS index_evaluations_status
    ON evaluations(status, created_at DESC);

CREATE TABLE IF NOT EXISTS radiographs (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL UNIQUE REFERENCES evaluations(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type IN ('image/png', 'image/jpeg')),
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    sha256 TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    uploaded_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL
);
