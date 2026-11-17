CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN (
        'administrador',
        'tecnologo_radiologia',
        'medico_programa_tb',
        'medico_radiologo',
        'coordinador_estrategia_tb'
    )),
    password_hash TEXT NOT NULL,
    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
    failed_login_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_login_count >= 0),
    locked_until TEXT,
    created_at TEXT NOT NULL,
    created_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_digest TEXT NOT NULL UNIQUE,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE INDEX IF NOT EXISTS index_auth_sessions_active
    ON auth_sessions(token_digest, expires_at, revoked_at);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    actor_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    subject_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS index_audit_events_occurred_at
    ON audit_events(occurred_at DESC);
