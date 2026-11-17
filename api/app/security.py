"""Primitivas de contraseñas y sesiones sin dependencias criptográficas externas."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets


SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_LENGTH = 64


def _encode_binary(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode_binary(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_LENGTH,
    )
    return "$".join(
        [
            "scrypt",
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            _encode_binary(salt),
            _encode_binary(derived_key),
        ]
    )


def verify_password(password: str, encoded_password: str) -> bool:
    try:
        algorithm, encoded_n, encoded_r, encoded_p, encoded_salt, encoded_key = encoded_password.split("$")
        if algorithm != "scrypt":
            return False
        derived_key = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode_binary(encoded_salt),
            n=int(encoded_n),
            r=int(encoded_r),
            p=int(encoded_p),
            dklen=len(_decode_binary(encoded_key)),
        )
        return hmac.compare_digest(derived_key, _decode_binary(encoded_key))
    except (ValueError, TypeError):
        return False


def create_session_token() -> str:
    return secrets.token_urlsafe(32)


def session_token_digest(token: str, session_secret: str) -> str:
    return hmac.new(
        session_secret.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


DUMMY_PASSWORD_HASH = hash_password("invalid-password-for-timing-only")
