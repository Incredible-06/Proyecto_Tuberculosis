"""Comando local para crear la primera cuenta administradora."""

from __future__ import annotations

import argparse
import getpass
import os

from .config import Settings
from .database import connect_database, initialize_database
from .schemas import CreateUserRequest, UserRole
from .services import DuplicateUsernameError, count_users, create_user_account


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea la cuenta administradora inicial.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--display-name", required=True)
    arguments = parser.parse_args()

    password = os.getenv("BOOTSTRAP_ADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Contraseña inicial: ")
        confirmation = getpass.getpass("Repita la contraseña: ")
        if password != confirmation:
            raise SystemExit("Las contraseñas no coinciden.")

    try:
        payload = CreateUserRequest(
            username=arguments.username,
            display_name=arguments.display_name,
            role=UserRole.ADMINISTRADOR,
            password=password,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    settings = Settings.from_environment()
    initialize_database(settings)
    connection = connect_database(settings.database_path)
    try:
        if count_users(connection) > 0:
            raise SystemExit("Ya existen cuentas. Use la API autenticada para crear usuarios.")
        created_user = create_user_account(
            connection,
            username=payload.username,
            display_name=payload.display_name,
            role=UserRole.ADMINISTRADOR,
            password=payload.password.get_secret_value(),
            actor_user_id=None,
        )
    except DuplicateUsernameError as error:
        raise SystemExit("El identificador de acceso ya existe.") from error
    finally:
        connection.close()
    print(f"Cuenta administradora creada para {created_user['username']}.")


if __name__ == "__main__":
    main()
