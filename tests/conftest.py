import socket

import pytest


def service_up(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def require_service(port: int, name: str) -> None:
    if not service_up(port):
        pytest.skip(f"{name} is not running on port {port} (docker compose up -d {name})")
