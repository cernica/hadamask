"""Friendly errors for optional dependencies (``pip install "hadamask[<extra>]"``)."""

from __future__ import annotations


def missing_extra(extra: str, exc: ImportError) -> ImportError:
    """An ImportError telling the user which extra to install."""
    module = exc.name or str(exc)
    return ImportError(
        f"this part of hadamask needs the optional dependency {module!r}, which is not installed. "
        f'Install it with: pip install "hadamask[{extra}]"'
    )
