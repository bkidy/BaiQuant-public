from __future__ import annotations


def strip_exchange(code: str) -> str:
    """Return the six-digit stock code without exchange suffix."""
    return str(code).split(".", maxsplit=1)[0].zfill(6)


def with_exchange(code: str) -> str:
    """Normalize an A-share code to the local six-digit.SZ/SH/BJ format."""
    value = str(code).strip().upper()
    if "." in value:
        raw, exchange = value.split(".", maxsplit=1)
        return f"{raw.zfill(6)}.{exchange}"
    raw = value.zfill(6)
    if raw.startswith(("8", "4")):
        exchange = "BJ"
    elif raw.startswith(("5", "6", "9")):
        exchange = "SH"
    else:
        exchange = "SZ"
    return f"{raw}.{exchange}"
