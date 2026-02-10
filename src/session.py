"""
Session-level helpers: device-specific max session length for early-failure checks.
Use with CGM session ID when grouping; reading-level masks are in instability.py.
"""

from typing import Optional


def max_session_days(device_id: Optional[str]) -> Optional[float]:
    """
    Maximum expected session length in days for the given source device ID.

    - If device_id contains "G7" → 10.5 days.
    - If device_id contains "G6" and not "G7" → 10 days.
    - Otherwise → None (unknown device).

    Use when flagging sessions that ended before max (e.g. potential failure).
    """
    if device_id is None:
        return None
    s = str(device_id).upper()
    if "G7" in s:
        return 10.5
    if "G6" in s:
        return 10.0
    return None
