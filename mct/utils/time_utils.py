"""
Time utility functions for MCT system.
"""
from datetime import datetime, timezone, timedelta


def get_vn_time():
    """Get current time in Vietnam timezone (UTC+7)."""
    vn_tz = timezone(timedelta(hours=7))
    return datetime.now(vn_tz)
