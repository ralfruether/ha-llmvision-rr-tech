"""Absolute-time arithmetic and validation for timeline intervals."""

from datetime import date, datetime, time, timezone


def as_utc(value: date | datetime) -> datetime:
    """Normalize an instant, retaining the timeline's naive-means-UTC convention."""
    if not isinstance(value, datetime):
        if not isinstance(value, date):
            raise ValueError("Timeline endpoints must be dates or datetimes")
        value = datetime.combine(value, time.min)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def validate_interval(
    start: date | datetime, end: date | datetime
) -> tuple[datetime, datetime]:
    """Return UTC endpoints; allow zero duration, but never reversed instants."""
    start_utc, end_utc = as_utc(start), as_utc(end)
    if end_utc < start_utc:
        raise ValueError("Timeline event end must not precede start")
    return start_utc, end_utc
