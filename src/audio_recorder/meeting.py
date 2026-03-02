"""Meeting metadata loading and matching."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path

_TOLERANCE = timedelta(minutes=10)

_WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_time(s: str) -> time:
    """Parse an HH:MM string into a time object."""
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]))


def load_meetings(path: Path | str) -> list[dict]:
    """Load and validate meetings from a JSON file.

    Returns a list of meeting dicts. Raises on invalid structure.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    meetings = data.get("meetings", [])

    for m in meetings:
        if "name" not in m:
            raise ValueError(f"Meeting missing required field 'name': {m}")
        if "start_time" not in m or "end_time" not in m:
            raise ValueError(f"Meeting '{m['name']}' missing required 'start_time'/'end_time'")

        m["_start_time"] = _parse_time(m["start_time"])
        m["_end_time"] = _parse_time(m["end_time"])

        recurrence = m.get("recurrence", "weekly")
        if recurrence == "weekly" and "day" not in m:
            raise ValueError(f"Weekly meeting '{m['name']}' missing required 'day' field")
        if recurrence == "weekly":
            day = m["day"].lower()
            if day not in _WEEKDAY_NAMES:
                raise ValueError(f"Invalid day '{m['day']}' for meeting '{m['name']}'")
            m["_weekday"] = _WEEKDAY_NAMES[day]

    return meetings


def _matches_day(meeting: dict, dt: datetime) -> bool:
    """Check if the meeting occurs on the given date."""
    recurrence = meeting.get("recurrence", "weekly")

    if recurrence == "daily_weekdays":
        return dt.weekday() < 5  # Mon-Fri

    if recurrence == "weekly":
        return dt.weekday() == meeting["_weekday"]

    return False


def find_active_meeting(meetings: list[dict], at: datetime | None = None) -> dict | None:
    """Find a meeting whose time window contains the given time.

    Checks recurrence (daily_weekdays or weekly) and applies a 10-minute
    tolerance before start and after end.
    Returns the meeting dict or None.
    """
    at = at or datetime.now()

    for m in meetings:
        if not _matches_day(m, at):
            continue

        start_dt = datetime.combine(at.date(), m["_start_time"])
        end_dt = datetime.combine(at.date(), m["_end_time"])

        if start_dt - _TOLERANCE <= at <= end_dt + _TOLERANCE:
            return m

    return None


def format_meeting_context(meeting: dict) -> str:
    """Format a meeting dict into a markdown section for prompt injection."""
    lines = [
        "## Meeting Context",
        "",
        f"This recording is from the meeting: **{meeting['name']}**",
    ]

    if meeting.get("description"):
        lines.append(f"- **Description:** {meeting['description']}")

    start = meeting["_start_time"]
    end = meeting["_end_time"]
    lines.append(f"- **Scheduled:** {start.strftime('%H:%M')} – {end.strftime('%H:%M')}")

    if meeting.get("participants"):
        lines.append(f"- **Participants:** {', '.join(meeting['participants'])}")

    if meeting.get("agenda"):
        lines.append("- **Expected Agenda:**")
        for item in meeting["agenda"]:
            lines.append(f"  - {item}")

    lines.extend([
        "",
        "Use this context to improve your summary:",
        "- Attribute statements to the listed participants when possible based on voice/context clues",
        "- Track which agenda items were covered and which were skipped",
        "- Note any significant off-agenda topics that came up",
    ])

    return "\n".join(lines)
