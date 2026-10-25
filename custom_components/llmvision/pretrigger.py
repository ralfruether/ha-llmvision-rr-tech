"""Pre-trigger frame for stream_analyzer_pro.

The last captured frame of a camera's previous run is kept in memory and sent first in
the camera's next run, so the model can tell an object that was already there from one
that just arrived. Only the frame, its capture time and the household members face
recognition identified in that run are carried over: never model output, never
appearance (ReID) matches.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

from . import face_client

MIN_MAX_AGE = 60
MAX_MAX_AGE = 3600
# A 2880x1616 stream frame is well below this; anything larger is not cached.
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 32
LABEL_SUFFIX = "vortrigger"


@dataclass
class PretriggerEntry:
    frame: bytes = field(repr=False)
    # Local wall clock (for the prompt) and monotonic clock (for the age) at capture
    captured_at: datetime
    monotonic: float
    # Provider entry the frame was captured for; never sent to another provider
    provider: str
    # None when recognition did not run or did not succeed for every clip
    names: tuple[str, ...] | None = field(default=None, repr=False)


def household_names(results) -> tuple[str, ...] | None:
    """Face-recognized names of a run; None when there is no result or any clip failed,
    so a failed lookup never reads as "nobody recognized"."""
    results = list(results or [])
    if not results or not all(result.ok for result in results):
        return None
    names = {person.name for result in results for person in result.persons}
    return tuple(
        sorted(name for name in names if face_client.NAME_PATTERN.fullmatch(name))
    )


def frame_label(image_entity: str, include_filename: bool) -> str:
    """Image tag in the style of the stream frame labels, from the sanitized slug."""
    prefix = (
        face_client.camera_slug_for_entity(image_entity)
        if include_filename
        else "camera0"
    )
    return f"{prefix}-{LABEL_SUFFIX}"


def usable_age(
    entry: PretriggerEntry | None, trigger_monotonic: float, max_age, provider: str
) -> int | None:
    """Whole seconds the entry was captured before the trigger; None when unusable
    (missing, captured after the trigger, too old, or for another provider)."""
    if entry is None or entry.provider != provider:
        return None
    age = trigger_monotonic - entry.monotonic
    if not 0 <= age <= max_age:
        return None
    return int(age)


class PretriggerCache:
    """Last frame per camera. Each call takes a sequence number on arrival; a call only
    replaces the slot of an earlier call, so a slow older call never overwrites a newer
    frame or a newer clear (option off, or a newer run without a usable frame)."""

    def __init__(self) -> None:
        self._seq = itertools.count()
        # entity -> (sequence, entry or None for "cleared", monotonic stamp for expiry)
        self._slots: dict[str, tuple[int, PretriggerEntry | None, float]] = {}

    def next_seq(self) -> int:
        return next(self._seq)

    def get(self, image_entity: str) -> PretriggerEntry | None:
        slot = self._slots.get(image_entity)
        return slot[1] if slot else None

    def publish(
        self,
        image_entity: str,
        seq: int,
        entry: PretriggerEntry | None,
        now_monotonic: float,
    ) -> bool:
        """Store entry (None clears) unless a later call already published."""
        slot = self._slots.get(image_entity)
        if slot is not None and slot[0] > seq:
            return False
        stamp = entry.monotonic if entry is not None else now_monotonic
        self._slots[image_entity] = (seq, entry, stamp)
        self.prune(now_monotonic)
        return True

    def prune(self, now_monotonic: float) -> None:
        """Drop slots no call may use any more and bound the number of cameras."""
        for key in [
            k for k, s in self._slots.items() if now_monotonic - s[2] > MAX_MAX_AGE
        ]:
            self._slots.pop(key, None)
        while len(self._slots) > MAX_ENTRIES:
            oldest = min(self._slots, key=lambda k: self._slots[k][2])
            self._slots.pop(oldest)

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, image_entity: str) -> bool:
        return image_entity in self._slots


def fact_text(label: str, entry: PretriggerEntry, age_s: int) -> str:
    """Prompt paragraph describing the pre-trigger frame."""
    sentences = [
        f'Pre-trigger frame: the first image, labeled "{label}", is the last frame of '
        "this camera's previous analysis, captured at "
        f"{entry.captured_at.strftime('%H:%M:%S')} local time, {age_s} s before this "
        "trigger. It shows the scene before this trigger and is not part of the "
        "current sequence: never count it as a frame of the current sequence and "
        "never list it in any frame field."
    ]
    if entry.names is None:
        sentences.append(
            "Whether face recognition identified a household member in that previous "
            "analysis is unknown."
        )
    elif entry.names:
        quoted = ", ".join(f'"{name}"' for name in entry.names)
        sentences.append(
            "In that previous analysis, local face recognition identified these "
            f"household members: {quoted}."
        )
    else:
        sentences.append(
            "In that previous analysis, local face recognition identified no "
            "household member."
        )
    return " ".join(sentences)
