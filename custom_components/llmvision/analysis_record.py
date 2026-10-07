"""Persist one camera analysis record next to its clip and snapshots.

`llmvision.store_analysis_record` writes `analysis.json` into a directory below
/media/llmvision (the evidence directory of an analysis). Automations call it after
their notification decision, so the record holds the decision and the complete LLM
response for later analysis. The directory is confined to /media/llmvision and the
file name is fixed; the record must be JSON and at most MAX_RECORD_BYTES.
"""

from __future__ import annotations

import json
import os
import tempfile

from homeassistant.exceptions import ServiceValidationError

from .const import DOMAIN

MEDIA_ROOT = "/media"
RECORD_FILENAME = "analysis.json"
MAX_RECORD_BYTES = 4 * 1024 * 1024


def resolve_record_directory(directory) -> str:
    """Resolve a directory below /media/llmvision; relative paths are placed there."""
    if not isinstance(directory, str) or not directory.strip():
        raise ServiceValidationError("directory must not be empty")
    root = os.path.realpath(os.path.join(MEDIA_ROOT, DOMAIN))
    raw = directory.strip()
    candidate = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(root, raw))
    if candidate == root or not candidate.startswith(root + os.sep):
        raise ServiceValidationError(
            f"directory must resolve to a subdirectory of {root}"
        )
    return candidate


def serialize_record(record) -> bytes:
    """A dict, or a JSON string of a dict (what `| to_json` in a template gives)."""
    if isinstance(record, str):
        try:
            record = json.loads(record)
        except ValueError as err:
            raise ServiceValidationError("record must be a JSON object") from err
    if not isinstance(record, dict):
        raise ServiceValidationError("record must be a JSON object")
    try:
        data = json.dumps(
            record, ensure_ascii=False, indent=1, default=str, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as err:
        raise ServiceValidationError("record is not JSON-serializable") from err
    if len(data) > MAX_RECORD_BYTES:
        raise ServiceValidationError("record is too large")
    return data


def write_record(directory: str, data: bytes) -> str:
    """Executor job: atomically (re)write <directory>/analysis.json."""
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, RECORD_FILENAME)
    fd, tmp = tempfile.mkstemp(prefix=".analysis-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


async def async_store_analysis_record(hass, directory, record) -> dict:
    data = serialize_record(record)
    path = resolve_record_directory(directory)
    try:
        target = await hass.async_add_executor_job(write_record, path, data)
    except OSError as err:
        raise ServiceValidationError(
            f"could not write {RECORD_FILENAME}: {type(err).__name__}"
        ) from err
    return {"path": target, "bytes": len(data)}
