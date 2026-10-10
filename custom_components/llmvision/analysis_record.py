"""Persist one camera analysis record next to its clip and snapshots.

`llmvision.store_analysis_record` writes `analysis.json` into a directory below
/media/llmvision (the evidence directory of an analysis). Automations call it after
their notification decision, so the record holds the decision and the complete LLM
response for later analysis. The directory is confined to /media/llmvision and the
file name is fixed; the record must be JSON and at most MAX_RECORD_BYTES.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import tempfile

from homeassistant.exceptions import ServiceValidationError

from .const import DOMAIN

MEDIA_ROOT = "/media"
RECORD_FILENAME = "analysis.json"
MAX_RECORD_BYTES = 4 * 1024 * 1024
# Read access (admin-only HTTP views): only <MEDIA_ROOT>/llmvision/camera-analysis/<id>/.
# Ids are camera + timestamp, i.e. presence metadata; records hold identification data.
RECORDS_SUBDIR = "camera-analysis"
RECORD_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}_[0-9]{8}T[0-9]{6}_[0-9]{1,9}")
MAX_LISTED_RECORDS = 1000


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


# ------------------------------------------------------------------ read access


class RecordTooLarge(Exception):
    """The stored record exceeds MAX_RECORD_BYTES."""


def records_root() -> str:
    return os.path.join(MEDIA_ROOT, DOMAIN, RECORDS_SUBDIR)


def is_record_id(value) -> bool:
    return isinstance(value, str) and RECORD_ID_PATTERN.fullmatch(value) is not None


def list_records(since: float, limit: int) -> tuple[list[dict], bool]:
    """Executor job: records modified at or after `since`, oldest first.

    Only direct subdirectories with a valid id and a regular analysis.json count;
    symlinks are never followed. Returns (records, truncated).
    """
    found = []
    try:
        with os.scandir(records_root()) as entries:
            for entry in entries:
                if not is_record_id(entry.name) or not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    info = os.lstat(os.path.join(entry.path, RECORD_FILENAME))
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode) and info.st_mtime >= since:
                    found.append(
                        {"id": entry.name, "mtime": info.st_mtime, "bytes": info.st_size}
                    )
    except (FileNotFoundError, NotADirectoryError):
        return [], False
    found.sort(key=lambda r: (r["mtime"], r["id"]))
    return found[:limit], len(found) > limit


def read_record(analysis_id: str) -> bytes | None:
    """Executor job: the bytes of <id>/analysis.json, or None if there is none.

    The id directory and the file are opened relative to the records root without
    following symlinks (no path is re-resolved between check and read), and the file
    without blocking, so a FIFO cannot hang the executor.
    """
    if not is_record_id(analysis_id):
        raise ValueError("invalid analysis id")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(records_root(), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as err:
        if _is_missing(err):
            return None
        raise
    try:
        try:
            dir_fd = os.open(
                analysis_id, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=root_fd
            )
        except OSError as err:
            if _is_missing(err):
                return None
            raise
        try:
            try:
                fd = os.open(
                    RECORD_FILENAME,
                    os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=dir_fd,
                )
            except OSError as err:
                if _is_missing(err):
                    return None
                raise
            try:
                info = os.fstat(fd)
            except OSError:
                os.close(fd)
                raise
            if not stat.S_ISREG(info.st_mode):
                os.close(fd)
                return None
            with os.fdopen(fd, "rb") as handle:
                if info.st_size > MAX_RECORD_BYTES:
                    raise RecordTooLarge
                data = handle.read(MAX_RECORD_BYTES + 1)
            if len(data) > MAX_RECORD_BYTES:
                raise RecordTooLarge
            return data
        finally:
            os.close(dir_fd)
    finally:
        os.close(root_fd)


def _is_missing(err: OSError) -> bool:
    """Missing, not a directory, or a symlink where none may be (O_NOFOLLOW)."""
    return err.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
