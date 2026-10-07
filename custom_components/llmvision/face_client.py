"""Client for the optional local face identification service.

The service receives a recorded clip and returns validated household member names
with the face positions over time. Everything returned by the service is treated as
untrusted input: names, numbers and booleans are validated strictly, and failures
map to a fixed set of reasons so that no exception text, token or response content
ever reaches logs, prompts or service responses.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_FACE_SERVICE_TOKEN,
    CONF_FACE_SERVICE_URL,
    CONF_PROVIDER,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

IDENTIFY_PATH = "/v1/identify"
REQUEST_TIMEOUT = 20
MAX_CLIP_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_PERSONS = 10
MAX_SAMPLES = 500
MAX_SAMPLE_TIME = 3600.0
# The face service samples 3 faces per second, so a visible face always has a sample
# within ~0.17 s; a wider window borrows positions from moments the person has moved on.
LABEL_TIME_WINDOW = 0.25
HEAD_BOX_SCALE = 1.6
SCORE_DECIMALS = 3

NAME_PATTERN = re.compile(r"[a-z][a-z .'-]{0,39}")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9._~+/=-]{16,512}")
_SLUG_INVALID = re.compile(r"[^A-Za-z0-9_.-]")
_SLUG_MAX = 64
_GENERIC_STEMS = {"clip", "video", "recording"}
_PLAIN_HTTP_SUFFIXES = (".local", ".lan")

STATUS_OK = "ok"
STATUS_DISABLED = "disabled"
REASON_TIMEOUT = "timeout"
REASON_CONNECTION = "connection"
REASON_REDIRECT = "redirect"
REASON_INVALID_RESPONSE = "invalid_response"
REASON_TOO_LARGE = "too_large"
REASON_NO_CLIP = "no_clip"


class FaceServiceError(Exception):
    """Failure with a fixed, log-safe reason (never carries response data)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class FaceSettings:
    url: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class FaceSample:
    t: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class FacePerson:
    name: str
    score: float
    samples: tuple[FaceSample, ...] = ()


@dataclass(frozen=True)
class AppearanceMatch:
    """A person named by appearance (mostly clothing), not by face. Shadow mode: reported
    only, never added to the prompt, the frame labels or `persons`."""

    name: str
    score: float
    tier: str
    qualifies: bool
    seed_face_score: float
    seed_age_s: float
    same_camera: bool


@dataclass(frozen=True)
class AppearanceInfo:
    status: str
    matches: tuple[AppearanceMatch, ...] = ()
    person_tracks: int | None = None
    unnamed_tracks: int | None = None


@dataclass
class FaceResult:
    camera: str
    status: str
    persons: list[FacePerson] = field(default_factory=list, repr=False)
    has_faces: bool = False
    has_unknown: bool = False
    elapsed_ms: int = 0
    appearance: AppearanceInfo | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


# ------------------------------------------------------------------ settings


def normalize_face_service_url(value) -> str:
    """Return the normalized service URL, "" when unset; raise ValueError if invalid.

    Plain http is only accepted for private/loopback IP addresses and .local/.lan
    hosts; https is accepted for any host. Credentials, query and fragment are
    rejected so the URL cannot smuggle data or change the request target.
    """
    url = str(value or "").strip()
    while url.endswith("/"):
        url = url[:-1]
    if not url:
        return ""
    if any(ch.isspace() for ch in url) or "?" in url or "#" in url:
        raise ValueError("invalid_face_service_url")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("invalid_face_service_url")
    if "@" in parts.netloc or parts.query or parts.fragment:
        raise ValueError("invalid_face_service_url")
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError("invalid_face_service_url")
    try:
        parts.port
    except ValueError as err:
        raise ValueError("invalid_face_service_url") from err
    if scheme == "http" and not _is_lan_host(host):
        raise ValueError("invalid_face_service_url")
    return url


def _is_lan_host(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return any(
            host.endswith(suffix) and len(host) > len(suffix)
            for suffix in _PLAIN_HTTP_SUFFIXES
        )
    if address.is_unspecified or address.is_multicast:
        return False
    return address.is_private or address.is_loopback


def is_valid_face_service_token(value) -> bool:
    return isinstance(value, str) and TOKEN_PATTERN.fullmatch(value) is not None


def get_face_settings(hass) -> FaceSettings | None:
    """Read the face service settings from the Settings entry; None = disabled."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        data = entry.data or {}
        if data.get(CONF_PROVIDER) != "Settings":
            continue
        try:
            url = normalize_face_service_url(data.get(CONF_FACE_SERVICE_URL))
        except ValueError:
            return None
        token = data.get(CONF_FACE_SERVICE_TOKEN)
        if not url or not is_valid_face_service_token(token):
            return None
        return FaceSettings(url=url, token=token)
    return None


# ------------------------------------------------------------------ camera slug


def _sanitize_slug(value: str) -> str:
    slug = _SLUG_INVALID.sub("_", value or "")[:_SLUG_MAX]
    return slug if slug.strip("._-") else "camera"


def camera_slug_for_entity(entity_id: str) -> str:
    """Camera slug for a camera entity id (camera.front_door -> front_door)."""
    entity_id = str(entity_id or "")
    if entity_id.startswith("camera."):
        entity_id = entity_id[len("camera.") :]
    return _sanitize_slug(entity_id)


def camera_slug_for_video(video_path: str) -> str:
    """Camera slug for a video path or URL.

    Uses the file stem; generic stems (clip, video, recording) use the parent
    directory cut before the first "_<digit>" (haustuer_20260926T1801/clip.mp4 ->
    haustuer).
    """
    path = str(video_path or "").strip()
    if "://" in path or path.startswith("/api"):
        path = urlsplit(path).path
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.lower() in _GENERIC_STEMS:
        parent = os.path.basename(os.path.dirname(path))
        stem = re.split(r"_\d", parent, maxsplit=1)[0] if parent else ""
    return _sanitize_slug(stem)


# ------------------------------------------------------------------ validation


def _finite_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _parse_sample(raw) -> FaceSample | None:
    if not isinstance(raw, dict):
        return None
    t = _finite_number(raw.get("t"))
    if t is None or not 0.0 <= t <= MAX_SAMPLE_TIME:
        return None
    box_raw = raw.get("box")
    if not isinstance(box_raw, list) or len(box_raw) != 4:
        return None
    box = tuple(_finite_number(value) for value in box_raw)
    if any(value is None or not 0.0 <= value <= 1.0 for value in box):
        return None
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2):
        return None
    return FaceSample(t=t, box=(x1, y1, x2, y2))


def _parse_person(raw) -> FacePerson | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or NAME_PATTERN.fullmatch(name) is None:
        return None
    score = _finite_number(raw.get("score"))
    if score is None or not 0.0 <= score <= 1.0:
        return None
    samples_raw = raw.get("samples", [])
    if not isinstance(samples_raw, list):
        return None
    samples = []
    for sample_raw in samples_raw:
        sample = _parse_sample(sample_raw)
        if sample is not None:
            samples.append(sample)
            if len(samples) >= MAX_SAMPLES:
                break
    return FacePerson(name=name, score=score, samples=tuple(samples))


def parse_identify_payload(payload) -> tuple[list[FacePerson], bool, bool]:
    """Validate a decoded /v1/identify body -> (persons, has_faces, has_unknown).

    Invalid persons and samples are dropped; a structurally invalid body raises
    FaceServiceError(invalid_response). Unknown fields are ignored.
    """
    if not isinstance(payload, dict):
        raise FaceServiceError(REASON_INVALID_RESPONSE)
    has_faces = payload.get("has_faces")
    has_unknown = payload.get("has_unknown")
    persons_raw = payload.get("persons")
    if (
        not isinstance(has_faces, bool)
        or not isinstance(has_unknown, bool)
        or not isinstance(persons_raw, list)
    ):
        raise FaceServiceError(REASON_INVALID_RESPONSE)
    persons = []
    seen = set()
    for person_raw in persons_raw:
        person = _parse_person(person_raw)
        if person is None or person.name in seen:
            continue
        seen.add(person.name)
        persons.append(person)
        if len(persons) >= MAX_PERSONS:
            break
    return persons, has_faces, has_unknown


def _reject_constant(_value):
    raise ValueError("non-finite number")


APPEARANCE_STATUSES = {"ok", "skipped", "disabled", "error"}
APPEARANCE_TIERS = {"strong", "weak"}
MAX_TRACKS = 1000
MAX_SEED_AGE = 86400.0
# A match may only ever "qualify" with a strong seed of at most this age, whatever the
# service reports (decision of 2026-10-06; defense in depth for a later alarm gate).
QUALIFY_MIN_SEED_SCORE = 0.50
QUALIFY_MAX_SEED_AGE = 3600.0


def _bounded_int(value, upper: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
        return None
    return value


def _parse_appearance_match(raw, face_names) -> AppearanceMatch | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or NAME_PATTERN.fullmatch(name) is None or name in face_names:
        return None
    score = _finite_number(raw.get("score"))
    tier = raw.get("tier")
    qualifies = raw.get("qualifies")
    seed = raw.get("seed")
    if (
        score is None
        or not 0.0 <= score <= 1.0
        or not isinstance(tier, str)
        or tier not in APPEARANCE_TIERS
        or not isinstance(qualifies, bool)
        or not isinstance(seed, dict)
    ):
        return None
    face_score = _finite_number(seed.get("face_score"))
    age = _finite_number(seed.get("age_s"))
    same_camera = seed.get("same_camera")
    if (
        face_score is None
        or not 0.0 <= face_score <= 1.0
        or age is None
        or not 0.0 <= age <= MAX_SEED_AGE
        or not isinstance(same_camera, bool)
    ):
        return None
    qualifies = (
        qualifies
        and tier == "strong"
        and face_score >= QUALIFY_MIN_SEED_SCORE
        and age <= QUALIFY_MAX_SEED_AGE
    )
    return AppearanceMatch(
        name=name,
        score=score,
        tier=tier,
        qualifies=qualifies,
        seed_face_score=face_score,
        seed_age_s=age,
        same_camera=same_camera,
    )


def parse_appearance_payload(payload, face_names=()) -> AppearanceInfo | None:
    """Validate the optional re-identification part of a /v1/identify body.

    Returns None when the service reported no (or an invalid) appearance part; the face
    result is never affected. Names that face recognition reported are ignored.
    """
    try:
        return _parse_appearance(payload, set(face_names))
    except Exception:  # noqa: BLE001 - a broken optional part must never fail the face result
        return None


def _parse_appearance(payload, face_names: set) -> AppearanceInfo | None:
    if not isinstance(payload, dict) or "reid" not in payload:
        return None
    status = payload.get("reid")
    if not isinstance(status, str) or status not in APPEARANCE_STATUSES:
        return None
    if status != "ok":  # counts are only meaningful when matching ran (the service sends null)
        return AppearanceInfo(status=status)
    matches_raw = payload.get("appearance_matches")
    persons = _bounded_int(payload.get("person_tracks"), MAX_TRACKS)
    unnamed = _bounded_int(payload.get("unnamed_tracks"), MAX_TRACKS)
    if (
        not isinstance(matches_raw, list)
        or persons is None
        or unnamed is None
        or unnamed > persons
    ):
        return None
    matches = []
    seen = set()
    for raw in matches_raw:
        match = _parse_appearance_match(raw, face_names)
        if match is None or match.name in seen:
            continue
        seen.add(match.name)
        matches.append(match)
        if len(matches) >= MAX_PERSONS:
            break
    return AppearanceInfo(
        status=status,
        matches=tuple(matches),
        person_tracks=persons,
        unnamed_tracks=unnamed,
    )


def decode_identify_response(
    body: bytes,
) -> tuple[list[FacePerson], bool, bool, AppearanceInfo | None]:
    """Decode a response body: face result plus the optional appearance part."""
    if len(body) > MAX_RESPONSE_BYTES:
        raise FaceServiceError(REASON_INVALID_RESPONSE)
    try:
        payload = json.loads(body, parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as err:
        raise FaceServiceError(REASON_INVALID_RESPONSE) from err
    persons, has_faces, has_unknown = parse_identify_payload(payload)
    appearance = parse_appearance_payload(payload, {p.name for p in persons})
    return persons, has_faces, has_unknown, appearance


def decode_identify_body(body: bytes) -> tuple[list[FacePerson], bool, bool]:
    """Decode and validate a response body; never echoes its content."""
    return decode_identify_response(body)[:3]


# ------------------------------------------------------------------ request


def _read_clip(clip_path: str) -> bytes:
    """Executor job: size-check and read the clip."""
    try:
        if not os.path.isfile(clip_path):
            raise FaceServiceError(REASON_NO_CLIP)
        size = os.path.getsize(clip_path)
        if size <= 0:
            raise FaceServiceError(REASON_NO_CLIP)
        if size > MAX_CLIP_BYTES:
            raise FaceServiceError(REASON_TOO_LARGE)
        with open(clip_path, "rb") as handle:
            data = handle.read(MAX_CLIP_BYTES + 1)
    except OSError as err:
        raise FaceServiceError(REASON_NO_CLIP) from err
    if not data:
        raise FaceServiceError(REASON_NO_CLIP)
    if len(data) > MAX_CLIP_BYTES:
        raise FaceServiceError(REASON_TOO_LARGE)
    # Only ISO-BMFF video (mp4/m4v/mov) is uploaded, never arbitrary local files
    if data[4:8] != b"ftyp":
        raise FaceServiceError(REASON_NO_CLIP)
    return data


async def async_load_clip(hass, clip_path) -> bytes:
    """Read a clip without blocking the event loop; raises FaceServiceError."""
    if not clip_path or not isinstance(clip_path, str):
        raise FaceServiceError(REASON_NO_CLIP)
    return await hass.loop.run_in_executor(None, _read_clip, clip_path)


async def _read_limited(response) -> bytes:
    body = bytearray()
    while len(body) <= MAX_RESPONSE_BYTES:
        chunk = await response.content.read(MAX_RESPONSE_BYTES + 1 - len(body))
        if not chunk:
            break
        body += chunk
    if len(body) > MAX_RESPONSE_BYTES:
        raise FaceServiceError(REASON_INVALID_RESPONSE)
    return bytes(body)


async def _async_post_clip(hass, settings: FaceSettings, camera: str, clip: bytes):
    session = async_get_clientsession(hass)
    headers = {
        "Content-Type": "video/mp4",
        "Authorization": f"Bearer {settings.token}",
    }
    async with session.post(
        f"{settings.url}{IDENTIFY_PATH}",
        params={"camera": camera},
        data=clip,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        allow_redirects=False,
    ) as response:
        status = int(response.status)
        if 300 <= status < 400:
            raise FaceServiceError(REASON_REDIRECT)
        if not 200 <= status < 300:
            if not 100 <= status <= 599:
                raise FaceServiceError(REASON_INVALID_RESPONSE)
            raise FaceServiceError(f"http_{status}")
        if (response.content_type or "").lower() != "application/json":
            raise FaceServiceError(REASON_INVALID_RESPONSE)
        body = await _read_limited(response)
    return decode_identify_response(body)


def error_result(camera: str, reason: str, elapsed_ms: int = 0) -> FaceResult:
    """Build a failure result and log one warning with the fixed reason only."""
    _LOGGER.warning(
        "Face service identification failed for camera '%s' (error:%s); "
        "continuing without face identification",
        camera,
        reason,
    )
    return FaceResult(camera=camera, status=f"error:{reason}", elapsed_ms=elapsed_ms)


async def async_identify(
    hass,
    settings: FaceSettings,
    camera: str,
    *,
    clip_path: str | None = None,
    clip_data: bytes | None = None,
) -> FaceResult:
    """Identify known persons in a clip. Never raises except on cancellation."""
    started = time.monotonic()
    try:
        if clip_data is None:
            clip_data = await async_load_clip(hass, clip_path)
        persons, has_faces, has_unknown, appearance = await _async_post_clip(
            hass, settings, camera, clip_data
        )
    except asyncio.CancelledError:
        raise
    except FaceServiceError as err:
        reason = err.reason
    except (asyncio.TimeoutError, TimeoutError):
        reason = REASON_TIMEOUT
    except (aiohttp.ClientError, OSError):
        reason = REASON_CONNECTION
    except Exception:  # noqa: BLE001 - the face service must never fail the analysis
        reason = REASON_INVALID_RESPONSE
    else:
        return FaceResult(
            camera=camera,
            status=STATUS_OK,
            persons=persons,
            has_faces=has_faces,
            has_unknown=has_unknown,
            elapsed_ms=_elapsed_ms(started),
            appearance=appearance,
        )
    return error_result(camera, reason, _elapsed_ms(started))


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.monotonic() - started) * 1000)))


# ------------------------------------------------------------------ labels


def nearest_sample(
    samples, t: float, window: float = LABEL_TIME_WINDOW
) -> FaceSample | None:
    """Sample closest to t within +/- window seconds, else None."""
    best = None
    best_delta = None
    for sample in samples:
        delta = abs(sample.t - t)
        if delta <= window and (best_delta is None or delta < best_delta):
            best, best_delta = sample, delta
    return best


def fps_frame_time(index: int, rate: float) -> float:
    """Clip time shown by output frame `index` of an ffmpeg `fps=rate` filter.

    The filter rounds input timestamps to the nearest output slot and keeps the last
    input frame of each slot, so output frame n shows the scene at about
    (n + 0.5) / rate (minus one source frame), not n / rate. Measured on real
    recordings at fps=1: +0.47 s, which put name labels half a second behind.
    """
    return (index + 0.5) / rate


def labels_for_time(persons, t: float) -> list[tuple[str, tuple]]:
    """(name, normalized box) for every person seen near clip time t."""
    labels = []
    for person in persons:
        sample = nearest_sample(person.samples, t)
        if sample is not None:
            labels.append((person.name, sample.box))
    return labels


def head_box(box, width: int, height: int, scale: float = HEAD_BOX_SCALE):
    """Pixel rectangle of a face box enlarged around its centre, clamped to the image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    max_x, max_y = max(0, width - 1), max(0, height - 1)
    left = min(max_x, max(0, round((cx - half_w) * width)))
    top = min(max_y, max(0, round((cy - half_h) * height)))
    right = min(max_x, max(0, round((cx + half_w) * width)))
    bottom = min(max_y, max(0, round((cy + half_h) * height)))
    return left, top, right, bottom


# ------------------------------------------------------------------ prompt/response


def _identified(results) -> tuple[dict[str, float], bool]:
    scores: dict[str, float] = {}
    has_unknown = False
    for result in results or []:
        if not result.ok:
            continue
        has_unknown = has_unknown or result.has_unknown
        for person in result.persons:
            scores[person.name] = max(person.score, scores.get(person.name, 0.0))
    return scores, has_unknown


def _quoted(names) -> str:
    return ", ".join(f'"{name}"' for name in sorted(names))


def build_fact_text(results, labeled_names=()) -> str:
    """Prompt paragraph with the identified persons; "" when there is nothing to add.

    labeled_names are the names actually drawn on at least one model frame; only
    those are described as marked in the images.
    """
    scores, has_unknown = _identified(results)
    labeled = {name for name in scores if name in set(labeled_names or ())}
    unlabeled = set(scores) - labeled
    sentences = []
    if scores:
        if labeled:
            sentences.append(
                "Local face recognition identified household members, marked in "
                f"the images with a green box and their name: {_quoted(labeled)}."
            )
            if unlabeled:
                sentences.append(
                    "It also identified these household members in this clip: "
                    f"{_quoted(unlabeled)}."
                )
        else:
            sentences.append(
                "Local face recognition identified household members in this clip: "
                f"{_quoted(unlabeled)}."
            )
        sentences.append("Treat these persons as confirmed known persons.")
        sentences.append("Do not assign names to any other person.")
        if has_unknown:
            sentences.append(
                "At least one other face could not be identified (it may still be "
                "a household member seen at a bad angle)."
            )
    elif has_unknown:
        sentences.append(
            "Local face recognition could not identify at least one face (it may "
            "still be a household member seen at a bad angle)."
        )
    return " ".join(sentences)


def service_response_fields(enabled: bool, results) -> dict:
    """persons / face_service / face_service_ms fields for the service response."""
    if not enabled:
        return {"persons": [], "face_service": STATUS_DISABLED, "face_service_ms": 0}
    results = list(results or [])
    scores, _ = _identified(results)
    persons = [
        {"name": name, "score": round(score, SCORE_DECIMALS)}
        for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]
    failed = [result.status for result in results if not result.ok]
    if not results:
        status = f"error:{REASON_NO_CLIP}"
    elif failed:
        status = failed[0]
    else:
        status = STATUS_OK
    elapsed = max((result.elapsed_ms for result in results), default=0)
    return {"persons": persons, "face_service": status, "face_service_ms": elapsed}


REID_UNAVAILABLE = "unavailable"


def appearance_response_fields(enabled: bool, results) -> dict:
    """Shadow-mode re-identification fields for the service response.

    appearance_persons: best appearance match per name over all clips (never a name face
    recognition reported). person_tracks / unnamed_tracks are summed over the clips and are
    None unless every clip reported ReID counts, so a check such as
    "unnamed_tracks == 0" can never pass on missing data.
    """
    results = list(results or [])
    infos = [result.appearance if result.ok else None for result in results]
    if not enabled or not infos:
        status = STATUS_DISABLED if not enabled else REID_UNAVAILABLE
        return {"appearance_persons": [], "person_tracks": None, "unnamed_tracks": None,
                "reid": status}
    face_names, _ = _identified(results)
    best: dict[str, AppearanceMatch] = {}
    for info in infos:
        for match in info.matches if info else ():
            if match.name in face_names:
                continue
            current = best.get(match.name)
            if current is None or (match.qualifies, match.score) > (current.qualifies, current.score):
                best[match.name] = match
    complete = all(info is not None and info.unnamed_tracks is not None for info in infos)
    statuses = [info.status if info else REID_UNAVAILABLE for info in infos]
    status = next((s for s in statuses if s != "ok"), "ok")
    return {
        "appearance_persons": [
            {
                "name": m.name,
                "score": round(m.score, SCORE_DECIMALS),
                "tier": m.tier,
                "qualifies": m.qualifies,
                "seed_face_score": round(m.seed_face_score, SCORE_DECIMALS),
                "seed_age_s": round(m.seed_age_s),
                "same_camera": m.same_camera,
            }
            for m in sorted(best.values(), key=lambda m: (-m.score, m.name))
        ],
        "person_tracks": sum(i.person_tracks for i in infos) if complete else None,
        "unnamed_tracks": sum(i.unnamed_tracks for i in infos) if complete else None,
        "reid": status,
    }
