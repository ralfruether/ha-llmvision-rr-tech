"""Helpers for decoding camera streams into analysis frames (stream_analyzer_pro).

Pure functions and small classes so they can be unit tested without Home Assistant.
"""

from __future__ import annotations

import math
import re
from urllib.parse import unquote, urlsplit

from homeassistant.exceptions import ServiceValidationError

# JPEG settings for frames produced by the pro services. 4:4:4 chroma keeps the thin
# red guide polylines crisp; Pillow's default (q75, 4:2:0) smears them.
PRO_JPEG_OPTIONS = {"quality": 90, "subsampling": 0}
PRO_FFMPEG_JPEG_Q = 2

FRAME_SOURCE_SNAPSHOT = "snapshot"
FRAME_SOURCE_STREAM = "stream"
FRAME_SOURCES = (FRAME_SOURCE_SNAPSHOT, FRAME_SOURCE_STREAM)

ALLOWED_STREAM_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https")
ALLOWED_CLIP_EXTENSIONS = (".mp4", ".m4v", ".mov")

# Resource bounds for untrusted stream data
MAX_STREAM_FRAMES = 300
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_FRAME_PIXELS = 40_000_000
MAX_MALFORMED_FRAMES = 5
STREAM_READ_CHUNK = 256 * 1024
STREAM_STARTUP_TIMEOUT = 15.0
STREAM_STALL_TIMEOUT = 15.0
STREAM_GRACE = 30.0
PROCESS_STOP_TIMEOUT = 5.0
STDERR_TAIL_BYTES = 8192

_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/@'\"]+@")
_QUERY_VALUE_RE = re.compile(r"([?&][^=&\s'\"]+=)[^&\s'\"]*")


def redact(text, known_url=None) -> str:
    """Mask credentials (URL userinfo, query values, known secrets) in text."""
    if text is None:
        return ""
    result = str(text)
    if known_url:
        secrets = {known_url}
        try:
            parts = urlsplit(known_url)
            for value in (parts.username, parts.password):
                if value and len(value) >= 3:
                    secrets.update({value, unquote(value)})
        except ValueError:
            pass
        for secret in sorted(secrets, key=len, reverse=True):
            if secret == known_url:
                continue
            result = result.replace(secret, "***")
    result = _USERINFO_RE.sub(r"\1***@", result)
    result = _QUERY_VALUE_RE.sub(r"\1***", result)
    return result


def stream_scheme(url) -> str:
    """Return the lower-case URL scheme, or an empty string."""
    try:
        return (urlsplit(str(url)).scheme or "").lower()
    except ValueError:
        return ""


def stream_input_args(url) -> list[str]:
    """ffmpeg input options valid for the given stream URL."""
    if stream_scheme(url) in ("rtsp", "rtsps"):
        return ["-rtsp_transport", "tcp"]
    return []


def normalize_frame_source(value) -> str:
    """Validate the frame_source service field (default: snapshot)."""
    if value is None:
        return FRAME_SOURCE_SNAPSHOT
    normalized = str(value).strip().lower()
    if not normalized:
        return FRAME_SOURCE_SNAPSHOT
    if normalized not in FRAME_SOURCES:
        raise ServiceValidationError("frame_source must be 'snapshot' or 'stream'")
    return normalized


def coerce_number(value, name, minimum, maximum, integer=False, allow_none=True):
    """Convert a service field to a bounded finite number.

    Values reach ffmpeg arguments and filter graphs, so only plain numbers are accepted.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_none:
            return None
        raise ServiceValidationError(f"{name} is required")
    if isinstance(value, bool):
        raise ServiceValidationError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ServiceValidationError(f"{name} must be a number")
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ServiceValidationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    if integer:
        if number != int(number):
            raise ServiceValidationError(f"{name} must be a whole number")
        return int(number)
    return number


def stream_frame_rate(fps, interval) -> tuple[str, float]:
    """Return the ffmpeg fps filter value and the numeric rate.

    fps wins when set; otherwise the legacy capture interval is expressed as an exact
    fraction (e.g. 1/3).
    """
    if fps:
        rate = float(fps)
        return f"{rate:g}", rate
    whole = max(1, int(round(float(interval))))
    return f"1/{whole}", 1.0 / whole


def stream_frame_cap(duration, rate) -> int:
    """Upper bound of frames decoded from one camera stream."""
    return max(1, min(MAX_STREAM_FRAMES, math.ceil(float(duration) * rate) + 1))


def build_stream_capture_cmd(
    stream_url,
    duration,
    frame_rate,
    max_dimension,
    max_frames,
    jpeg_q=PRO_FFMPEG_JPEG_Q,
    clip_output_args=None,
) -> list[str]:
    """ffmpeg command that pipes MJPEG frames and optionally writes a clip.

    Both outputs share one input so the camera stream is opened only once. All
    numeric values must be validated by the caller.
    """
    side = int(max_dimension)
    scale = (
        f"scale='min({side},iw)':'min({side},ih)':force_original_aspect_ratio=decrease"
    )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        *stream_input_args(stream_url),
        "-i",
        stream_url,
    ]
    if clip_output_args:
        cmd += ["-map", "0:v:0", *clip_output_args]
    cmd += [
        "-map",
        "0:v:0",
        "-t",
        str(duration),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"fps={frame_rate},{scale}",
        "-frames:v",
        str(int(max_frames)),
        "-q:v",
        str(int(jpeg_q)),
        "-c:v",
        "mjpeg",
        "-f",
        "image2pipe",
        "pipe:1",
    ]
    return cmd


def redacted_command(cmd, stream_url) -> str:
    """Printable command with the stream URL and its credentials masked."""
    return redact(" ".join(str(part) for part in cmd), stream_url)


class JpegStreamSplitter:
    """Incrementally split a concatenated MJPEG byte stream into JPEG images.

    Bounded: bytes before a start-of-image marker are dropped, and a frame that grows
    beyond max_frame_bytes without an end-of-image marker is discarded as malformed.
    """

    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    def __init__(self, max_frame_bytes=MAX_FRAME_BYTES):
        self.max_frame_bytes = max_frame_bytes
        self.malformed = 0
        self._buffer = bytearray()
        self._scan_from = 0

    def feed(self, chunk) -> list[bytes]:
        """Add bytes and return every complete JPEG found."""
        self._buffer += chunk
        frames = []
        while True:
            soi = self._buffer.find(self.SOI)
            if soi == -1:
                # Keep one byte in case a marker is split across reads
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                self._scan_from = 0
                break
            if soi > 0:
                del self._buffer[:soi]
                self._scan_from = 0
            eoi = self._buffer.find(self.EOI, max(2, self._scan_from))
            if eoi == -1:
                if len(self._buffer) > self.max_frame_bytes:
                    self.malformed += 1
                    self._buffer.clear()
                    self._scan_from = 0
                else:
                    self._scan_from = max(2, len(self._buffer) - 1)
                break
            frames.append(bytes(self._buffer[: eoi + 2]))
            del self._buffer[: eoi + 2]
            self._scan_from = 0
        return frames
