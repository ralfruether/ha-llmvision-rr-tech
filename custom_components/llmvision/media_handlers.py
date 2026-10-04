import base64
import io
import os
import uuid
import logging
import time
import asyncio
import tempfile
from aiofile import async_open
from datetime import timedelta
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.components.http.auth import async_sign_path
from homeassistant.components.media_source import is_media_source_id
from homeassistant.components.media_player.browse_media import (
    async_process_play_media_url,
)

from urllib.parse import urlparse
from functools import partial
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import numpy as np
from homeassistant.helpers.network import get_url
from homeassistant.exceptions import ServiceValidationError

from . import face_client
from .const import DOMAIN
from .timeline import mark_key_frame_pending, release_key_frame
from .stream_capture import (
    ALLOWED_CLIP_EXTENSIONS,
    ALLOWED_STREAM_SCHEMES,
    FRAME_SOURCE_STREAM,
    MAX_FRAME_PIXELS,
    MAX_MALFORMED_FRAMES,
    MAX_STREAM_FRAMES,
    PROCESS_STOP_TIMEOUT,
    RECORD_CODEC_COPY,
    RECORD_CODEC_H264,
    REMUX_TIMEOUT,
    STDERR_TAIL_BYTES,
    STREAM_GRACE,
    STREAM_READ_CHUNK,
    STREAM_STALL_TIMEOUT,
    STREAM_STARTUP_TIMEOUT,
    JpegStreamSplitter,
    build_hvc1_remux_cmd,
    build_stream_capture_cmd,
    coerce_number,
    mp4_video_sample_entry,
    normalize_frame_source,
    normalize_record_codec,
    redact,
    redacted_command,
    stream_frame_cap,
    stream_frame_rate,
    stream_input_args,
    stream_scheme,
)

_LOGGER = logging.getLogger(__name__)


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


async def _async_get_stream_source(hass, entity_id):
    """Lazily resolve a camera's stream source.

    Imported on demand so the camera component (and its heavy optional deps) is
    only loaded when a clip is actually recorded.
    """
    from homeassistant.components.camera import async_get_stream_source

    return await async_get_stream_source(hass, entity_id)


FACE_LABEL_COLOR = (0, 255, 0)


def _label_font(size):
    """Pillow's bundled default font at a readable size (bitmap font fallback)."""
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, AttributeError):
        return ImageFont.load_default()


def _draw_name_label(draw, rect, name, line_width, width, height, font):
    """Draw a green head box and the name on a green tag above it."""
    left, top, right, bottom = rect
    draw.rectangle((left, top, right, bottom), outline=FACE_LABEL_COLOR, width=line_width)
    text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), name, font=font)
    pad = max(2, line_width)
    tag_w = text_right - text_left + 2 * pad
    tag_h = text_bottom - text_top + 2 * pad
    x = max(0, min(left, width - tag_w))
    y = top - tag_h
    if y < 0:
        y = min(max(0, height - tag_h), bottom + 1)
    draw.rectangle((x, y, x + tag_w, y + tag_h), fill=FACE_LABEL_COLOR)
    draw.text((x + pad - text_left, y + pad - text_top), name, fill=(0, 0, 0), font=font)


class MediaProcessor:
    def __init__(self, hass, client):
        self.hass = hass
        self.session = async_get_clientsession(self.hass)
        self.client = client
        self.base64_images = []
        self.filenames = []
        self.snapshots_path = f"/media/{DOMAIN}/snapshots/"
        self.key_frame = ""
        # Populated by stream_analyzer_pro when debug_polylines is enabled
        self.debug_info = None
        # Populated by stream_analyzer_pro when a clip is recorded
        self.clip_paths = []
        self.requested_clip_paths = []
        self.clip_task = None
        # record_codec of the call: h264 transcode or copy of the original stream
        self.clip_codec = RECORD_CODEC_H264
        # JPEG encoder options; the pro services opt into higher quality
        self.jpeg_options = {}
        self.ffmpeg_jpeg_q = 5
        # Optional face identification, set by the pro service handlers
        self.identify_persons = False
        self.face_settings = None
        self.face_results = []
        self.face_labeled_names = set()
        self._face_done = False
        # Snapshot mode: clip bytes (or failure reason) read under the camera lock
        self._face_clip_data = None
        # Stream mode: frame label -> (camera entity, raw ffmpeg output index)
        self._face_stream_frames = {}
        # Camera entity -> requested clip path (None when no clip is recorded)
        self._face_clip_plan = {}

    @property
    def face_active(self):
        """True when identify_persons is requested and the service is configured."""
        return bool(self.identify_persons and self.face_settings is not None)

    async def _encode_image(self, img):
        """Encode image as base64"""
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG", **self.jpeg_options)
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        return base64_image

    async def _save_clip(
        self, clip_data=None, clip_path=None, image_data=None, image_path=None
    ):
        _LOGGER.debug(f"Saving clip to {clip_path} and image to {image_path}")
        # Ensure dir exists
        await self.hass.loop.run_in_executor(
            None, partial(os.makedirs, self.snapshots_path, exist_ok=True)
        )

        def _run_save_clips(clip_data, clip_path, image_data, image_path):
            _LOGGER.info(f"[save_clip] clip: {clip_path}, image: {image_path}")
            if image_data:
                with open(image_path, "wb") as f:
                    if type(image_data) == bytes:
                        f.write(image_data)
                    else:
                        f.write(base64.b64decode(image_data))
            elif clip_data:
                with open(clip_path, "wb") as f:
                    f.write(clip_data)

        await self.hass.loop.run_in_executor(
            None, _run_save_clips, clip_data, clip_path, image_data, image_path
        )

    def _convert_to_rgb(self, img):
        if img.mode == "RGBA" or img.format == "GIF":
            img = img.convert("RGB")
        return img

    def release_key_frame(self):
        """Allow timeline cleanup to manage the exposed key frame again."""
        release_key_frame(self.hass, self.key_frame)

    async def _expose_image(self, frame_name, image_data, uid, frame_path=None):
        # ensure /media/llmvision/snapshots dir exists
        await self.hass.loop.run_in_executor(
            None,
            partial(os.makedirs, f"/media/{DOMAIN}/snapshots", exist_ok=True),
        )
        if self.key_frame == "":
            filename = f"/media/{DOMAIN}/snapshots/{uid}-{frame_name}.jpg"
            self.key_frame = filename
            # Protect the snapshot from timeline cleanup while the request runs.
            mark_key_frame_pending(self.hass, filename)
            if image_data is None and frame_path is not None:
                # open image in hass.loop
                with await self.hass.loop.run_in_executor(
                    None, Image.open, frame_path
                ) as image:
                    await self.hass.loop.run_in_executor(None, image.load)
                    image_data = await self._encode_image(image)
            await self._save_clip(image_data=image_data, image_path=filename)

    def _similarity_score(self, previous_frame, current_frame_gray):
        """
        SSIM by Z. Wang: https://ece.uwaterloo.ca/~z70wang/research/ssim/
        Paper:  Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
        "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
        """
        K1 = 0.005
        K2 = 0.015
        L = 255

        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        previous_frame_np = np.array(previous_frame)
        current_frame_np = np.array(current_frame_gray)

        # Ensure both frames have same dimensions
        if previous_frame_np.shape != current_frame_np.shape:
            min_shape = np.minimum(previous_frame_np.shape, current_frame_np.shape)
            previous_frame_np = previous_frame_np[: min_shape[0], : min_shape[1]]
            current_frame_np = current_frame_np[: min_shape[0], : min_shape[1]]

        # Calculate mean (mu)
        mu1 = np.mean(previous_frame_np, dtype=np.float64)
        mu2 = np.mean(current_frame_np, dtype=np.float64)

        # Calculate variance (sigma^2) and covariance (sigma12)
        sigma1_sq = np.var(previous_frame_np, dtype=np.float64, mean=mu1)
        sigma2_sq = np.var(current_frame_np, dtype=np.float64, mean=mu2)
        sigma12 = np.cov(
            previous_frame_np.flatten(), current_frame_np.flatten(), dtype=np.float64
        )[0, 1]

        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim

    async def _select_keyframe_index(
        self, reference_frame_bytes, candidate_frames_bytes
    ):
        """
        Pick the index of the frame most different from the reference frame.
        """
        # Decode reference to grayscale
        ref_img = Image.open(io.BytesIO(reference_frame_bytes))
        try:
            await self.hass.loop.run_in_executor(None, ref_img.load)
            ref_gray = np.array(ref_img.convert("L"))
        finally:
            ref_img.close()

        best_idx = 0
        best_score = float("inf")  # minimize SSIM
        for idx, frame_bytes in enumerate(candidate_frames_bytes):
            img = Image.open(io.BytesIO(frame_bytes))
            try:
                await self.hass.loop.run_in_executor(None, img.load)
                curr_gray = np.array(img.convert("L"))
            finally:
                img.close()
            score = self._similarity_score(ref_gray, curr_gray)
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    async def resize_image(
        self, target_width, image_path=None, image_data=None, img=None
    ):
        """Resize image to target_width"""
        base64_image = None

        if image_path:
            # Open the image file
            img = await self.hass.loop.run_in_executor(None, Image.open, image_path)
            with img:
                await self.hass.loop.run_in_executor(None, img.load)
                # Check if the image is a GIF and convert if necessary
                img = self._convert_to_rgb(img)
                # calculate new height based on aspect ratio
                width, height = img.size
                aspect_ratio = width / height
                target_height = int(target_width / aspect_ratio)

                # Resize the image only if it's larger than the target size
                if width > target_width or height > target_height:
                    img = img.resize((target_width, target_height))

                # Encode the image to base64
                base64_image = await self._encode_image(img)

        elif image_data:
            # Convert the image to base64
            img_byte_arr = io.BytesIO()
            img_byte_arr.write(image_data)
            img = await self.hass.loop.run_in_executor(None, Image.open, img_byte_arr)
            with img:
                await self.hass.loop.run_in_executor(None, img.load)
                img = self._convert_to_rgb(img)
                # calculate new height based on aspect ratio
                width, height = img.size
                aspect_ratio = width / height
                target_height = int(target_width / aspect_ratio)

                if width > target_width or height > target_height:
                    img = img.resize((target_width, target_height))

                base64_image = await self._encode_image(img)
        elif img:
            with img:
                img = self._convert_to_rgb(img)
                # calculate new height based on aspect ratio
                width, height = img.size
                aspect_ratio = width / height
                target_height = int(target_width / aspect_ratio)

                if width > target_width or height > target_height:
                    img = img.resize((target_width, target_height))

                base64_image = await self._encode_image(img)

        if base64_image is None:
            raise ServiceValidationError("No image data provided for resize_image")

        return base64_image

    async def _fetch(
        self, url, target_file=None, max_retries=2, retry_delay=1, entity_name=None
    ):
        """Fetch image from url and return image data"""
        retries = 0
        entity_prefix = f"Camera {entity_name}: " if entity_name else ""
        while retries < max_retries:
            _LOGGER.info(f"Fetching {url} (attempt {retries + 1}/{max_retries})")
            try:
                async with self.session.get(url) as response:
                    if not response.ok:
                        _LOGGER.warning(
                            f"{entity_prefix}Couldn't fetch frame (status code: {response.status})"
                        )
                        retries += 1
                        await asyncio.sleep(retry_delay)
                        continue
                    # Just read file into buffer
                    if target_file is None:
                        data = await response.read()
                        return data
                    else:  # Save response into file in stream fashion to avoid memory leaks
                        _LOGGER.debug(f"writing response into file {target_file}")
                        written = 0
                        chunks = 0
                        async with async_open(target_file, "wb") as output:
                            async for data in response.content.iter_any():
                                await output.write(data)
                                written += len(data)
                                chunks += 1
                        _LOGGER.debug(
                            f"wrote {written} bytes ({chunks} chunks) into {target_file}"
                        )

                        return None
            except Exception as e:
                _LOGGER.error(f"{entity_prefix}Fetch failed: {e}")
                retries += 1
                await asyncio.sleep(retry_delay)
        _LOGGER.warning(
            f"{entity_prefix}Failed to fetch {url} after {max_retries} retries"
        )

    @staticmethod
    def _select_stream_frames(
        image_entities, first_frames, frames_with_scores, max_frames
    ):
        """Select frames to analyze from captured stream frames.

        Prepends the first frame of each camera, then fills with the
        highest-movement frames. max_frames=None keeps every captured frame.
        Returns a list of (frame_name, frame_bytes, score) tuples.
        """
        selected_frames = []
        if max_frames is None:
            remaining = len(image_entities) + len(frames_with_scores)
        else:
            remaining = max(0, max_frames)

        # Prepend first frames in the order of requested entities
        for entity in image_entities:
            if remaining <= 0:
                break
            if entity in first_frames:
                label, data = first_frames[entity]
                selected_frames.append((label, data, None))
                remaining -= 1

        # Fill remaining slots with best scored frames, then restore capture order
        best_rest = frames_with_scores[:remaining]
        best_rest.sort(key=lambda x: (x[4], x[3]))
        for name, data, score, _, _ in best_rest:
            selected_frames.append((name, data, score))
        return selected_frames

    @staticmethod
    def _compute_interval(duration, fps=None):
        """Return the capture interval in seconds.

        When fps is provided the interval is 1/fps; otherwise the legacy
        duration-based cadence is used.
        """
        if fps:
            try:
                fps_value = float(fps)
            except (TypeError, ValueError):
                fps_value = 0
            if fps_value > 0:
                return 1.0 / fps_value
        if duration is None or duration < 3:
            return 1
        elif duration < 10:
            return 2
        elif duration < 30:
            return 3
        else:
            return 5

    @staticmethod
    def _validate_polylines(polylines):
        """Validate and normalize polylines.

        Each polyline is a list of at least two [x, y] points with x and y
        normalized to the [0, 1] range. Returns a list of lists of (x, y)
        float tuples, or None when no polylines are supplied.
        """
        if polylines is None:
            return None
        if not isinstance(polylines, (list, tuple)):
            raise ServiceValidationError(
                "polylines must be a list of polylines, each a list of [x, y] points"
            )
        normalized = []
        for polyline in polylines:
            if not isinstance(polyline, (list, tuple)) or len(polyline) < 2:
                raise ServiceValidationError(
                    "each polyline must be a list of at least two [x, y] points"
                )
            points = []
            for point in polyline:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ServiceValidationError(
                        "each polyline point must be an [x, y] pair"
                    )
                x, y = point
                if isinstance(x, bool) or isinstance(y, bool):
                    raise ServiceValidationError("polyline coordinates must be numbers")
                try:
                    x = float(x)
                    y = float(y)
                except (TypeError, ValueError):
                    raise ServiceValidationError("polyline coordinates must be numbers")
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ServiceValidationError(
                        "polyline coordinates must be normalized to the [0, 1] range"
                    )
                points.append((x, y))
            normalized.append(points)
        return normalized if normalized else None

    def _resolve_storage_path(self, storage_path):
        """Resolve and confine a caller-supplied storage directory.

        Relative paths are placed under /media/<domain>/. Absolute paths must
        resolve inside the Home Assistant media or config directory. Path
        traversal outside those roots is rejected.
        """
        if not storage_path or not str(storage_path).strip():
            raise ServiceValidationError("storage_path must not be empty")
        raw = str(storage_path).strip()

        media_root = os.path.realpath(f"/media")
        allowed_roots = [media_root]
        config_dir = getattr(self.hass.config, "config_dir", None)
        if isinstance(config_dir, str) and config_dir:
            allowed_roots.append(os.path.realpath(config_dir))

        if os.path.isabs(raw):
            candidate = os.path.realpath(raw)
        else:
            candidate = os.path.realpath(os.path.join(media_root, DOMAIN, raw))

        if not any(
            candidate == root or candidate.startswith(root + os.sep)
            for root in allowed_roots
        ):
            raise ServiceValidationError(
                "storage_path resolves outside the allowed media/config directory"
            )
        return candidate

    def _resolve_output_file(self, path):
        """Resolve and confine a caller-supplied output file path.

        Relative paths are placed under /media/<domain>/. Absolute paths must
        resolve inside the Home Assistant media or config directory. Path
        traversal outside those roots is rejected.
        """
        if not path or not str(path).strip():
            raise ServiceValidationError("clip_path must not be empty")
        raw = str(path).strip()

        media_root = os.path.realpath(f"/media")
        allowed_roots = [media_root]
        config_dir = getattr(self.hass.config, "config_dir", None)
        if isinstance(config_dir, str) and config_dir:
            allowed_roots.append(os.path.realpath(config_dir))

        if os.path.isabs(raw):
            candidate = os.path.realpath(raw)
        else:
            candidate = os.path.realpath(os.path.join(media_root, DOMAIN, raw))

        if not any(
            candidate == root or candidate.startswith(root + os.sep)
            for root in allowed_roots
        ):
            raise ServiceValidationError(
                "clip_path resolves outside the allowed media/config directory"
            )
        # ffmpeg overwrites (-y); only video container files may be written
        if os.path.splitext(candidate)[1].lower() not in ALLOWED_CLIP_EXTENSIONS:
            raise ServiceValidationError(
                "clip_path must end with " + ", ".join(ALLOWED_CLIP_EXTENSIONS)
            )
        return candidate

    @staticmethod
    def _suffix_path(path, camera_entity):
        """Insert a camera name before the extension for multi-camera clips."""
        stem, ext = os.path.splitext(path)
        safe = camera_entity.replace("camera.", "").replace("/", "_")
        return f"{stem}-{safe}{ext}"

    @staticmethod
    def _clip_output_args(
        duration, record_fps, out_path, scale_width=1920, record_codec=None
    ):
        """ffmpeg output options that transcode a stream to a phone-friendly mp4.

        Downscales oversized streams (never upscales), lets libx264 pick the
        correct H.264 level for the resolution, and uses a short keyframe
        interval so playback decodes smoothly. Frame timing is preserved
        (native) unless a record_fps is given to force a constant rate.
        With record_codec "copy" the original video stream is stored unchanged.
        """
        if record_codec == RECORD_CODEC_COPY:
            return [
                "-t",
                str(duration),
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                "-avoid_negative_ts",
                "make_zero",
                "-y",
                out_path,
            ]
        # Values end up in an ffmpeg filter graph: accept plain numbers only
        record_fps = coerce_number(record_fps, "record_fps", 1, 60)
        scale_width = coerce_number(
            scale_width, "record_scale", 0, 7680, integer=True
        )
        filters = []
        if scale_width and scale_width > 0:
            # Cap width, keep aspect (even height); min() avoids upscaling
            filters.append(f"scale='min({scale_width},iw)':-2")
        if record_fps:
            filters.append(f"fps={record_fps:g}")
        gop = int((record_fps or 15) * 2)

        args = [
            "-t",
            str(duration),
            "-an",
            "-sn",
            "-dn",
        ]
        if filters:
            args += ["-vf", ",".join(filters)]
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-g",
            str(gop),
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            out_path,
        ]
        return args

    @classmethod
    def _build_clip_ffmpeg_cmd(
        cls,
        stream_url,
        duration,
        record_fps,
        out_path,
        scale_width=1920,
        record_codec=None,
    ):
        """Build the ffmpeg command to transcode a stream to a phone-friendly mp4."""
        return [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            *stream_input_args(stream_url),
            "-i",
            stream_url,
            *cls._clip_output_args(
                duration, record_fps, out_path, scale_width, record_codec
            ),
        ]

    async def record_clip(
        self, image_entities, duration, resolved_path, record_fps=None, scale_width=1920
    ):
        """Record each camera's live stream to a fluent H.264 mp4.

        Runs independently of the snapshot analysis. A camera without a stream
        source or a failed transcode is skipped (logged), not fatal. Returns the
        list of written clip paths.
        """
        entities = list(image_entities or [])
        written = []
        for camera_entity in entities:
            out_path = (
                resolved_path
                if len(entities) == 1
                else self._suffix_path(resolved_path, camera_entity)
            )
            try:
                stream_url = await _async_get_stream_source(self.hass, camera_entity)
            except Exception as err:
                _LOGGER.warning(
                    f"Could not get stream source for {camera_entity}: {redact(err)}"
                )
                continue
            if not stream_url:
                _LOGGER.warning(
                    f"Camera {camera_entity} has no stream source; skipping clip"
                )
                continue

            await self.hass.loop.run_in_executor(
                None,
                partial(os.makedirs, os.path.dirname(out_path), exist_ok=True),
            )
            cmd = self._build_clip_ffmpeg_cmd(
                stream_url,
                duration,
                record_fps,
                out_path,
                scale_width,
                self.clip_codec,
            )
            _LOGGER.debug(f"Recording clip: {redacted_command(cmd, stream_url)}")
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=float(duration) + 30
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    _LOGGER.error(f"Clip recording for {camera_entity} timed out")
                    continue
                if proc.returncode == 0 and os.path.exists(out_path):
                    await self._finalize_copied_clip(out_path)
                    written.append(out_path)
                else:
                    detail = (
                        stderr[-STDERR_TAIL_BYTES:].decode(errors="ignore")
                        if stderr
                        else "unknown error"
                    )
                    _LOGGER.error(
                        f"Clip recording failed for {camera_entity}: "
                        f"{redact(detail, stream_url)}"
                    )
            except asyncio.CancelledError:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                raise
            except Exception as err:
                _LOGGER.error(
                    f"Clip recording error for {camera_entity}: "
                    f"{redact(err, stream_url)}"
                )

        self.clip_paths = written
        return written

    async def _finalize_copied_clip(self, clip_path):
        """Make a stream-copied H.265 clip playable on Apple devices (hev1 -> hvc1).

        Only the container is rewritten, into a temporary file next to the clip that
        atomically replaces it. On any failure the original copy is kept (the face
        service decodes hev1 as well) and a warning is logged.
        """
        if self.clip_codec != RECORD_CODEC_COPY:
            return
        loop = self.hass.loop
        try:
            entry = await loop.run_in_executor(None, mp4_video_sample_entry, clip_path)
        except OSError as err:
            _LOGGER.warning(f"Could not inspect clip {clip_path}: {err}")
            return
        if entry != "hev1":
            return
        root, ext = os.path.splitext(clip_path)
        tmp_path = f"{root}.hvc1-{uuid.uuid4().hex[:12]}{ext}"
        replaced = False
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *build_hvc1_remux_cmd(clip_path, tmp_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=REMUX_TIMEOUT
                )
            except asyncio.TimeoutError:
                _LOGGER.warning(f"Retagging clip {clip_path} as hvc1 timed out")
                return
            if proc.returncode != 0:
                detail = (
                    stderr[-STDERR_TAIL_BYTES:].decode(errors="ignore")
                    if stderr
                    else "unknown error"
                )
                _LOGGER.warning(
                    f"Retagging clip {clip_path} as hvc1 failed: "
                    f"{redact(detail)[-1000:]}"
                )
                return
            if (
                await loop.run_in_executor(None, mp4_video_sample_entry, tmp_path)
            ) != "hvc1":
                _LOGGER.warning(f"Retagging clip {clip_path} as hvc1 had no effect")
                return
            await loop.run_in_executor(None, os.replace, tmp_path, clip_path)
            replaced = True
        except Exception as err:
            _LOGGER.warning(
                f"Retagging clip {clip_path} as hvc1 failed: {redact(err)}"
            )
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            if not replaced:
                try:
                    await loop.run_in_executor(None, _remove_quietly, tmp_path)
                except asyncio.CancelledError:
                    # Cancelled while cleaning up: still never leave the temp file
                    _remove_quietly(tmp_path)
                    raise

    @staticmethod
    def _frame_label(image_entity, camera_number, frame_index, include_filename):
        """Frame label matching the snapshot capture naming."""
        prefix = (
            image_entity.replace("camera.", "")
            if include_filename
            else f"camera{camera_number}"
        )
        return f"{prefix}-frame-{frame_index}"

    def _analyze_stream_frame(self, jpeg_data, previous_gray):
        """Decode one piped frame; return (gray, score) or None to skip it.

        Runs in the executor. The pixel count is checked from the header before the
        image is decoded.
        """
        try:
            with Image.open(io.BytesIO(jpeg_data)) as img:
                width, height = img.size
                if width * height > MAX_FRAME_PIXELS:
                    _LOGGER.warning(
                        f"Skipping oversized stream frame ({width}x{height})"
                    )
                    return None
                img.load()
                gray = np.array(img.convert("L"))
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as err:
            _LOGGER.warning(f"Skipping undecodable stream frame: {type(err).__name__}")
            return None
        if previous_gray is None:
            return gray, None
        if previous_gray.shape != gray.shape:
            # Resolution changed mid-stream: treat as maximal change
            return gray, 0.0
        return gray, self._similarity_score(previous_gray, gray)

    @staticmethod
    async def _drain_stderr(stream, tail):
        """Consume stderr so ffmpeg never blocks on it, keeping only the tail."""
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            tail += chunk
            if len(tail) > STDERR_TAIL_BYTES:
                del tail[:-STDERR_TAIL_BYTES]

    @staticmethod
    async def _discard_stream(stream):
        """Read and drop a pipe until EOF."""
        if stream is None:
            return
        while await stream.read(STREAM_READ_CHUNK):
            pass

    async def _stop_process(self, proc):
        """Stop ffmpeg: terminate so outputs are finalized, then kill.

        stdout keeps being drained meanwhile so ffmpeg cannot block on a full pipe.
        Returns True when the process had to be killed.
        """
        if proc.returncode is not None:
            return False
        drain = asyncio.ensure_future(self._discard_stream(proc.stdout))
        killed = False
        try:
            try:
                proc.terminate()
            except ProcessLookupError:
                return False
            try:
                await asyncio.wait_for(proc.wait(), timeout=PROCESS_STOP_TIMEOUT)
            except asyncio.TimeoutError:
                killed = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
        finally:
            drain.cancel()
            try:
                await drain
            except (asyncio.CancelledError, Exception):
                pass
        return killed

    async def _capture_stream_camera(
        self,
        image_entity,
        camera_number,
        duration,
        frame_rate,
        frame_cap,
        target_width,
        include_filename,
        clip_plan=None,
    ):
        """Decode analysis frames from a camera's video stream with one ffmpeg process.

        When clip_plan (path, record_fps, scale_width) is given, the same process also
        writes the clip, so the stream is opened only once. Returns (first_frame,
        frames) in the snapshot capture format, or None when the stream is unusable so
        the caller can fall back to snapshots.
        """
        try:
            stream_url = await _async_get_stream_source(self.hass, image_entity)
        except Exception as err:
            _LOGGER.warning(
                f"Could not get stream source for {image_entity}: {redact(err)}"
            )
            return None
        if not stream_url:
            _LOGGER.warning(f"Camera {image_entity} has no stream source")
            return None
        scheme = stream_scheme(stream_url)
        if scheme not in ALLOWED_STREAM_SCHEMES:
            _LOGGER.warning(
                f"Camera {image_entity} uses unsupported stream scheme '{scheme}'"
            )
            return None

        clip_path = None
        clip_args = None
        if clip_plan:
            clip_path, record_fps, scale_width = clip_plan
            await self.hass.loop.run_in_executor(
                None,
                partial(os.makedirs, os.path.dirname(clip_path), exist_ok=True),
            )
            clip_args = self._clip_output_args(
                duration, record_fps, clip_path, scale_width, self.clip_codec
            )

        cmd = build_stream_capture_cmd(
            stream_url,
            duration,
            frame_rate,
            target_width,
            frame_cap,
            jpeg_q=self.ffmpeg_jpeg_q,
            clip_output_args=clip_args,
        )
        _LOGGER.debug(
            f"Capturing stream frames for {image_entity}: "
            f"{redacted_command(cmd, stream_url)}"
        )

        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + float(duration) + STREAM_GRACE
        startup_deadline = started + STREAM_STARTUP_TIMEOUT
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as err:
            _LOGGER.error(
                f"Could not start ffmpeg for {image_entity}: {redact(err, stream_url)}"
            )
            return None

        stderr_tail = bytearray()
        stderr_task = asyncio.ensure_future(
            self._drain_stderr(proc.stderr, stderr_tail)
        )
        splitter = JpegStreamSplitter()
        first_frame = None
        frames = {}
        previous_gray = None
        frame_index = 0
        # Every JPEG ffmpeg emitted (incl. capped/undecodable): its time is raw/rate
        raw_count = 0
        timed_out = False
        stop_now = False
        killed = False
        try:
            while True:
                limit = (
                    startup_deadline if first_frame is None else deadline
                ) - loop.time()
                if limit <= 0:
                    timed_out = True
                    break
                try:
                    # Once the frame cap is reached ffmpeg's frame output is done while
                    # the clip may still be recording: only the wall deadline applies.
                    timeout = (
                        min(limit, STREAM_STALL_TIMEOUT)
                        if frame_index < frame_cap
                        else limit
                    )
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(STREAM_READ_CHUNK), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    break
                if not chunk:
                    break
                malformed_before = splitter.malformed
                for jpeg_data in splitter.feed(chunk):
                    raw_index = raw_count
                    raw_count += 1
                    # Keep draining past the cap; ffmpeg's -frames:v bounds output too
                    if frame_index >= frame_cap:
                        continue
                    analyzed = await self.hass.loop.run_in_executor(
                        None, self._analyze_stream_frame, jpeg_data, previous_gray
                    )
                    if analyzed is None:
                        continue
                    gray, score = analyzed
                    label = self._frame_label(
                        image_entity, camera_number, frame_index, include_filename
                    )
                    self._face_stream_frames[label] = (image_entity, raw_index)
                    if first_frame is None:
                        first_frame = (label, jpeg_data)
                    else:
                        frames[label] = {
                            "frame_data": jpeg_data,
                            "ssim_score": score,
                            "camera_number": camera_number,
                            "frame_index": frame_index,
                        }
                    previous_gray = gray
                    frame_index += 1
                # Oversized frames dropped by the splitter follow this chunk's frames
                raw_count += splitter.malformed - malformed_before
                if splitter.malformed >= MAX_MALFORMED_FRAMES:
                    _LOGGER.warning(
                        f"Stream for {image_entity} produced malformed frames; stopping"
                    )
                    stop_now = True
                    break
            if not timed_out and not stop_now:
                # stdout closed: give the clip output time to finalize
                try:
                    await asyncio.wait_for(
                        proc.wait(), timeout=max(1.0, deadline - loop.time())
                    )
                except asyncio.TimeoutError:
                    timed_out = True
            killed = await self._stop_process(proc)
        except asyncio.CancelledError:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
            killed = True
            raise
        finally:
            if proc.returncode is None:
                # Unexpected error: never leave ffmpeg running unattended
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                killed = True
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass
            if clip_path and killed:
                # A killed ffmpeg leaves an unplayable mp4 without its index
                try:
                    await self.hass.loop.run_in_executor(None, os.remove, clip_path)
                except OSError:
                    pass

        returncode = proc.returncode
        if timed_out or returncode not in (0, None):
            detail = redact(stderr_tail.decode(errors="ignore"), stream_url)
            _LOGGER.warning(
                f"Stream capture for {image_entity} ended early "
                f"(returncode={returncode}, timed_out={timed_out}): {detail[-1000:]}"
            )
        if clip_path and not killed:
            try:
                size = await self.hass.loop.run_in_executor(
                    None, os.path.getsize, clip_path
                )
            except OSError:
                size = 0
            if size > 0:
                await self._finalize_copied_clip(clip_path)
                self.clip_paths.append(clip_path)

        if first_frame is None:
            return None
        return first_frame, frames

    async def _draw_polylines_on_image(self, image_data, target_width, polylines):
        """Resize an image to target_width and draw red polylines on it.

        Returns a tuple of (base64_jpeg, (width, height), resolved_polylines)
        where resolved_polylines are the pixel coordinates actually drawn.
        """
        base64_image, size, resolved, _ = await self._annotate_image(
            image_data, target_width, polylines, None
        )
        return base64_image, size, resolved

    async def _annotate_image(
        self, image_data, target_width, polylines=None, face_labels=None
    ):
        """Resize an image to target_width, draw red polylines, then name labels.

        face_labels is a list of (name, normalized face box); each is drawn as a
        green head box with the name above it. Returns (base64_jpeg, (width,
        height), resolved_polylines, resolved_faces) in pixel coordinates.
        """

        def _work():
            img = Image.open(io.BytesIO(image_data))
            img.load()
            img = self._convert_to_rgb(img)
            width, height = img.size
            aspect_ratio = width / height
            target_height = int(target_width / aspect_ratio)
            # Width alone decides: a float-rounded target_height must not trigger a
            # 1 px resample of frames ffmpeg already scaled to target_width.
            if width > target_width:
                img = img.resize((target_width, target_height))
            w, h = img.size
            resolved = []
            faces = []
            if polylines or face_labels:
                draw = ImageDraw.Draw(img)
                line_width = max(2, round(min(w, h) * 0.005))
                for polyline in polylines or []:
                    px_points = [
                        (round(x * w), round(y * h)) for (x, y) in polyline
                    ]
                    resolved.append(px_points)
                    draw.line(px_points, fill=(255, 0, 0), width=line_width)
                font = (
                    _label_font(max(12, round(min(w, h) * 0.03)))
                    if face_labels
                    else None
                )
                for name, box in face_labels or []:
                    rect = face_client.head_box(box, w, h)
                    _draw_name_label(draw, rect, name, line_width, w, h, font)
                    faces.append({"name": name, "box": list(rect)})
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", **self.jpeg_options)
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return base64_image, (w, h), resolved, faces

        return await self.hass.loop.run_in_executor(None, _work)

    async def _write_snapshot(self, directory, filename, image_data):
        """Write a snapshot (bytes or base64) into directory, returning the path."""
        await self.hass.loop.run_in_executor(
            None, partial(os.makedirs, directory, exist_ok=True)
        )
        path = os.path.join(directory, filename)

        def _write():
            with open(path, "wb") as handle:
                if isinstance(image_data, bytes):
                    handle.write(image_data)
                else:
                    handle.write(base64.b64decode(image_data))

        try:
            await self.hass.loop.run_in_executor(None, _write)
        except OSError as err:
            raise ServiceValidationError(
                f"Failed to write snapshot to {path}: {err}"
            )
        return path

    async def _storage_copy(
        self, model_b64, frame_data, target_width, polylines, face_labels
    ):
        """Image for storage_path: the model frame, but never with name labels."""
        if not face_labels:
            return model_b64
        if polylines:
            clean, _, _, _ = await self._annotate_image(
                frame_data, target_width, polylines, None
            )
            return clean
        return await self.resize_image(target_width=target_width, image_data=frame_data)

    def _start_face_task(self, clip_data, camera):
        """Upload an already-read clip in the background; returns the task."""
        return asyncio.ensure_future(
            face_client.async_identify(
                self.hass, self.face_settings, camera, clip_data=clip_data
            )
        )

    async def _prepare_face_task(self, clip_path, camera):
        """Read the clip now (before it may be deleted) and start the upload.

        Returns a future that resolves to a FaceResult and never raises except on
        cancellation.
        """
        try:
            clip_data = await face_client.async_load_clip(self.hass, clip_path)
        except face_client.FaceServiceError as err:
            future = asyncio.get_running_loop().create_future()
            future.set_result(face_client.error_result(camera, err.reason))
            return future
        return self._start_face_task(clip_data, camera)

    async def _identify_camera_clips(self, entities, clip_paths, preloaded=None):
        """One identification call per camera clip; returns {entity: FaceResult}.

        preloaded maps entity -> clip bytes, or a failure reason when reading failed.
        """
        written = set(self.clip_paths or [])
        preloaded = preloaded or {}

        async def _one(entity):
            camera = face_client.camera_slug_for_entity(entity)
            if entity in preloaded:
                item = preloaded[entity]
                if isinstance(item, str):
                    return entity, face_client.error_result(camera, item)
                return entity, await face_client.async_identify(
                    self.hass, self.face_settings, camera, clip_data=item
                )
            path = clip_paths.get(entity)
            if not path or path not in written:
                return entity, face_client.error_result(
                    camera, face_client.REASON_NO_CLIP
                )
            return entity, await face_client.async_identify(
                self.hass, self.face_settings, camera, clip_path=path
            )

        results = dict(await asyncio.gather(*(_one(entity) for entity in entities)))
        self.face_results.extend(results[entity] for entity in entities)
        return results

    async def _identify_stream_clips(
        self, image_entities, clip_plans, rate, selected_frames
    ):
        """Stream mode: identify persons per camera clip and map them to frames.

        A frame's clip time is its raw ffmpeg output index divided by the stream
        frame rate. Returns {selected frame index: [(name, box), ...]}.
        """
        self._face_done = True
        if not self.face_active:
            return {}
        entities = list(dict.fromkeys(image_entities))
        clip_paths = {
            entity: plan[0] for entity, plan in (clip_plans or {}).items() if plan
        }
        results = await self._identify_camera_clips(entities, clip_paths)
        labels = {}
        if not rate:
            return labels
        for idx, (frame_name, _, _) in enumerate(selected_frames):
            info = self._face_stream_frames.get(frame_name)
            if info is None:
                continue
            entity, raw_index = info
            result = results.get(entity)
            if result is None or not result.ok:
                continue
            frame_labels = face_client.labels_for_time(
                result.persons, face_client.fps_frame_time(raw_index, rate)
            )
            if frame_labels:
                labels[idx] = frame_labels
        return labels

    async def load_recorded_clips(self):
        """Snapshot mode: read the recorded clips while the camera lock is held.

        A follow-up capture may overwrite the same clip_path as soon as the lock is
        released, so the bytes are kept in memory until identify_recorded_clips.
        """
        if self._face_done or not self.face_active:
            return
        written = set(self.clip_paths or [])
        loaded = {}
        for entity, path in self._face_clip_plan.items():
            if not path or path not in written:
                continue
            try:
                loaded[entity] = await face_client.async_load_clip(self.hass, path)
            except face_client.FaceServiceError as err:
                loaded[entity] = err.reason
        self._face_clip_data = loaded

    async def identify_recorded_clips(self):
        """Snapshot mode: identify persons in the recorded clips (facts only).

        Called by stream_analyzer_pro after the capture locks are released. Does
        nothing when identification already ran during stream capture.
        """
        if self._face_done or not self.face_active:
            return
        self._face_done = True
        entities = list(self._face_clip_plan)
        preloaded, self._face_clip_data = self._face_clip_data, None
        if not entities:
            self.face_results.append(
                face_client.error_result("camera", face_client.REASON_NO_CLIP)
            )
            return
        await self._identify_camera_clips(entities, self._face_clip_plan, preloaded)

    async def record(
        self,
        image_entities,
        duration,
        max_frames,
        target_width,
        include_filename,
        expose_images,
        fps=None,
        polylines=None,
        storage_path=None,
        debug_polylines=False,
        frame_source=None,
        clip_plans=None,
    ):
        """Wrapper for client.add_frame with integrated recorder

        Args:
            image_entities (list[string]): List of camera entities to record
            duration (float): Duration in seconds to record
            target_width (int): Target width for the images in pixels
            fps (float): Optional capture rate; overrides the duration cadence
            polylines (list): Optional normalized polylines drawn in red on every
                analyzed frame; the exposed key frame remains clean
            storage_path (str): Optional directory to persist analyzed snapshots
            debug_polylines (bool): When True, collect polyline debug info and
                persist annotated frames for inspection
            frame_source (str): "stream" decodes frames from the camera stream
                (falling back to snapshots per camera); default uses snapshots
            clip_plans (dict): Stream mode only; camera entity to
                (clip_path, record_fps, scale_width) written by the same ffmpeg
        """

        polylines = self._validate_polylines(polylines)
        resolved_storage = (
            self._resolve_storage_path(storage_path) if storage_path else None
        )
        interval = self._compute_interval(duration, fps)
        camera_frames = {}
        first_frames = {}
        # Track successful image entities (cameras that successfully captured frames)
        successful_image_entities = set()

        # Record on a separate thread for each camera
        async def record_camera(image_entity, camera_number):
            start = time.time()
            frame_counter = 0
            frames = {}
            previous_frame = None
            iteration_time = 0

            base_url = get_url(self.hass)

            while time.time() - start < duration + iteration_time:
                fetch_start_time = time.time()
                entity_state = self.hass.states.get(image_entity)

                # Check if entity exists
                if entity_state is None:
                    _LOGGER.error(f"Camera {image_entity} does not exist")
                    await asyncio.sleep(interval)
                    continue

                entity_picture = entity_state.attributes.get("entity_picture")

                # Skip if camera is offline or entity_picture unavailable
                if not entity_picture:
                    _LOGGER.warning(
                        f"Camera {image_entity} is offline or does not have entity_picture attribute"
                    )
                    await asyncio.sleep(interval)
                    continue

                frame_url = base_url + entity_picture
                frame_data = await self._fetch(frame_url, entity_name=image_entity)

                # Skip frame if fetch failed
                if not frame_data:
                    await asyncio.sleep(interval)
                    continue

                fetch_duration = time.time() - fetch_start_time
                _LOGGER.info(f"Fetched {image_entity} in {fetch_duration:.2f} seconds")

                preprocessing_start_time = time.time()

                with await self.hass.loop.run_in_executor(
                    None, Image.open, io.BytesIO(frame_data)
                ) as img:
                    current_frame_gray = np.array(img.convert("L"))

                    if previous_frame is not None:
                        score = self._similarity_score(
                            previous_frame, current_frame_gray
                        )
                        # Encode the image back to bytes
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", **self.jpeg_options)
                        frame_data = buffer.getvalue()

                        # Use either entity name or assign number to each camera
                        if include_filename:
                            parts = [
                                image_entity.replace("camera.", ""),
                                "frame",
                                str(frame_counter),
                            ]
                        else:
                            parts = [
                                f"camera{camera_number}",
                                "frame",
                                str(frame_counter),
                            ]
                        frame_label = "-".join(parts)
                        frames.update(
                            {
                                frame_label: {
                                    "frame_data": frame_data,
                                    "ssim_score": score,
                                    "camera_number": camera_number,
                                    "frame_index": frame_counter,
                                }
                            }
                        )

                        frame_counter += 1
                        previous_frame = current_frame_gray
                    else:
                        # Current frame is first frame
                        previous_frame = current_frame_gray
                        # Normalize to JPEG
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", **self.jpeg_options)
                        first_bytes = buffer.getvalue()
                        if include_filename:
                            parts = [
                                image_entity.replace("camera.", ""),
                                "frame",
                                str(frame_counter),
                            ]
                        else:
                            parts = [
                                f"camera{camera_number}",
                                "frame",
                                str(frame_counter),
                            ]
                        frame_label = "-".join(parts)
                        first_frames[image_entity] = (frame_label, first_bytes)
                        # Mark this camera as successful
                        successful_image_entities.add(image_entity)
                        frame_counter += 1

                preprocessing_duration = time.time() - preprocessing_start_time
                _LOGGER.info(
                    f"Preprocessing took: {preprocessing_duration:.2f} seconds"
                )

                adjusted_interval = max(
                    0, interval - fetch_duration - preprocessing_duration
                )

                if iteration_time == 0:
                    iteration_time = time.time() - start
                    _LOGGER.info(
                        f"First iteration took: {iteration_time:.2f} seconds, interval adjusted to: {adjusted_interval}"
                    )

                await asyncio.sleep(adjusted_interval)

            camera_frames.update({image_entity: frames})

        camera_names = ", ".join(
            entity.replace("camera.", "") for entity in image_entities
        )
        _LOGGER.info(f"Recording {camera_names} for {duration} seconds")

        frame_rate = frame_cap = None
        if frame_source == FRAME_SOURCE_STREAM:
            frame_rate, rate = stream_frame_rate(fps, interval)
            frame_cap = stream_frame_cap(duration, rate)

        async def capture_camera(image_entity, camera_number):
            if frame_source == FRAME_SOURCE_STREAM:
                captured = await self._capture_stream_camera(
                    image_entity=image_entity,
                    camera_number=camera_number,
                    duration=duration,
                    frame_rate=frame_rate,
                    frame_cap=frame_cap,
                    target_width=target_width,
                    include_filename=include_filename,
                    clip_plan=(clip_plans or {}).get(image_entity),
                )
                if captured is not None:
                    first_frames[image_entity], camera_frames[image_entity] = captured
                    successful_image_entities.add(image_entity)
                    return
                _LOGGER.warning(
                    f"Falling back to snapshot capture for {image_entity}"
                )
            await record_camera(image_entity, camera_number)

        # start threads for each camera
        await asyncio.gather(
            *(
                capture_camera(image_entity, image_entities.index(image_entity))
                for image_entity in image_entities
            )
        )

        # Check if any cameras successfully captured frames
        if len(successful_image_entities) == 0:
            raise ServiceValidationError(
                "No cameras available - all cameras offline or unavailable"
            )

        # Extract frames and their SSIM scores
        frames_with_scores = []
        for frame in camera_frames:
            for frame_name, frame_data in camera_frames[frame].items():
                frames_with_scores.append(
                    (
                        frame_name,
                        frame_data["frame_data"],
                        frame_data["ssim_score"],
                        frame_data["camera_number"],
                        frame_data["frame_index"],
                    )
                )

        # Sort frames by SSIM score
        frames_with_scores.sort(key=lambda x: x[2])

        # Frame selection (respects max_frames; None means unbounded)
        selected_frames = self._select_stream_frames(
            image_entities, first_frames, frames_with_scores, max_frames
        )

        # Add selected frames to client
        if selected_frames:
            # Choose keyframe among the selected frames using the last as reference
            reference_bytes = selected_frames[-1][1]
            candidate_bytes = [data for _, data, _ in selected_frames]
            key_idx = await self._select_keyframe_index(
                reference_bytes, candidate_bytes
            )

            # Stream clips: identify persons once the clips are complete
            frame_labels = {}
            if self.identify_persons and frame_source == FRAME_SOURCE_STREAM:
                frame_labels = await self._identify_stream_clips(
                    image_entities, clip_plans, rate, selected_frames
                )

            debug_info = [] if debug_polylines else None
            # Add annotated frames to the model and analyzed-snapshot storage.
            resized_base64 = []
            for idx, (frame_name, frame_data, _) in enumerate(selected_frames):
                labels = frame_labels.get(idx)
                if polylines or labels:
                    resized_image, (fw, fh), resolved, faces = (
                        await self._annotate_image(
                            image_data=frame_data,
                            target_width=target_width,
                            polylines=polylines,
                            face_labels=labels,
                        )
                    )
                    self.face_labeled_names.update(face["name"] for face in faces)
                    if debug_info is not None:
                        entry = {
                            "frame": frame_name,
                            "width": fw,
                            "height": fh,
                            "polylines": resolved,
                        }
                        if self.identify_persons:
                            entry["faces"] = faces
                        debug_info.append(entry)
                else:
                    resized_image = await self.resize_image(
                        target_width=target_width, image_data=frame_data
                    )
                resized_base64.append(resized_image)
                self.client.add_frame(base64_image=resized_image, filename=frame_name)

            if resolved_storage or debug_polylines:
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                for idx, (frame_name, frame_data, _) in enumerate(selected_frames):
                    safe_name = frame_name.replace("/", "_").replace("..", "_")
                    filename = f"{safe_name}-{stamp}-{idx}.jpg"
                    if resolved_storage:
                        await self._write_snapshot(
                            directory=resolved_storage,
                            filename=filename,
                            image_data=await self._storage_copy(
                                resized_base64[idx],
                                frame_data,
                                target_width,
                                polylines,
                                frame_labels.get(idx),
                            ),
                        )
                    # In debug mode also persist annotated frames for inspection,
                    # even when expose_images is off.
                    if debug_polylines and (polylines or frame_labels.get(idx)):
                        await self._write_snapshot(
                            directory=self.snapshots_path,
                            filename=f"debug-{filename}",
                            image_data=resized_base64[idx],
                        )

            if debug_info is not None:
                self.debug_info = debug_info

            if expose_images:
                key_name = selected_frames[key_idx][0]
                key_b64 = resized_base64[key_idx]
                if polylines or frame_labels.get(key_idx):
                    key_b64 = await self.resize_image(
                        target_width=target_width,
                        image_data=selected_frames[key_idx][1],
                    )
                await self._expose_image(
                    frame_name=key_name.split("-")[0],
                    image_data=key_b64,
                    uid=str(uuid.uuid4())[:8],
                )

    async def add_images(
        self, image_entities, image_paths, target_width, include_filename, expose_images
    ):
        """Wrapper for client.add_frame for images"""
        base_url = get_url(self.hass)
        # Track successful image entities (cameras that successfully provided frames)
        successful_image_entities = 0

        if image_entities:
            for image_entity in image_entities:
                try:
                    entity_state = self.hass.states.get(image_entity)

                    # Check if entity exists
                    if entity_state is None:
                        _LOGGER.error(f"Camera {image_entity} does not exist")
                        continue

                    entity_picture = entity_state.attributes.get("entity_picture")

                    # Skip if camera is offline or entity_picture unavailable
                    if not entity_picture:
                        _LOGGER.warning(
                            f"Camera {image_entity} is offline or does not have entity_picture attribute"
                        )
                        continue

                    image_url = base_url + entity_picture
                    image_data = await self._fetch(image_url, entity_name=image_entity)

                    # Skip frame if fetch failed
                    if not image_data:
                        _LOGGER.warning(f"Camera {image_entity}: Failed to fetch image")
                        continue

                    # If entity snapshot requested, use entity name as 'filename'
                    resized_image = await self.resize_image(
                        target_width=target_width, image_data=image_data
                    )
                    self.client.add_frame(
                        base64_image=resized_image,
                        filename=(
                            entity_state.attributes.get("friendly_name")
                            if include_filename
                            else ""
                        ),
                    )

                    if expose_images:
                        await self._expose_image(
                            frame_name="0",
                            image_data=resized_image,
                            uid=str(uuid.uuid4())[:8],
                        )

                    successful_image_entities += 1

                except AttributeError as e:
                    _LOGGER.error(
                        f"Camera {image_entity}: AttributeError accessing entity attributes: {e}"
                    )
                    raise ServiceValidationError(
                        f"Error accessing camera entity {image_entity}: {e}"
                    )

            # Check if any cameras were successful
            if successful_image_entities == 0:
                raise ServiceValidationError(
                    "No cameras available - all cameras offline or unavailable"
                )
        if image_paths:
            for image_path in image_paths:
                try:
                    image_path = image_path.strip()

                    if not os.path.exists(image_path):
                        raise ServiceValidationError(
                            f"File {image_path} does not exist"
                        )

                    filename = ""

                    if include_filename:
                        filename = image_path.split("/")[-1].split(".")[-2]

                    image_data = await self.resize_image(
                        target_width=target_width, image_path=image_path
                    )

                    self.client.add_frame(base64_image=image_data, filename=filename)

                    if expose_images:
                        await self._expose_image(
                            frame_name="0",
                            image_data=image_data,
                            uid=str(uuid.uuid4())[:8],
                        )
                except Exception as e:
                    raise ServiceValidationError(f"Error: {e}")
        return self.client

    async def add_video(
        self,
        video_path,
        base_url,
        max_frames=10,
        target_width=640,
        include_filename=False,
        expose_images=False,
        fps=None,
        polylines=None,
        resolved_storage=None,
        debug_polylines=False,
    ):
        face_task = None
        try:
            current_event_id = str(uuid.uuid4())
            video_path = video_path.strip()
            face_camera = face_client.camera_slug_for_video(video_path)
            sample_fps = None

            # Resolve media source (media-source://...)
            if is_media_source_id(video_path):
                _LOGGER.debug(f"Resolving media source id: {video_path}")
                video_path = async_process_play_media_url(self.hass, video_path)
                _LOGGER.debug(f"media url = {video_path}")

            # Sign local API URLs unless already signed
            if video_path.startswith("/api"):
                if "authSig" not in video_path:
                    # Add authorization signature with 5 minute expiration
                    _LOGGER.debug(f"Signing {video_path}")
                    video_path = async_sign_path(
                        self.hass, video_path, timedelta(minutes=5)
                    )
                    _LOGGER.debug(f"signed_path = {video_path}")
                else:
                    # Already signed, just use it
                    _LOGGER.debug(f"Already signed {video_path}")

                video_path = base_url + video_path

            # Sample at a fixed fps when requested, otherwise keep only I-frames
            frame_filter = "select=eq(pict_type\\,I)"
            if fps:
                try:
                    fps_value = float(fps)
                except (TypeError, ValueError):
                    fps_value = 0
                if fps_value > 0:
                    frame_filter = f"fps={fps_value}"
                    sample_fps = fps_value
            ffmpeg_tail = [
                "-vf",  # video filter
                frame_filter,
                "-vsync",
                "0",  # disable v-sync to avoid frame duplication
                "-q:v",  # quality level for JPEG (lower is better)
                str(self.ffmpeg_jpeg_q),  # 5 medium quality; pro services use 2
                "-f",
                "image2pipe",  # output to pipe
                "-vcodec",
                "mjpeg",  # encode as mjpeg
                "-",
            ]
            ffmpeg_stderr = None
            temp_file_path = None

            # If file is served over http(s)
            if video_path.startswith("http://") or video_path.startswith("https://"):
                # Download to a seekable temp file so ffmpeg can parse MP4 (moov at end)
                suffix = os.path.splitext(urlparse(video_path).path)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    temp_file_path = tmp.name
                await self._fetch(video_path, target_file=temp_file_path)
                if (
                    not os.path.exists(temp_file_path)
                    or os.path.getsize(temp_file_path) == 0
                ):
                    raise ServiceValidationError(
                        f"Failed to fetch video from {video_path}"
                    )
                if self.face_active:
                    face_task = await self._prepare_face_task(
                        temp_file_path, face_camera
                    )

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-hide_banner",  # cleaner logs
                    "-loglevel",
                    "error",
                    # input network robustness if ever used with URLs directly
                    "-an",
                    "-sn",
                    "-dn",
                    "-i",
                    temp_file_path,
                    *ffmpeg_tail,
                ]

                output = asyncio.subprocess.PIPE
                error_output = asyncio.subprocess.DEVNULL
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    error_output = asyncio.subprocess.PIPE
                ffmpeg_stderr = None

                ffmpeg_start = time.monotonic_ns()
                ffmpeg_timeout = 300  # seconds

                _LOGGER.debug(
                    f"Running FFMPEG to create keyframes: {' '.join(ffmpeg_cmd)}"
                )
                ffmpeg_process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=output,
                    stderr=error_output,
                )

            else:
                # Local file
                if self.face_active:
                    face_task = await self._prepare_face_task(video_path, face_camera)
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-hwaccel",
                    "auto",
                    "-an",
                    "-sn",
                    "-dn",
                    "-i",
                    video_path,
                    *ffmpeg_tail,
                ]

                output = asyncio.subprocess.PIPE
                error_output = asyncio.subprocess.DEVNULL
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    error_output = None

                ffmpeg_start = time.monotonic_ns()
                ffmpeg_timeout = 300  # seconds

                _LOGGER.debug(
                    f"Running FFMPEG to create keyframes: {' '.join(ffmpeg_cmd)}"
                )
                ffmpeg_process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd, stdout=output, stderr=error_output
                )
                _LOGGER.info(
                    f"Started ffmpeg pid={ffmpeg_process.pid} (stdin={'inherit' if video_path else 'none'}, "
                    f"stdout={'pipe' if ffmpeg_process.stdout else 'none'}, stderr={'inherited' if error_output is None else 'devnull'})"
                )

            previous_frame = None
            frames = []
            frame_counter = 0
            jpeg_buffer = b""
            first_frame = None
            # Raw ffmpeg output index (incl. undecodable frames) per decoded frame
            raw_count = 0
            raw_by_counter = {}

            def find_jpeg_frames(data):
                frames = []
                start = 0
                while True:
                    soi = data.find(b"\xff\xd8", start)
                    eoi = data.find(b"\xff\xd9", soi)
                    if soi == -1 or eoi == -1:
                        break
                    frames.append(data[soi : eoi + 2])
                    start = eoi + 2
                return frames, data[start:]

            try:
                per_read_timeout = 30  # seconds
                # Ensure stdout is readable
                if ffmpeg_process.stdout is None:
                    _LOGGER.error("ffmpeg stdout is not a PIPE; cannot read frames")
                    await ffmpeg_process.wait()
                    raise ServiceValidationError("ffmpeg stdout not available")
                # Read until ffmpeg closes stdout
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            ffmpeg_process.stdout.read(4096), timeout=per_read_timeout
                        )
                    except asyncio.TimeoutError:
                        _LOGGER.warning(
                            "Timeout while waiting for ffmpeg stdout read; terminating read loop"
                        )
                        break

                    if not chunk:
                        _LOGGER.debug("ffmpeg stdout closed or returned no data")
                        break

                    jpeg_buffer += chunk
                    found_frames, jpeg_buffer = find_jpeg_frames(jpeg_buffer)
                    raw_base = raw_count
                    raw_count += len(found_frames)

                    for raw_index, jpeg_data in enumerate(found_frames, raw_base):
                        try:
                            img = Image.open(io.BytesIO(jpeg_data))
                            await self.hass.loop.run_in_executor(None, img.load)
                            if img.mode == "RGBA":
                                img = img.convert("RGB")
                            current_frame_gray = np.array(img.convert("L"))
                            raw_by_counter[frame_counter] = raw_index
                            if previous_frame is not None:
                                score = self._similarity_score(
                                    previous_frame, current_frame_gray
                                )
                                frames.append((jpeg_data, score, frame_counter))
                                _LOGGER.debug(
                                    f"Appended frame {frame_counter} (score={score:.6f}, bytes={len(jpeg_data)})"
                                )
                            else:
                                # First frame, always include
                                first_frame = (jpeg_data, frame_counter)
                                _LOGGER.debug(
                                    f"Captured first frame {frame_counter} (bytes={len(jpeg_data)})"
                                )

                            previous_frame = current_frame_gray
                            frame_counter += 1
                        except UnidentifiedImageError:
                            _LOGGER.error(
                                f"Cannot identify image from ffmpeg pipe at frame {frame_counter}"
                            )
                            continue
                        if max_frames is not None and frame_counter >= max_frames:
                            break
                await ffmpeg_process.wait()

                if (
                    error_output == asyncio.subprocess.PIPE
                    and ffmpeg_process.stderr is not None
                ):
                    try:
                        ffmpeg_stderr = await ffmpeg_process.stderr.read()
                    except Exception:
                        ffmpeg_stderr = None

            except asyncio.TimeoutError:
                _LOGGER.info(
                    f"FFmpeg failed to process video within {ffmpeg_timeout} seconds"
                )
                if ffmpeg_process.returncode is not None:
                    ffmpeg_process.terminate()

            _LOGGER.debug(
                f"FFmpeg process finished with return code {ffmpeg_process.returncode}"
            )

            # Cleanup temp file (if any)
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

            if ffmpeg_process.returncode != 0:
                msg = f"FFmpeg failed with return code {ffmpeg_process.returncode}"
                if ffmpeg_stderr:
                    try:
                        msg += f": {ffmpeg_stderr.decode(errors='ignore')}"
                    except Exception:
                        pass
                raise ServiceValidationError(msg)

            ffmpeg_time = time.monotonic_ns() - ffmpeg_start
            _LOGGER.debug(f"FFmpeg took {ffmpeg_time / 1_000_000:.2f} ms")

            if len(frames) == 0 and first_frame is None:
                raise ServiceValidationError("No frames extracted from video.")

            # Log all frames with scores
            for fdata, fscore, findex in frames:
                _LOGGER.debug(f"Extracted frame {findex} with SSIM score {fscore:.6f}")

            # Sort scored frames by SSIM
            frames.sort(key=lambda x: x[1])

            # Frame selection: prepend first frame, then best-scored.
            # max_frames=None means unbounded: keep every extracted frame.
            selected_frames = []
            if max_frames is None:
                remaining = len(frames) + 1
            else:
                remaining = max(0, max_frames)

            if first_frame is not None and remaining > 0:
                first_data, first_idx = first_frame
                selected_frames.append((first_data, None, first_idx))
                remaining -= 1

            best_rest = frames[:remaining]
            # Keep chronological order for the rest
            best_rest.sort(key=lambda x: x[2])
            selected_frames.extend(best_rest)

            # The key frame is only needed for the exposed image.
            key_idx = None
            if selected_frames and expose_images:
                reference_bytes = selected_frames[0][0]
                candidate_bytes = [fd for (fd, _, _) in selected_frames]
                key_idx = await self._select_keyframe_index(
                    reference_bytes, candidate_bytes
                )

            video_base = os.path.splitext(os.path.basename(video_path))[0]

            # Face identification ran alongside ffmpeg; frame n of the fps output
            # is at n / fps seconds. I-frame mode has no frame times: facts only.
            frame_labels = {}
            if face_task is not None:
                face_result = await face_task
                face_task = None
                self.face_results.append(face_result)
                if face_result.ok and sample_fps:
                    for i, (_, _, counter) in enumerate(selected_frames):
                        raw_index = raw_by_counter.get(counter)
                        if raw_index is None:
                            continue
                        labels = face_client.labels_for_time(
                            face_result.persons,
                            face_client.fps_frame_time(raw_index, sample_fps),
                        )
                        if labels:
                            frame_labels[i] = labels

            # Every frame sent to the model carries the polylines and name labels.
            resized_base64 = []
            for i, (frame_data, _, _) in enumerate(selected_frames):
                idx = i + 1
                labels = frame_labels.get(i)
                if polylines or labels:
                    resized_image, (fw, fh), resolved, faces = (
                        await self._annotate_image(
                            image_data=frame_data,
                            target_width=target_width,
                            polylines=polylines,
                            face_labels=labels,
                        )
                    )
                    self.face_labeled_names.update(face["name"] for face in faces)
                    if debug_polylines and self.debug_info is not None:
                        entry = {
                            "frame": f"{video_base} frame {idx}",
                            "width": fw,
                            "height": fh,
                            "polylines": resolved,
                        }
                        if self.identify_persons:
                            entry["faces"] = faces
                        self.debug_info.append(entry)
                else:
                    resized_image = await self.resize_image(
                        target_width=target_width, image_data=frame_data
                    )
                resized_base64.append(resized_image)
                self.client.add_frame(
                    base64_image=resized_image,
                    filename=(
                        f"{video_base} (frame {idx})"
                        if include_filename
                        else f"Video frame {idx}"
                    ),
                )

            # Persist analyzed snapshots to disk if requested
            if resolved_storage or debug_polylines:
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                safe_base = video_base.replace("/", "_").replace("..", "_")
                for i in range(len(selected_frames)):
                    filename = f"{safe_base}-{stamp}-{i}.jpg"
                    if resolved_storage:
                        await self._write_snapshot(
                            directory=resolved_storage,
                            filename=filename,
                            image_data=await self._storage_copy(
                                resized_base64[i],
                                selected_frames[i][0],
                                target_width,
                                polylines,
                                frame_labels.get(i),
                            ),
                        )
                    if debug_polylines and (polylines or frame_labels.get(i)):
                        await self._write_snapshot(
                            directory=self.snapshots_path,
                            filename=f"debug-{filename}",
                            image_data=resized_base64[i],
                        )

            if expose_images and selected_frames and key_idx is not None:
                frame_idx_label = (selected_frames[key_idx][2] or 0) + 1
                key_b64 = resized_base64[key_idx]
                if polylines or frame_labels.get(key_idx):
                    # Expose a clean copy; the model's copy stays annotated.
                    key_b64 = await self.resize_image(
                        target_width=target_width,
                        image_data=selected_frames[key_idx][0],
                    )
                await self._expose_image(
                    frame_name=str(frame_idx_label),
                    image_data=key_b64,
                    uid=str(uuid.uuid4())[:8],
                )
        except Exception as e:
            raise ServiceValidationError(f"Error processing video {video_path}: {e}")
        finally:
            if face_task is not None and not face_task.done():
                face_task.cancel()

    async def add_videos(
        self,
        video_paths,
        event_ids,
        max_frames,
        target_width,
        include_filename,
        expose_images,
        fps=None,
        polylines=None,
        storage_path=None,
        debug_polylines=False,
    ):
        """Wrapper for client.add_frame for videos"""

        if not video_paths:
            video_paths = []

        base_url = get_url(self.hass)

        if event_ids:
            for event_id in event_ids:
                url = "/api/frigate/notifications/" + event_id + "/clip.mp4"
                # append to video_paths
                video_paths.append(url)

        _LOGGER.debug(f"Processing videos: {video_paths}")

        # Validate/resolve pro options once for all videos
        polylines = self._validate_polylines(polylines)
        resolved_storage = (
            self._resolve_storage_path(storage_path) if storage_path else None
        )
        if debug_polylines:
            self.debug_info = []

        def process_video(video_path):
            return self.add_video(
                video_path=video_path,
                base_url=base_url,
                max_frames=max_frames,
                target_width=target_width,
                include_filename=include_filename,
                expose_images=expose_images,
                fps=fps,
                polylines=polylines,
                resolved_storage=resolved_storage,
                debug_polylines=debug_polylines,
            )

        # Process videos in parallel
        await asyncio.gather(*map(process_video, video_paths))

        return self.client

    async def add_streams(
        self,
        image_entities,
        duration,
        max_frames,
        target_width,
        include_filename,
        expose_images,
        fps=None,
        polylines=None,
        storage_path=None,
        debug_polylines=False,
        clip_path=None,
        record_fps=None,
        record_scale=None,
        frame_source=None,
        record_codec=None,
    ):
        frame_source = normalize_frame_source(frame_source)
        self.clip_codec = normalize_record_codec(record_codec)
        if image_entities:
            # Resolve/confine the clip path before recording so a bad path fails
            # fast without cancelling the snapshot capture.
            resolved_clip = (
                self._resolve_output_file(clip_path) if clip_path else None
            )
            # Values below reach ffmpeg arguments and filters: plain numbers only
            if resolved_clip is not None:
                record_fps = coerce_number(record_fps, "record_fps", 1, 60)
                record_scale = coerce_number(
                    record_scale, "record_scale", 0, 7680, integer=True
                )
                if self.clip_codec == RECORD_CODEC_COPY and (
                    record_fps is not None or record_scale
                ):
                    raise ServiceValidationError(
                        "record_fps and record_scale cannot be used with "
                        "record_codec 'copy' (the original stream is stored "
                        "unchanged); leave them empty"
                    )
            if frame_source == FRAME_SOURCE_STREAM:
                duration = coerce_number(
                    duration, "duration", 1, 600, allow_none=False
                )
                fps = coerce_number(fps, "fps", 0.1, 30)
                target_width = coerce_number(
                    target_width, "target_width", 64, 7680, integer=True,
                    allow_none=False,
                )
                _, rate = stream_frame_rate(fps, self._compute_interval(duration, fps))
                if duration * rate > MAX_STREAM_FRAMES:
                    raise ServiceValidationError(
                        f"frame_source stream decodes at most {MAX_STREAM_FRAMES} "
                        "frames per camera; reduce duration or fps"
                    )
            clip_plans = None
            self._face_clip_plan = {entity: None for entity in image_entities}
            if resolved_clip is not None:
                # Default to a 1080p cap unless the caller sets record_scale (0 = native)
                scale_width = 1920 if record_scale is None else int(record_scale)
                if self.clip_codec == RECORD_CODEC_COPY:
                    scale_width = 0
                entities = list(image_entities)
                self.requested_clip_paths = [
                    (
                        resolved_clip
                        if len(entities) == 1
                        else self._suffix_path(resolved_clip, camera_entity)
                    )
                    for camera_entity in entities
                ]
                self._face_clip_plan = dict(zip(entities, self.requested_clip_paths))
                if frame_source == FRAME_SOURCE_STREAM:
                    # The frame-capturing ffmpeg process also writes the clip
                    clip_plans = {
                        camera_entity: (path, record_fps, scale_width)
                        for camera_entity, path in zip(
                            entities, self.requested_clip_paths
                        )
                    }
                else:
                    self.clip_task = self.hass.async_create_task(
                        self.record_clip(
                            image_entities=image_entities,
                            duration=duration,
                            resolved_path=resolved_clip,
                            record_fps=record_fps,
                            scale_width=scale_width,
                        ),
                        name="llmvision_clip_recording",
                    )
            await self.record(
                image_entities=image_entities,
                duration=duration,
                max_frames=max_frames,
                target_width=target_width,
                include_filename=include_filename,
                expose_images=expose_images,
                fps=fps,
                polylines=polylines,
                storage_path=storage_path,
                debug_polylines=debug_polylines,
                frame_source=frame_source,
                clip_plans=clip_plans,
            )
            if clip_plans:
                # Report clips in camera order, not in ffmpeg completion order
                written = set(self.clip_paths)
                self.clip_paths = [
                    path for path in self.requested_clip_paths if path in written
                ]
        return self.client

    async def add_visual_data(
        self, image_entities, image_paths, target_width, include_filename, expose_images
    ):
        """Wrapper for add_images for visual data"""
        await self.add_images(
            image_entities=image_entities,
            image_paths=image_paths,
            target_width=target_width,
            include_filename=include_filename,
            expose_images=expose_images,
        )
        return self.client
