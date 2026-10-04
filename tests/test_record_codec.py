"""Tests for record_codec on stream_analyzer_pro (original stream instead of transcode)."""
import asyncio
import logging
import os
import shutil
import struct
import subprocess

import pytest
from unittest.mock import AsyncMock, Mock, patch

from homeassistant.exceptions import ServiceValidationError

from custom_components.llmvision import media_handlers
from custom_components.llmvision.media_handlers import MediaProcessor
from custom_components.llmvision.stream_capture import (
    MAX_MP4_MOOV_BYTES,
    RECORD_CODEC_COPY,
    RECORD_CODEC_H264,
    build_hvc1_remux_cmd,
    build_stream_capture_cmd,
    mp4_video_sample_entry,
    normalize_record_codec,
)

EXEC = "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec"
SOURCE = "custom_components.llmvision.media_handlers._async_get_stream_source"

H264_ARGS_15FPS_1920 = [
    "-t", "5", "-an", "-sn", "-dn",
    "-vf", "scale='min(1920,iw)':-2,fps=15",
    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
    "-profile:v", "high", "-g", "30", "-movflags", "+faststart",
    "-avoid_negative_ts", "make_zero", "-y", "/media/a.mp4",
]
COPY_ARGS = [
    "-t", "5", "-an", "-sn", "-dn", "-c:v", "copy", "-movflags", "+faststart",
    "-avoid_negative_ts", "make_zero", "-y", "/media/a.mp4",
]


def _run_executor(_executor, func, *args):
    return func(*args)


@pytest.fixture
def processor():
    hass = Mock()
    hass.loop = Mock()
    hass.loop.run_in_executor = AsyncMock(side_effect=_run_executor)
    hass.config = Mock()
    hass.config.config_dir = "/config"
    hass.async_create_task = Mock(
        side_effect=lambda coro, **kwargs: asyncio.create_task(coro)
    )
    with patch("custom_components.llmvision.media_handlers.async_get_clientsession"):
        return MediaProcessor(hass, Mock())


# ---------------------------------------------------------------- MP4 builders


def _box(kind, payload=b""):
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full(kind, payload=b""):
    return _box(kind, b"\x00\x00\x00\x00" + payload)


def _trak(handler, entry):
    hdlr = _full(b"hdlr", b"\x00\x00\x00\x00" + handler + b"\x00" * 12)
    stsd = _full(b"stsd", struct.pack(">I", 1) + _box(entry, b"\x00" * 8))
    stbl = _box(b"stbl", stsd)
    return _box(b"trak", _box(b"mdia", hdlr + _box(b"minf", stbl)))


def _mp4(*traks, moov_first=True, mdat=b"\x00" * 32):
    ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isom")
    moov = _box(b"moov", _box(b"mvhd", b"\x00" * 100) + b"".join(traks))
    data = _box(b"mdat", mdat)
    return ftyp + (moov + data if moov_first else data + moov)


# ---------------------------------------------------------------- normalize


class TestNormalizeRecordCodec:
    @pytest.mark.parametrize("value", [None, "", "  ", "h264", "H264"])
    def test_default_is_h264(self, value):
        assert normalize_record_codec(value) == RECORD_CODEC_H264

    def test_copy(self):
        assert normalize_record_codec(" Copy ") == RECORD_CODEC_COPY

    @pytest.mark.parametrize("value", ["h265", "copy;rm", 1])
    def test_invalid_raises(self, value):
        with pytest.raises(ServiceValidationError):
            normalize_record_codec(value)


# ---------------------------------------------------------------- ffmpeg arguments


class TestClipArguments:
    @pytest.mark.parametrize("codec", [None, RECORD_CODEC_H264])
    def test_h264_arguments_unchanged(self, codec):
        args = MediaProcessor._clip_output_args(5, 15, "/media/a.mp4", 1920, codec)
        assert args == H264_ARGS_15FPS_1920
        assert MediaProcessor._clip_output_args(5, 15, "/media/a.mp4", 1920) == args

    def test_copy_arguments(self):
        assert (
            MediaProcessor._clip_output_args(5, None, "/media/a.mp4", 0, "copy")
            == COPY_ARGS
        )

    def test_build_clip_cmd_copy(self):
        cmd = MediaProcessor._build_clip_ffmpeg_cmd(
            "rtsp://cam/x", 5, None, "/media/a.mp4", 0, RECORD_CODEC_COPY
        )
        assert cmd[-len(COPY_ARGS):] == COPY_ARGS
        assert "-vf" not in cmd and "libx264" not in cmd
        assert cmd.count("-i") == 1 and "-rtsp_transport" in cmd

    def test_stream_mode_frames_unchanged_with_copy_clip(self):
        plain = build_stream_capture_cmd("rtsp://cam/x", 5, "1", 2048, 6)
        clip_args = MediaProcessor._clip_output_args(
            5, None, "/media/a.mp4", 0, RECORD_CODEC_COPY
        )
        cmd = build_stream_capture_cmd(
            "rtsp://cam/x", 5, "1", 2048, 6, clip_output_args=clip_args
        )
        # MJPEG pipe output (everything after the second -map) is identical
        frames_part = cmd[len(cmd) - cmd[::-1].index("-map") - 1:]
        plain_part = plain[len(plain) - plain[::-1].index("-map") - 1:]
        assert frames_part == plain_part
        assert cmd.index("/media/a.mp4") < cmd.index("pipe:1")
        assert cmd[cmd.index("-c:v") + 1] == "copy"

    def test_remux_cmd(self):
        cmd = build_hvc1_remux_cmd("/media/a.mp4", "/media/a.hvc1-x.mp4")
        assert cmd[cmd.index("-c") + 1] == "copy"
        assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
        assert cmd[cmd.index("-i") + 1] == "/media/a.mp4"
        assert cmd[-1] == "/media/a.hvc1-x.mp4"
        assert "-nostdin" in cmd and "+faststart" in cmd


# ---------------------------------------------------------------- add_streams


class TestAddStreamsRecordCodec:
    async def _add(self, processor, **kwargs):
        processor.record = AsyncMock()
        processor.record_clip = AsyncMock()
        params = dict(
            image_entities=["camera.a"], duration=15, max_frames=3,
            target_width=2048, include_filename=False, expose_images=False,
            fps=1, clip_path="clips/a.mp4",
        )
        params.update(kwargs)
        await processor.add_streams(**params)
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_default_keeps_h264(self, processor):
        await self._add(processor)
        assert processor.clip_codec == RECORD_CODEC_H264
        assert processor.record_clip.await_args.kwargs["scale_width"] == 1920

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scale", [None, 0, "0", ""])
    async def test_copy_stream_mode_plans_native_clip(self, processor, scale):
        await self._add(
            processor, record_codec="copy", record_scale=scale, frame_source="stream"
        )
        assert processor.clip_codec == RECORD_CODEC_COPY
        path = processor.requested_clip_paths[0]
        assert processor.record.await_args.kwargs["clip_plans"] == {
            "camera.a": (path, None, 0)
        }

    @pytest.mark.asyncio
    async def test_copy_snapshot_mode_records_clip(self, processor):
        await self._add(processor, record_codec="copy")
        processor.record_clip.assert_awaited_once()
        assert processor.record_clip.await_args.kwargs["scale_width"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field,value", [("record_scale", 1920), ("record_fps", 15)]
    )
    async def test_copy_rejects_transcode_options(self, processor, field, value):
        with pytest.raises(ServiceValidationError, match="record_codec 'copy'"):
            await self._add(processor, record_codec="copy", **{field: value})
        processor.record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_codec_raises_without_clip(self, processor):
        with pytest.raises(ServiceValidationError):
            await self._add(processor, clip_path=None, record_codec="h265")
        processor.record.assert_not_awaited()


# ---------------------------------------------------------------- MP4 parsing


class TestMp4VideoSampleEntry:
    @pytest.mark.parametrize("entry", [b"hev1", b"hvc1", b"avc1"])
    def test_reads_video_entry(self, tmp_path, entry):
        path = tmp_path / "a.mp4"
        path.write_bytes(_mp4(_trak(b"vide", entry)))
        assert mp4_video_sample_entry(str(path)) == entry.decode()

    def test_skips_non_video_track_and_moov_at_end(self, tmp_path):
        path = tmp_path / "a.mov"
        path.write_bytes(
            _mp4(_trak(b"soun", b"mp4a"), _trak(b"vide", b"hev1"), moov_first=False)
        )
        assert mp4_video_sample_entry(str(path)) == "hev1"

    def test_large_size_box_before_moov(self, tmp_path):
        mdat = struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 16 + 8) + b"\x00" * 8
        ftyp = _box(b"ftyp", b"isom")
        moov = _box(b"moov", _trak(b"vide", b"hvc1"))
        path = tmp_path / "a.mp4"
        path.write_bytes(ftyp + mdat + moov)
        assert mp4_video_sample_entry(str(path)) == "hvc1"

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"not an mp4 file at all",
            _box(b"ftyp", b"isom") + _box(b"mdat", b"\x00" * 16),
            _box(b"ftyp", b"isom") + struct.pack(">I", 4) + b"moov",
            _box(b"ftyp", b"isom") + _box(b"moov", b"\xff\xff\xff\xfftrak"),
            _box(b"ftyp", b"isom") + _box(b"moov", _trak(b"vide", b"\x00\x01\x02\x03")),
        ],
    )
    def test_invalid_files_return_none(self, tmp_path, data):
        path = tmp_path / "a.mp4"
        path.write_bytes(data)
        assert mp4_video_sample_entry(str(path)) is None

    def test_oversized_moov_is_not_read(self, tmp_path):
        path = tmp_path / "a.mp4"
        header = struct.pack(">I", MAX_MP4_MOOV_BYTES + 9) + b"moov"
        path.write_bytes(_box(b"ftyp", b"isom") + header + b"\x00" * 16)
        assert mp4_video_sample_entry(str(path)) is None

    def test_huge_large_size_box_returns_none(self, tmp_path):
        path = tmp_path / "a.mp4"
        huge = struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 2**63) + b"\x00" * 8
        path.write_bytes(_box(b"ftyp", b"isom") + huge + _box(b"moov"))
        assert mp4_video_sample_entry(str(path)) is None

    def test_size_zero_child_in_moov_returns_none(self, tmp_path):
        path = tmp_path / "a.mp4"
        child = struct.pack(">I", 0) + b"free" + b"\x00" * 24
        path.write_bytes(_box(b"ftyp", b"isom") + _box(b"moov", child))
        assert mp4_video_sample_entry(str(path)) is None


# ---------------------------------------------------------------- hvc1 finalization


class _Proc:
    def __init__(self, returncode=0, on_run=None, block=False, stderr=b""):
        self._final = returncode
        self.returncode = None
        self._on_run = on_run
        self._block = block
        self._stderr = stderr
        self.kill = Mock(side_effect=self._kill)

    def _kill(self):
        self.returncode = -9

    async def communicate(self):
        if self._block:
            await asyncio.Event().wait()
        if self._on_run:
            self._on_run()
        self.returncode = self._final
        return b"", self._stderr

    async def wait(self):
        return self.returncode


def _tmp_from(cmd):
    return cmd[-1]


class TestFinalizeCopiedClip:
    def _clip(self, tmp_path, entry=b"hev1"):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(_mp4(_trak(b"vide", entry)))
        return clip

    @pytest.mark.asyncio
    async def test_h264_mode_does_nothing(self, processor, tmp_path):
        clip = self._clip(tmp_path)
        with patch(EXEC, AsyncMock()) as exec_mock:
            await processor._finalize_copied_clip(str(clip))
        exec_mock.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entry", [b"hvc1", b"avc1"])
    async def test_playable_entry_is_left_alone(self, processor, tmp_path, entry):
        processor.clip_codec = RECORD_CODEC_COPY
        clip = self._clip(tmp_path, entry)
        with patch(EXEC, AsyncMock()) as exec_mock:
            await processor._finalize_copied_clip(str(clip))
        exec_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_hev1_is_retagged_in_place(self, processor, tmp_path):
        processor.clip_codec = RECORD_CODEC_COPY
        clip = self._clip(tmp_path)
        retagged = _mp4(_trak(b"vide", b"hvc1"))
        calls = []

        async def _exec(*cmd, **kwargs):
            calls.append(cmd)
            tmp = _tmp_from(cmd)
            return _Proc(on_run=lambda: open(tmp, "wb").write(retagged))

        with patch(EXEC, AsyncMock(side_effect=_exec)):
            await processor._finalize_copied_clip(str(clip))
        cmd = calls[0]
        assert cmd[cmd.index("-i") + 1] == str(clip)
        tmp = _tmp_from(cmd)
        assert os.path.dirname(tmp) == str(tmp_path)
        assert tmp.endswith(".mp4") and tmp != str(clip)
        assert mp4_video_sample_entry(str(clip)) == "hvc1"
        assert os.listdir(tmp_path) == ["clip.mp4"]

    @pytest.mark.asyncio
    async def test_failed_remux_keeps_original(self, processor, tmp_path, caplog):
        processor.clip_codec = RECORD_CODEC_COPY
        clip = self._clip(tmp_path)
        original = clip.read_bytes()

        async def _exec(*cmd, **kwargs):
            tmp = _tmp_from(cmd)
            return _Proc(
                returncode=1,
                on_run=lambda: open(tmp, "wb").write(b"partial"),
                stderr=b"Invalid data",
            )

        caplog.set_level(logging.WARNING, logger=media_handlers.__name__)
        with patch(EXEC, AsyncMock(side_effect=_exec)):
            await processor._finalize_copied_clip(str(clip))
        assert clip.read_bytes() == original
        assert os.listdir(tmp_path) == ["clip.mp4"]
        assert "hvc1 failed" in caplog.text

    @pytest.mark.asyncio
    async def test_remux_without_effect_keeps_original(self, processor, tmp_path):
        processor.clip_codec = RECORD_CODEC_COPY
        clip = self._clip(tmp_path)
        original = clip.read_bytes()

        async def _exec(*cmd, **kwargs):
            tmp = _tmp_from(cmd)
            return _Proc(on_run=lambda: open(tmp, "wb").write(original))

        with patch(EXEC, AsyncMock(side_effect=_exec)):
            await processor._finalize_copied_clip(str(clip))
        assert clip.read_bytes() == original
        assert os.listdir(tmp_path) == ["clip.mp4"]

    @pytest.mark.asyncio
    async def test_timeout_kills_and_cleans_up(self, processor, tmp_path, monkeypatch):
        monkeypatch.setattr(media_handlers, "REMUX_TIMEOUT", 0.05)
        processor.clip_codec = RECORD_CODEC_COPY
        clip = self._clip(tmp_path)
        procs = []

        async def _exec(*cmd, **kwargs):
            open(_tmp_from(cmd), "wb").write(b"partial")
            procs.append(_Proc(block=True))
            return procs[0]

        with patch(EXEC, AsyncMock(side_effect=_exec)):
            await processor._finalize_copied_clip(str(clip))
        procs[0].kill.assert_called_once()
        assert os.listdir(tmp_path) == ["clip.mp4"]

    @pytest.mark.asyncio
    async def test_cancellation_kills_and_cleans_up(self, processor, tmp_path):
        processor.clip_codec = RECORD_CODEC_COPY
        clip = self._clip(tmp_path)
        procs = []
        started = asyncio.Event()

        async def _exec(*cmd, **kwargs):
            open(_tmp_from(cmd), "wb").write(b"partial")
            procs.append(_Proc(block=True))
            started.set()
            return procs[0]

        with patch(EXEC, AsyncMock(side_effect=_exec)):
            task = asyncio.create_task(processor._finalize_copied_clip(str(clip)))
            await started.wait()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        procs[0].kill.assert_called_once()
        assert os.listdir(tmp_path) == ["clip.mp4"]

    @pytest.mark.asyncio
    async def test_unreadable_clip_is_kept(self, processor, tmp_path, caplog):
        processor.clip_codec = RECORD_CODEC_COPY
        caplog.set_level(logging.WARNING, logger=media_handlers.__name__)
        with patch(EXEC, AsyncMock()) as exec_mock:
            await processor._finalize_copied_clip(str(tmp_path / "missing.mp4"))
        exec_mock.assert_not_awaited()
        assert "Could not inspect clip" in caplog.text


class TestRecordClipCopy:
    @pytest.mark.asyncio
    async def test_copy_clip_finalized_before_it_is_reported(self, processor, tmp_path):
        processor.clip_codec = RECORD_CODEC_COPY
        out = str(tmp_path / "clip.mp4")
        seen = []

        async def _finalize(path):
            seen.append((path, list(processor.clip_paths)))

        processor._finalize_copied_clip = _finalize

        async def _exec(*cmd, **kwargs):
            return _Proc(on_run=lambda: open(out, "wb").write(b"video"))

        exec_mock = AsyncMock(side_effect=_exec)
        with patch(SOURCE, AsyncMock(return_value="rtsp://x")), patch(EXEC, exec_mock):
            result = await processor.record_clip(
                ["camera.front"], 5, out, scale_width=0
            )
        assert result == [out]
        assert seen == [(out, [])]
        cmd = exec_mock.await_args.args
        assert cmd[cmd.index("-c:v") + 1] == "copy" and "-vf" not in cmd


# ---------------------------------------------------------------- real ffmpeg


def _has_libx265():
    return _has_encoders("libx265")


def _has_encoders(*names):
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return all(name in out for name in names)


def _frame_id(gray):
    """Source frame number encoded by _numbered_stream in the two image halves."""
    import numpy as np

    g = np.asarray(gray, dtype=float)
    h, w = g.shape

    def digit(region):
        return int(round((region.mean() * 219 / 255 + 8) / 16))

    lo = digit(g[h // 4 : 3 * h // 4, w // 8 : 3 * w // 8])
    hi = digit(g[h // 4 : 3 * h // 4, 5 * w // 8 : 7 * w // 8])
    return hi * 16 + lo


def _numbered_stream(tmp_path, encoder):
    """MPEG-TS, 15 fps, keyframe every 15 frames, cut inside a GOP like a live join."""
    source = tmp_path / "numbered.ts"
    params = (
        ["-x265-params", "keyint=15:min-keyint=15:scenecut=0:bframes=0:log-level=error"]
        if encoder == "libx265"
        else ["-g", "15", "-keyint_min", "15", "-sc_threshold", "0", "-bf", "0"]
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "color=black:s=128x96:r=15,format=yuv420p,"
         "geq=lum='if(lt(X,W/2),mod(N,16)*16+8,mod(floor(N/16),16)*16+8)'"
         ":cb=128:cr=128",
         "-t", "6", "-c:v", encoder, *params, "-f", "mpegts", str(source)],
        check=True,
    )
    data = source.read_bytes()
    cut = tmp_path / "joined.ts"
    # 188-byte TS packets; start roughly a third into the second GOP
    cut.write_bytes(data[(len(data) * 4 // 9) // 188 * 188 :])
    return cut


@pytest.mark.skipif(
    not _has_encoders("libx265", "libx264"), reason="ffmpeg encoders not available"
)
class TestCopyClipTiming:
    @pytest.mark.parametrize("encoder", ["libx265", "libx264"])
    def test_copy_clip_starts_with_first_analysis_frame(self, tmp_path, encoder):
        """Face labels map clip time to frame index; both must share time zero."""
        import io

        from PIL import Image

        from custom_components.llmvision.stream_capture import JpegStreamSplitter

        joined = _numbered_stream(tmp_path, encoder)
        clip = tmp_path / "clip.mp4"
        clip_args = MediaProcessor._clip_output_args(3, None, str(clip), 0, "copy")
        cmd = build_stream_capture_cmd(
            str(joined), 3, "15", 128, 60, clip_output_args=clip_args
        )
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        frames = JpegStreamSplitter().feed(out)
        pipe_ids = [_frame_id(Image.open(io.BytesIO(f)).convert("L")) for f in frames]

        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(clip), "-f", "rawvideo",
             "-pix_fmt", "gray", "-"],
            capture_output=True, check=True,
        ).stdout
        size = 128 * 96
        clip_ids = [
            _frame_id(
                Image.frombytes("L", (128, 96), raw[i * size : (i + 1) * size])
            )
            for i in range(len(raw) // size)
        ]
        assert pipe_ids and clip_ids
        # The join is mid-GOP: both outputs start at the same (key)frame
        assert clip_ids[0] == pipe_ids[0]
        assert clip_ids[0] % 15 == 0
        assert clip_ids[: len(pipe_ids)] == pipe_ids[: len(clip_ids)]


@pytest.mark.skipif(not _has_libx265(), reason="ffmpeg with libx265 not available")
class TestRealFfmpeg:
    @pytest.mark.asyncio
    async def test_copied_hevc_clip_becomes_hvc1(self, processor, tmp_path):
        source = tmp_path / "source.ts"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             "testsrc2=size=128x72:rate=15", "-t", "1", "-c:v", "libx265",
             "-x265-params", "log-level=error", "-f", "mpegts", str(source)],
            check=True,
        )
        clip = tmp_path / "clip.mp4"
        args = MediaProcessor._clip_output_args(1, None, str(clip), 0, "copy")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), *args], check=True
        )
        # ffmpeg's default H.265 tag in MP4 is hev1, which Apple players reject
        assert mp4_video_sample_entry(str(clip)) == "hev1"

        processor.clip_codec = RECORD_CODEC_COPY
        await processor._finalize_copied_clip(str(clip))
        assert mp4_video_sample_entry(str(clip)) == "hvc1"
        assert sorted(os.listdir(tmp_path)) == ["clip.mp4", "source.ts"]
        probe = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(clip), "-f", "null", "-"],
            capture_output=True, text=True,
        )
        assert probe.returncode == 0 and not probe.stderr
