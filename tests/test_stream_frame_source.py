"""Tests for stream-sourced frames and high-quality JPEG on the pro analyzers."""
import asyncio
import base64
import io
import logging

import pytest
from unittest.mock import AsyncMock, Mock, patch
from PIL import Image, JpegImagePlugin

from homeassistant.exceptions import ServiceValidationError

from custom_components.llmvision import ServiceCallData
from custom_components.llmvision import media_handlers
from custom_components.llmvision.media_handlers import MediaProcessor
from custom_components.llmvision.stream_capture import (
    PRO_FFMPEG_JPEG_Q,
    PRO_JPEG_OPTIONS,
    JpegStreamSplitter,
    build_stream_capture_cmd,
    coerce_number,
    normalize_frame_source,
    redact,
    stream_frame_cap,
    stream_frame_rate,
    stream_input_args,
)

SECRET_URL = "rtsp://admin:s3cretpw@192.168.1.20:554/h264Preview_01_main"


def _jpeg(color, size=(64, 32)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _run_executor(_executor, func, *args):
    return func(*args)


@pytest.fixture
def processor():
    hass = Mock()
    hass.loop = Mock()
    hass.loop.run_in_executor = AsyncMock(side_effect=_run_executor)
    hass.states = Mock()
    hass.config = Mock()
    hass.config.config_dir = "/config"
    hass.async_create_task = Mock(
        side_effect=lambda coro, **kwargs: asyncio.create_task(coro)
    )
    client = Mock()
    client.add_frame = Mock()
    with patch("custom_components.llmvision.media_handlers.async_get_clientsession"):
        return MediaProcessor(hass, client)


class FakeStream:
    """Minimal asyncio StreamReader double fed with predefined chunks."""

    def __init__(self, chunks=(), block=False):
        self._chunks = list(chunks)
        self._block = block

    async def read(self, _n=-1):
        if self._chunks:
            return self._chunks.pop(0)
        if self._block:
            await asyncio.Event().wait()
        return b""


class FakeProcess:
    def __init__(
        self, stdout, stderr=b"", on_exit=None, returncode=0, ignore_terminate=False
    ):
        self.stdout = stdout
        self.stderr = FakeStream([stderr] if stderr else [])
        self.returncode = None
        self._final_returncode = returncode
        self._on_exit = on_exit
        self._ignore_terminate = ignore_terminate
        self._killed = asyncio.Event()
        self.terminate = Mock(
            side_effect=None if ignore_terminate else self._finish
        )
        self.kill = Mock(side_effect=self._kill)

    def _finish(self):
        if self.returncode is None:
            self.returncode = self._final_returncode

    def _kill(self):
        self._killed.set()
        if self.returncode is None:
            self.returncode = -9

    async def wait(self):
        if self._on_exit:
            self._on_exit()
            self._on_exit = None
        if self._ignore_terminate and self.returncode is None:
            await self._killed.wait()
        self._finish()
        return self.returncode


# ---------------------------------------------------------------- stream_capture


class TestRedact:
    def test_masks_userinfo(self):
        assert "s3cretpw" not in redact(f"cmd -i {SECRET_URL}")
        assert "rtsp://***@192.168.1.20" in redact(SECRET_URL)

    def test_masks_query_values(self):
        text = redact("rtmp://cam/bcs/channel0_main.bcs?user=admin&password=s3cretpw")
        assert "s3cretpw" not in text and "admin" not in text
        assert "?user=***&password=***" in text

    def test_masks_known_password_in_stderr(self):
        stderr = "method DESCRIBE failed: 401 Unauthorized (s3cretpw)"
        assert "s3cretpw" not in redact(stderr, SECRET_URL)

    def test_masks_percent_encoded_password(self):
        url = "rtsp://admin:p%40ss%21word@cam/stream"
        assert "p@ss!word" not in redact("auth p@ss!word failed", url)

    def test_none_is_empty(self):
        assert redact(None) == ""


class TestNormalizeFrameSource:
    @pytest.mark.parametrize("value", [None, "", "  ", "snapshot", "SNAPSHOT"])
    def test_snapshot_default(self, value):
        assert normalize_frame_source(value) == "snapshot"

    def test_stream(self):
        assert normalize_frame_source(" Stream ") == "stream"

    @pytest.mark.parametrize("value", ["bogus", "rtsp", 1])
    def test_invalid_raises(self, value):
        with pytest.raises(ServiceValidationError):
            normalize_frame_source(value)


class TestCoerceNumber:
    @pytest.mark.parametrize(
        "value",
        ["15,metadata=mode=print:file=/config/x", "inf", "nan", -1, True, 1e9, "abc"],
    )
    def test_rejects_unsafe_values(self, value):
        with pytest.raises(ServiceValidationError):
            coerce_number(value, "record_fps", 1, 60)

    def test_accepts_numbers(self):
        assert coerce_number("15", "record_fps", 1, 60) == 15.0
        assert coerce_number(2048, "target_width", 64, 7680, integer=True) == 2048

    def test_integer_rejects_fraction(self):
        with pytest.raises(ServiceValidationError):
            coerce_number(2048.5, "target_width", 64, 7680, integer=True)

    def test_none_allowed_or_required(self):
        assert coerce_number(None, "fps", 0.1, 30) is None
        with pytest.raises(ServiceValidationError):
            coerce_number(None, "duration", 1, 600, allow_none=False)


class TestFrameRate:
    def test_fps_wins(self):
        assert stream_frame_rate(2, 0.5) == ("2", 2.0)
        assert stream_frame_rate(0.5, 2) == ("0.5", 0.5)

    def test_legacy_interval_is_exact_fraction(self):
        text, rate = stream_frame_rate(None, 3)
        assert text == "1/3"
        assert rate == pytest.approx(1 / 3)

    def test_frame_cap(self):
        assert stream_frame_cap(15, 1.0) == 16
        assert stream_frame_cap(600, 30.0) == 300


class TestStreamCommand:
    def test_input_args_scoped_to_rtsp(self):
        assert stream_input_args("rtsp://cam/x") == ["-rtsp_transport", "tcp"]
        assert stream_input_args("rtsps://cam/x") == ["-rtsp_transport", "tcp"]
        assert stream_input_args("rtmp://cam/x") == []
        assert stream_input_args("https://cam/flv?x=1") == []

    def test_frames_only(self):
        cmd = build_stream_capture_cmd("http://cam/flv", 15, "1", 2048, 16)
        assert "-rtsp_transport" not in cmd
        assert cmd.count("-map") == 1
        assert cmd[cmd.index("-t") + 1] == "15"
        vf = cmd[cmd.index("-vf") + 1]
        assert vf.startswith("fps=1,")
        assert "min(2048,iw)" in vf and "min(2048,ih)" in vf
        assert "force_original_aspect_ratio=decrease" in vf
        assert cmd[cmd.index("-frames:v") + 1] == "16"
        assert cmd[cmd.index("-q:v") + 1] == str(PRO_FFMPEG_JPEG_Q)
        assert cmd[-3:] == ["-f", "image2pipe", "pipe:1"]

    def test_with_clip_shares_input(self, processor):
        clip_args = processor._clip_output_args(15, None, "/media/a.mp4", 1920)
        cmd = build_stream_capture_cmd(
            "rtsp://cam/x", 15, "1/3", 2048, 6, clip_output_args=clip_args
        )
        assert cmd.count("-i") == 1
        assert cmd.count("-map") == 2
        assert cmd.index("/media/a.mp4") < cmd.index("pipe:1")
        assert "libx264" in cmd and "-rtsp_transport" in cmd
        assert "fps=1/3," in cmd[len(cmd) - cmd[::-1].index("-vf")]


class TestJpegStreamSplitter:
    def test_splits_across_chunks(self):
        a, b = _jpeg("red"), _jpeg("blue")
        data = a + b
        splitter = JpegStreamSplitter()
        found = []
        for i in range(0, len(data), 7):
            found += splitter.feed(data[i : i + 7])
        assert found == [a, b]

    def test_multiple_per_chunk_and_junk(self):
        a, b = _jpeg("red"), _jpeg("green")
        splitter = JpegStreamSplitter()
        assert splitter.feed(b"garbage" + a + b + b"\xff") == [a, b]

    def test_oversized_frame_is_dropped(self):
        splitter = JpegStreamSplitter(max_frame_bytes=100)
        assert splitter.feed(b"\xff\xd8" + b"\x00" * 200) == []
        assert splitter.malformed == 1
        good = _jpeg("red")
        splitter.max_frame_bytes = len(good) + 10
        assert splitter.feed(good) == [good]


# ---------------------------------------------------------------- media_handlers


class TestClipCommandHardening:
    def test_rejects_filter_injection(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._build_clip_ffmpeg_cmd(
                "rtsp://x", 5, "15,movie=/config/secrets.yaml", "/media/a.mp4"
            )

    def test_rtsp_transport_only_for_rtsp(self, processor):
        cmd = processor._build_clip_ffmpeg_cmd("rtmp://x/y", 5, None, "/media/a.mp4")
        assert "-rtsp_transport" not in cmd

    def test_clip_path_extension_enforced(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._resolve_output_file("clips/secrets.yaml")
        assert processor._resolve_output_file("clips/a.MOV").endswith("a.MOV")


class TestJpegQuality:
    @pytest.mark.asyncio
    async def test_default_keeps_pillow_defaults(self, processor):
        encoded = await processor._encode_image(Image.new("RGB", (64, 64), "red"))
        img = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert JpegImagePlugin.get_sampling(img) == 2  # 4:2:0

    @pytest.mark.asyncio
    async def test_pro_options_use_444(self, processor):
        processor.jpeg_options = PRO_JPEG_OPTIONS
        encoded = await processor._encode_image(Image.new("RGB", (64, 64), "red"))
        img = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert JpegImagePlugin.get_sampling(img) == 0  # 4:4:4

    @pytest.mark.asyncio
    async def test_polyline_frames_use_pro_options_without_resampling(self, processor):
        processor.jpeg_options = PRO_JPEG_OPTIONS
        source = _jpeg("white", size=(2048, 1149))
        b64, size, _ = await processor._draw_polylines_on_image(
            source, 2048, [[(0.1, 0.1), (0.9, 0.9)]]
        )
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert size == (2048, 1149)
        assert JpegImagePlugin.get_sampling(img) == 0

    def test_video_ffmpeg_quality_default(self, processor):
        assert processor.ffmpeg_jpeg_q == 5


class TestAddStreamsFrameSource:
    @pytest.mark.asyncio
    async def test_invalid_frame_source_raises_before_capture(self, processor):
        processor.record = AsyncMock()
        with pytest.raises(ServiceValidationError):
            await processor.add_streams(
                image_entities=["camera.a"], duration=5, max_frames=3,
                target_width=1280, include_filename=False, expose_images=False,
                frame_source="bogus",
            )
        processor.record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_mode_plans_clip_in_same_process(self, processor):
        processor.record = AsyncMock()
        processor.record_clip = AsyncMock()
        await processor.add_streams(
            image_entities=["camera.a"], duration=15, max_frames=3,
            target_width=2048, include_filename=False, expose_images=False,
            fps=1, clip_path="clips/a.mp4", record_fps=15, record_scale=0,
            frame_source="stream",
        )
        processor.record_clip.assert_not_awaited()
        assert processor.clip_task is None
        kwargs = processor.record.await_args.kwargs
        assert kwargs["frame_source"] == "stream"
        path = processor.requested_clip_paths[0]
        assert kwargs["clip_plans"] == {"camera.a": (path, 15.0, 0)}

    @pytest.mark.asyncio
    async def test_snapshot_mode_keeps_background_clip(self, processor):
        processor.record = AsyncMock()
        processor.record_clip = AsyncMock()
        await processor.add_streams(
            image_entities=["camera.a"], duration=5, max_frames=3,
            target_width=1280, include_filename=False, expose_images=False,
            clip_path="clips/a.mp4",
        )
        await asyncio.sleep(0)
        processor.record_clip.assert_awaited_once()
        assert processor.record.await_args.kwargs["clip_plans"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field,value",
        [("fps", "1,movie=/x"), ("target_width", "2048;x"), ("duration", 0)],
    )
    async def test_stream_mode_validates_numbers(self, processor, field, value):
        processor.record = AsyncMock()
        kwargs = dict(
            image_entities=["camera.a"], duration=5, max_frames=3,
            target_width=1280, include_filename=False, expose_images=False,
            frame_source="stream",
        )
        kwargs[field] = value
        with pytest.raises(ServiceValidationError):
            await processor.add_streams(**kwargs)


class TestAddStreamsBudgetAndOrder:
    @pytest.mark.asyncio
    async def test_rejects_more_than_frame_budget(self, processor):
        processor.record = AsyncMock()
        with pytest.raises(ServiceValidationError, match="at most 300"):
            await processor.add_streams(
                image_entities=["camera.a"], duration=60, max_frames=3,
                target_width=2048, include_filename=False, expose_images=False,
                fps=10, frame_source="stream",
            )
        processor.record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clip_paths_follow_camera_order(self, processor):
        async def _record(**kwargs):
            plans = kwargs["clip_plans"]
            # Simulate the second camera's ffmpeg finishing first
            processor.clip_paths.extend(p[0] for p in reversed(list(plans.values())))

        processor.record = AsyncMock(side_effect=_record)
        await processor.add_streams(
            image_entities=["camera.a", "camera.b"], duration=5, max_frames=3,
            target_width=2048, include_filename=False, expose_images=False,
            clip_path="clips/x.mp4", frame_source="stream",
        )
        assert processor.clip_paths == processor.requested_clip_paths
        assert processor.clip_paths[0].endswith("x-a.mp4")


class TestRecordStreamMode:
    @pytest.mark.asyncio
    async def test_stream_frames_feed_existing_pipeline(self, processor):
        first = ("a-frame-0", _jpeg("red"))
        frames = {
            "a-frame-1": {
                "frame_data": _jpeg("blue"), "ssim_score": 0.1,
                "camera_number": 0, "frame_index": 1,
            }
        }
        processor._capture_stream_camera = AsyncMock(return_value=(first, frames))
        await processor.record(
            image_entities=["camera.a"], duration=5, max_frames=None,
            target_width=2048, include_filename=True, expose_images=False,
            fps=1, frame_source="stream",
        )
        labels = [c.kwargs["filename"] for c in processor.client.add_frame.call_args_list]
        assert labels == ["a-frame-0", "a-frame-1"]
        processor.hass.states.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_unusable_stream_falls_back_to_snapshots(self, processor):
        processor._capture_stream_camera = AsyncMock(return_value=None)
        processor.hass.states.get = Mock(return_value=None)
        with patch("custom_components.llmvision.media_handlers.get_url", return_value=""):
            with pytest.raises(ServiceValidationError, match="No cameras available"):
                await processor.record(
                    image_entities=["camera.a"], duration=0.1, max_frames=None,
                    target_width=2048, include_filename=True, expose_images=False,
                    fps=30, frame_source="stream",
                )
        processor.hass.states.get.assert_called()

    def _snapshot_camera(self, processor):
        state = Mock()
        state.attributes = {"entity_picture": "/api/camera_proxy/camera.a"}
        processor.hass.states.get = Mock(return_value=state)
        colors = iter(["red", "blue", "green", "white", "black"] * 20)
        processor._fetch = AsyncMock(side_effect=lambda *a, **k: _jpeg(next(colors)))

    @pytest.mark.asyncio
    async def test_fallback_snapshots_are_analyzed(self, processor, caplog):
        processor._capture_stream_camera = AsyncMock(return_value=None)
        self._snapshot_camera(processor)
        with patch("custom_components.llmvision.media_handlers.get_url", return_value=""):
            await processor.record(
                image_entities=["camera.a"], duration=0.15, max_frames=None,
                target_width=2048, include_filename=True, expose_images=False,
                fps=20, frame_source="stream",
            )
        assert processor.client.add_frame.call_count >= 2
        assert "Falling back to snapshot capture for camera.a" in caplog.text

    @pytest.mark.asyncio
    async def test_mixed_cameras_stream_and_fallback(self, processor):
        first = ("b-frame-0", _jpeg("red"))

        async def _capture(**kwargs):
            if kwargs["image_entity"] == "camera.b":
                return first, {}
            return None

        processor._capture_stream_camera = AsyncMock(side_effect=_capture)
        self._snapshot_camera(processor)
        with patch("custom_components.llmvision.media_handlers.get_url", return_value=""):
            await processor.record(
                image_entities=["camera.a", "camera.b"], duration=0.15,
                max_frames=None, target_width=2048, include_filename=True,
                expose_images=False, fps=20, frame_source="stream",
            )
        labels = [c.kwargs["filename"] for c in processor.client.add_frame.call_args_list]
        assert "b-frame-0" in labels
        assert any(label.startswith("a-frame-") for label in labels)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("options,sampling", [({}, 2), (PRO_JPEG_OPTIONS, 0)])
    async def test_snapshot_capture_encoding_follows_options(
        self, processor, options, sampling
    ):
        processor.jpeg_options = options
        self._snapshot_camera(processor)
        captured = []
        original = MediaProcessor._select_stream_frames

        def _spy(*args):
            selected = original(*args)
            captured.extend(data for _, data, _ in selected)
            return selected

        processor._select_stream_frames = _spy
        with patch("custom_components.llmvision.media_handlers.get_url", return_value=""):
            await processor.record(
                image_entities=["camera.a"], duration=0.15, max_frames=None,
                target_width=2048, include_filename=True, expose_images=False,
                fps=20,
            )
        assert captured
        for data in captured:
            assert JpegImagePlugin.get_sampling(Image.open(io.BytesIO(data))) == sampling

    @pytest.mark.asyncio
    async def test_snapshot_mode_never_resolves_stream(self, processor):
        processor.hass.states.get = Mock(return_value=None)
        with (
            patch("custom_components.llmvision.media_handlers.get_url", return_value=""),
            patch(
                "custom_components.llmvision.media_handlers._async_get_stream_source",
                AsyncMock(),
            ) as source,
        ):
            with pytest.raises(ServiceValidationError):
                await processor.record(
                    image_entities=["camera.a"], duration=0.1, max_frames=None,
                    target_width=1280, include_filename=True, expose_images=False,
                    fps=30,
                )
        source.assert_not_awaited()


class TestCaptureStreamCamera:
    def _patches(self, proc, url=SECRET_URL):
        exec_mock = AsyncMock(return_value=proc)
        return exec_mock, (
            patch(
                "custom_components.llmvision.media_handlers._async_get_stream_source",
                AsyncMock(return_value=url),
            ),
            patch(
                "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
                exec_mock,
            ),
        )

    async def _capture(self, processor, proc, url=SECRET_URL, **kwargs):
        exec_mock, (p1, p2) = self._patches(proc, url)
        params = dict(
            image_entity="camera.front", camera_number=0, duration=5,
            frame_rate="1", frame_cap=16, target_width=2048, include_filename=True,
        )
        params.update(kwargs)
        with p1, p2:
            result = await processor._capture_stream_camera(**params)
        return result, exec_mock

    @pytest.mark.asyncio
    async def test_frames_labels_and_scores(self, processor, caplog):
        data = _jpeg("red") + _jpeg("blue") + _jpeg("green")
        proc = FakeProcess(FakeStream([data[:50], data[50:]]), stderr=b"warn s3cretpw")
        caplog.set_level(logging.DEBUG, logger=media_handlers.__name__)
        result, exec_mock = await self._capture(processor, proc)
        first, frames = result
        assert first[0] == "front-frame-0"
        assert list(frames) == ["front-frame-1", "front-frame-2"]
        assert all(isinstance(f["ssim_score"], float) for f in frames.values())
        assert [f["frame_index"] for f in frames.values()] == [1, 2]
        assert "pipe:1" in exec_mock.await_args.args
        assert "s3cretpw" not in caplog.text

    @pytest.mark.asyncio
    async def test_frame_cap_keeps_draining(self, processor):
        data = b"".join(_jpeg(c) for c in ("red", "blue", "green", "white", "black"))
        stdout = FakeStream([data])
        proc = FakeProcess(stdout)
        result, _ = await self._capture(processor, proc, frame_cap=2)
        first, frames = result
        assert list(frames) == ["front-frame-1"]
        assert stdout._chunks == []

    @pytest.mark.asyncio
    async def test_no_frames_returns_none(self, processor, caplog):
        proc = FakeProcess(FakeStream([]), returncode=1, stderr=b"401 s3cretpw")
        caplog.set_level(logging.DEBUG, logger=media_handlers.__name__)
        result, _ = await self._capture(processor, proc)
        assert result is None
        assert "ended early" in caplog.text
        assert "s3cretpw" not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", [None, "file:///config/secrets.yaml"])
    async def test_unusable_source_returns_none(self, processor, url):
        proc = FakeProcess(FakeStream([]))
        result, exec_mock = await self._capture(processor, proc, url=url)
        assert result is None
        exec_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_startup_timeout_terminates(self, processor, monkeypatch):
        monkeypatch.setattr(media_handlers, "STREAM_STARTUP_TIMEOUT", 0.05)
        proc = FakeProcess(FakeStream([], block=True))
        result, _ = await self._capture(processor, proc)
        assert result is None
        proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_clip_written_by_same_process(self, processor, tmp_path):
        clip = tmp_path / "clips" / "a.mp4"

        def write_clip():
            clip.write_bytes(b"mp4")

        proc = FakeProcess(FakeStream([_jpeg("red")]), on_exit=write_clip)
        result, exec_mock = await self._capture(
            processor, proc, clip_plan=(str(clip), None, 1920)
        )
        assert result is not None
        assert processor.clip_paths == [str(clip)]
        args = exec_mock.await_args.args
        assert str(clip) in args and "libx264" in args and args.count("-i") == 1

    @pytest.mark.asyncio
    async def test_cancellation_kills_process(self, processor, tmp_path):
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"partial")
        proc = FakeProcess(FakeStream([], block=True))
        exec_mock, (p1, p2) = self._patches(proc)
        with p1, p2:
            task = asyncio.create_task(
                processor._capture_stream_camera(
                    image_entity="camera.front", camera_number=0, duration=5,
                    frame_rate="1", frame_cap=16, target_width=2048,
                    include_filename=True, clip_plan=(str(clip), None, 1920),
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        proc.kill.assert_called_once()
        assert not clip.exists()

    @pytest.mark.asyncio
    async def test_kill_after_terminate_timeout_removes_partial_clip(
        self, processor, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(media_handlers, "STREAM_STARTUP_TIMEOUT", 0.05)
        monkeypatch.setattr(media_handlers, "PROCESS_STOP_TIMEOUT", 0.05)
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"partial")
        proc = FakeProcess(FakeStream([], block=True), ignore_terminate=True)
        result, _ = await self._capture(
            processor, proc, clip_plan=(str(clip), None, 1920)
        )
        assert result is None
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert not clip.exists()
        assert processor.clip_paths == []

    @pytest.mark.asyncio
    async def test_frames_kept_when_wall_deadline_hits(self, processor, monkeypatch):
        monkeypatch.setattr(media_handlers, "STREAM_GRACE", 0.1)
        proc = FakeProcess(FakeStream([_jpeg("red") + _jpeg("blue")], block=True))
        result, _ = await self._capture(processor, proc, duration=0.05)
        first, frames = result
        assert first[0] == "front-frame-0" and list(frames) == ["front-frame-1"]
        proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_stall_timeout_after_frame_cap(self, processor, monkeypatch):
        # Frame output done, clip still recording: wait for the wall deadline only
        monkeypatch.setattr(media_handlers, "STREAM_STALL_TIMEOUT", 0.01)
        monkeypatch.setattr(media_handlers, "STREAM_GRACE", 0.3)
        proc = FakeProcess(FakeStream([_jpeg("red")], block=True))
        loop = asyncio.get_running_loop()
        started = loop.time()
        result, _ = await self._capture(processor, proc, duration=0.0, frame_cap=1)
        assert result is not None
        assert loop.time() - started >= 0.25

    @pytest.mark.asyncio
    async def test_unexpected_error_kills_process(self, processor, tmp_path):
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"partial")
        processor.hass.loop.run_in_executor = AsyncMock(
            side_effect=lambda _e, func, *args: (
                (_ for _ in ()).throw(RuntimeError("executor shut down"))
                if func == processor._analyze_stream_frame
                else func(*args)
            )
        )
        proc = FakeProcess(FakeStream([_jpeg("red")], block=True), returncode=None)
        with pytest.raises(RuntimeError):
            await self._capture(processor, proc, clip_plan=(str(clip), None, 1920))
        proc.kill.assert_called_once()
        assert not clip.exists()

    @pytest.mark.asyncio
    async def test_malformed_stream_stops_promptly(self, processor, monkeypatch):
        monkeypatch.setattr(media_handlers, "MAX_MALFORMED_FRAMES", 1)
        monkeypatch.setattr(
            media_handlers,
            "JpegStreamSplitter",
            lambda: JpegStreamSplitter(max_frame_bytes=10),
        )
        proc = FakeProcess(FakeStream([b"\xff\xd8" + b"\x00" * 50], block=True))
        loop = asyncio.get_running_loop()
        started = loop.time()
        result, _ = await self._capture(processor, proc)
        assert result is None
        assert loop.time() - started < 2
        proc.terminate.assert_called_once()

    def test_oversized_frame_skipped(self, processor, monkeypatch):
        monkeypatch.setattr(media_handlers, "MAX_FRAME_PIXELS", 100)
        assert processor._analyze_stream_frame(_jpeg("red"), None) is None

    def test_corrupt_frame_skipped(self, processor):
        assert processor._analyze_stream_frame(b"\xff\xd8junk\xff\xd9", None) is None


# ---------------------------------------------------------------- service handlers


def _build_data_call(data):
    data_call = Mock()
    data_call.data = Mock()
    data_call.data.get = Mock(side_effect=lambda key, default=None: data.get(key, default))
    return data_call


def _handlers():
    hass = Mock()
    hass.data = {}
    hass.config = Mock()
    hass.config.path = Mock(return_value="/tmp")
    hass.services = Mock()
    hass.http = Mock()
    from custom_components.llmvision import setup

    assert setup(hass, {}) is True
    return {c.args[1]: c.args[2] for c in hass.services.register.call_args_list}


class TestProHandlers:
    def test_service_call_data_reads_frame_source(self):
        call = ServiceCallData(_build_data_call({"frame_source": "stream"}))
        assert call.frame_source == "stream"
        assert ServiceCallData(_build_data_call({})).frame_source is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "service,method", [("stream_analyzer_pro", "add_streams"), ("video_analyzer_pro", "add_videos")]
    )
    async def test_pro_handlers_enable_high_quality(self, service, method):
        handlers = _handlers()
        data = {"provider": "e", "message": "m", "frame_source": "stream"}
        call_obj = ServiceCallData(_build_data_call(data))
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = ""
        processor.debug_info = None
        processor.clip_paths = []
        processor.requested_clip_paths = []
        setattr(processor, method, AsyncMock(return_value=request_obj))

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            await handlers[service](_build_data_call(data))

        assert processor.jpeg_options == PRO_JPEG_OPTIONS
        assert processor.ffmpeg_jpeg_q == PRO_FFMPEG_JPEG_Q
        if method == "add_streams":
            assert processor.add_streams.await_args.kwargs["frame_source"] == "stream"


class TestVideoQuality:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("q,expected", [(None, "5"), (PRO_FFMPEG_JPEG_Q, "2")])
    async def test_add_video_ffmpeg_quality(self, processor, tmp_path, q, expected):
        if q is not None:
            processor.ffmpeg_jpeg_q = q
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        proc = FakeProcess(FakeStream([]))
        exec_mock = AsyncMock(return_value=proc)
        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            exec_mock,
        ):
            with pytest.raises(Exception):
                await processor.add_video(str(video), base_url="")
        args = exec_mock.await_args.args
        assert args[args.index("-q:v") + 1] == expected


class TestUpstreamHandlersUnchanged:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "service,method",
        [
            ("image_analyzer", "add_images"),
            ("video_analyzer", "add_videos"),
            ("stream_analyzer", "add_streams"),
            ("data_analyzer", "add_visual_data"),
        ],
    )
    async def test_upstream_handlers_keep_default_jpeg(self, service, method):
        handlers = _handlers()
        data = {"provider": "e", "message": "m", "sensor_entity": "sensor.x"}
        call_obj = ServiceCallData(_build_data_call(data))
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = ""
        setattr(processor, method, AsyncMock(return_value=request_obj))

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            try:
                await handlers[service](_build_data_call(data))
            except Exception:
                pass  # later steps (e.g. data_analyzer sensor handling) are irrelevant

        getattr(processor, method).assert_awaited()
        # Attributes were never assigned: Mock returns child mocks, not real values
        assert not isinstance(processor.jpeg_options, dict)
        assert not isinstance(processor.ffmpeg_jpeg_q, int)
