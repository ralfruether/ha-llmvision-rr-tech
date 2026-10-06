"""Integration tests for identify_persons on the pro analyzers (mocked face service)."""
import asyncio
import base64
import io
import os

import pytest
from unittest.mock import AsyncMock, Mock, patch
from PIL import Image

from custom_components.llmvision import ServiceCallData, face_client
from custom_components.llmvision.face_client import (
    FacePerson,
    FaceResult,
    FaceSample,
    FaceSettings,
)
from custom_components.llmvision.media_handlers import MediaProcessor

SETTINGS = FaceSettings(url="http://192.168.9.230:8770", token="t" * 32)
BOX = (0.4, 0.4, 0.6, 0.6)
SIZE = (200, 100)
UNDECODABLE = b"\xff\xd8garbage\xff\xd9"
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def _jpeg(level, size=SIZE):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(level, level, level)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _green_pixels(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return sum(1 for r, g, b in img.getdata() if g > 150 and r < 100 and b < 100)


def _ok_result(t=2.5, camera="haustuer", has_unknown=False):
    return FaceResult(
        camera=camera,
        status="ok",
        persons=[FacePerson("lea", 0.75634, (FaceSample(t, BOX),))],
        has_faces=True,
        has_unknown=has_unknown,
        elapsed_ms=42,
    )


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
        proc = MediaProcessor(hass, client)
    proc._expose_image = AsyncMock()
    proc._write_snapshot = AsyncMock()
    return proc


def _enable(processor):
    processor.identify_persons = True
    processor.face_settings = SETTINGS


def _model_images(processor):
    return [c.kwargs["base64_image"] for c in processor.client.add_frame.call_args_list]


class FakeStream:
    def __init__(self, chunks=()):
        self._chunks = list(chunks)

    async def read(self, _n=-1):
        return self._chunks.pop(0) if self._chunks else b""


class FakeProcess:
    pid = 4242

    def __init__(self, stdout, returncode=0):
        self.stdout = stdout
        self.stderr = FakeStream()
        self.returncode = None
        self._final = returncode
        self.terminate = Mock(side_effect=self._finish)
        self.kill = Mock(side_effect=self._finish)

    def _finish(self):
        if self.returncode is None:
            self.returncode = self._final

    async def wait(self):
        self._finish()
        return self.returncode


# ------------------------------------------------------------------ drawing


class TestAnnotateImage:
    @pytest.mark.asyncio
    async def test_green_head_box_and_name_after_polylines(self, processor):
        b64, (w, h), resolved, faces = await processor._annotate_image(
            _jpeg(0, (400, 200)),
            400,
            polylines=[[(0.0, 0.9), (1.0, 0.9)]],
            face_labels=[("lea", BOX)],
        )
        assert (w, h) == (400, 200)
        assert resolved == [[(0, 180), (400, 180)]]
        assert faces == [{"name": "lea", "box": [136, 68, 264, 132]}]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        r, g, b = img.getpixel((137, 100))
        assert g > 150 and r < 100 and b < 100
        r, g, b = img.getpixel((200, 180))
        assert r > 150 and g < 100 and b < 100
        # Name tag sits above the box
        tag = [img.getpixel((x, 60)) for x in range(136, 160)]
        assert any(g > 150 and r < 100 for r, g, b in tag)
        # Inside the head box the frame stays untouched
        r, g, b = img.getpixel((200, 100))
        assert max(r, g, b) < 40

    @pytest.mark.asyncio
    async def test_box_at_edge_is_clamped(self, processor):
        _, _, _, faces = await processor._annotate_image(
            _jpeg(0, (400, 200)), 400, None, [("lea", (0.0, 0.0, 0.1, 0.1))]
        )
        assert faces == [{"name": "lea", "box": [0, 0, 52, 26]}]

    @pytest.mark.asyncio
    async def test_wrapper_keeps_polyline_contract(self, processor):
        result = await processor._draw_polylines_on_image(
            _jpeg(0), 200, [[(0.0, 0.5), (1.0, 0.5)]]
        )
        assert len(result) == 3
        assert _green_pixels(result[0]) == 0


# ------------------------------------------------------------------ stream mode


class TestStreamMode:
    def _patches(self, data):
        proc = FakeProcess(FakeStream([data]))
        return (
            patch(
                "custom_components.llmvision.media_handlers._async_get_stream_source",
                AsyncMock(return_value="rtsp://cam/stream"),
            ),
            patch(
                "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
        )

    @pytest.mark.asyncio
    async def test_raw_index_counts_undecodable_frames(self, processor):
        data = _jpeg(40) + UNDECODABLE + _jpeg(90) + _jpeg(140)
        p1, p2 = self._patches(data)
        with p1, p2:
            first, frames = await processor._capture_stream_camera(
                image_entity="camera.haustuer", camera_number=0, duration=5,
                frame_rate="1", frame_cap=16, target_width=2048,
                include_filename=True,
            )
        assert first[0] == "haustuer-frame-0"
        assert list(frames) == ["haustuer-frame-1", "haustuer-frame-2"]
        assert processor._face_stream_frames == {
            "haustuer-frame-0": ("camera.haustuer", 0),
            "haustuer-frame-1": ("camera.haustuer", 2),
            "haustuer-frame-2": ("camera.haustuer", 3),
        }

    @pytest.mark.asyncio
    async def test_raw_index_counts_frames_dropped_by_splitter(self, processor):
        from custom_components.llmvision.stream_capture import JpegStreamSplitter

        oversized = b"\xff\xd8" + b"\x00" * 5000
        chunks = [_jpeg(40), oversized, _jpeg(90) + _jpeg(140)]
        proc = FakeProcess(FakeStream(chunks))
        with (
            patch(
                "custom_components.llmvision.media_handlers._async_get_stream_source",
                AsyncMock(return_value="rtsp://cam/stream"),
            ),
            patch(
                "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
            patch(
                "custom_components.llmvision.media_handlers.JpegStreamSplitter",
                lambda: JpegStreamSplitter(max_frame_bytes=3000),
            ),
        ):
            await processor._capture_stream_camera(
                image_entity="camera.haustuer", camera_number=0, duration=5,
                frame_rate="1", frame_cap=16, target_width=2048,
                include_filename=True,
            )
        assert [raw for _, raw in processor._face_stream_frames.values()] == [0, 2, 3]

    async def _record(self, processor, tmp_path, clip_exists=True, clip_plans=True):
        clip = tmp_path / "haustuer.mp4"
        if clip_exists:
            clip.write_bytes(MP4)
        data = _jpeg(40) + UNDECODABLE + _jpeg(90) + _jpeg(140)
        p1, p2 = self._patches(data)
        processor._select_keyframe_index = AsyncMock(return_value=1)
        with p1, p2:
            await processor.record(
                image_entities=["camera.haustuer"], duration=5, max_frames=None,
                target_width=2048, include_filename=True, expose_images=True,
                fps=1, debug_polylines=True, frame_source="stream",
                clip_plans=(
                    {"camera.haustuer": (str(clip), None, 1920)} if clip_plans else None
                ),
            )
        return str(clip)

    @pytest.mark.asyncio
    async def test_labels_on_matching_frame_only_key_frame_clean(
        self, processor, tmp_path
    ):
        _enable(processor)
        identify = AsyncMock(return_value=_ok_result(t=2.5))
        with patch.object(face_client, "async_identify", identify):
            clip = await self._record(processor, tmp_path)

        identify.assert_awaited_once()
        assert identify.await_args.args[2] == "haustuer"
        assert identify.await_args.kwargs == {"clip_path": clip}
        # raw indices 0, 2, 3 at 1 fps show t = 0.5, 2.5, 3.5; the sample at 2.5 labels frame 1
        assert [_green_pixels(img) > 20 for img in _model_images(processor)] == [
            False, True, False,
        ]
        exposed = processor._expose_image.await_args.kwargs["image_data"]
        assert _green_pixels(exposed) == 0
        assert processor.face_labeled_names == {"lea"}
        assert [r.status for r in processor.face_results] == ["ok"]
        assert processor.debug_info == [
            {
                "frame": "haustuer-frame-1",
                "width": 200,
                "height": 100,
                "polylines": [],
                "faces": [{"name": "lea", "box": [68, 34, 132, 66]}],
            }
        ]
        written = processor._write_snapshot.await_args_list
        assert len(written) == 1
        assert written[0].kwargs["filename"].startswith("debug-haustuer-frame-1-")

    @pytest.mark.asyncio
    async def test_storage_path_never_gets_labels(self, processor, tmp_path):
        _enable(processor)
        processor._resolve_storage_path = Mock(return_value=str(tmp_path / "store"))
        with patch.object(
            face_client, "async_identify", AsyncMock(return_value=_ok_result(t=2.0))
        ):
            clip = tmp_path / "haustuer.mp4"
            clip.write_bytes(MP4)
            data = _jpeg(40) + UNDECODABLE + _jpeg(90) + _jpeg(140)
            p1, p2 = self._patches(data)
            with p1, p2:
                await processor.record(
                    image_entities=["camera.haustuer"], duration=5, max_frames=None,
                    target_width=2048, include_filename=True, expose_images=False,
                    fps=1, storage_path="store", frame_source="stream",
                    clip_plans={"camera.haustuer": (str(clip), None, 1920)},
                )
        stored = [
            c.kwargs["image_data"]
            for c in processor._write_snapshot.await_args_list
            if c.kwargs["directory"] == str(tmp_path / "store")
        ]
        assert len(stored) == 3
        assert all(_green_pixels(img) == 0 for img in stored)

    @pytest.mark.asyncio
    async def test_removed_clip_is_no_clip(self, processor, tmp_path, caplog):
        _enable(processor)
        identify = AsyncMock()
        with patch.object(face_client, "async_identify", identify):
            await self._record(processor, tmp_path, clip_exists=False)
        identify.assert_not_awaited()
        assert [r.status for r in processor.face_results] == ["error:no_clip"]
        assert all(_green_pixels(img) == 0 for img in _model_images(processor))
        assert "error:no_clip" in caplog.text

    @pytest.mark.asyncio
    async def test_without_clip_path_is_no_clip(self, processor, tmp_path):
        _enable(processor)
        identify = AsyncMock()
        with patch.object(face_client, "async_identify", identify):
            await self._record(processor, tmp_path, clip_plans=False)
        identify.assert_not_awaited()
        assert [r.status for r in processor.face_results] == ["error:no_clip"]
        # Identification already ran: the snapshot hook must not run it again
        await processor.identify_recorded_clips()
        assert len(processor.face_results) == 1

    @pytest.mark.asyncio
    async def test_option_off_no_request(self, processor, tmp_path):
        processor.face_settings = SETTINGS
        identify = AsyncMock()
        with patch.object(face_client, "async_identify", identify):
            await self._record(processor, tmp_path)
        identify.assert_not_awaited()
        assert processor.face_results == []
        assert all(_green_pixels(img) == 0 for img in _model_images(processor))
        assert all("faces" not in entry for entry in processor.debug_info or [])


class TestSnapshotMode:
    @pytest.mark.asyncio
    async def test_recorded_clips_identified_facts_only(self, processor, tmp_path):
        _enable(processor)
        clip = str(tmp_path / "a.mp4")
        processor._face_clip_plan = {"camera.a": clip, "camera.b": None}
        processor.clip_paths = [clip]
        identify = AsyncMock(return_value=_ok_result(camera="a"))
        with patch.object(face_client, "async_identify", identify):
            await processor.identify_recorded_clips()
            await processor.identify_recorded_clips()
        identify.assert_awaited_once()
        assert identify.await_args.kwargs == {"clip_path": clip}
        assert [r.status for r in processor.face_results] == ["ok", "error:no_clip"]
        assert processor.face_labeled_names == set()
        processor.client.add_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_clip_read_under_lock_survives_overwrite(self, processor, tmp_path):
        _enable(processor)
        clip = tmp_path / "a.mp4"
        clip.write_bytes(MP4 + b"first")
        processor._face_clip_plan = {"camera.a": str(clip)}
        processor.clip_paths = [str(clip)]
        await processor.load_recorded_clips()
        # A follow-up capture truncates the same clip_path after the lock release
        clip.write_bytes(b"")
        identify = AsyncMock(return_value=_ok_result(camera="a"))
        with patch.object(face_client, "async_identify", identify):
            await processor.identify_recorded_clips()
        assert identify.await_args.kwargs == {"clip_data": MP4 + b"first"}
        assert processor._face_clip_data is None

    @pytest.mark.asyncio
    async def test_preload_failure_maps_to_reason(self, processor, tmp_path):
        _enable(processor)
        clip = tmp_path / "a.mp4"
        clip.write_bytes(b"not a video")
        processor._face_clip_plan = {"camera.a": str(clip)}
        processor.clip_paths = [str(clip)]
        await processor.load_recorded_clips()
        identify = AsyncMock()
        with patch.object(face_client, "async_identify", identify):
            await processor.identify_recorded_clips()
        identify.assert_not_awaited()
        assert [r.status for r in processor.face_results] == ["error:no_clip"]

    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self, processor):
        processor.identify_persons = True
        processor._face_clip_plan = {"camera.a": "/media/a.mp4"}
        identify = AsyncMock()
        with patch.object(face_client, "async_identify", identify):
            await processor.identify_recorded_clips()
        identify.assert_not_awaited()
        assert processor.face_results == []

    @pytest.mark.asyncio
    async def test_add_streams_records_clip_plan(self, processor):
        processor.record = AsyncMock()
        processor.record_clip = AsyncMock(return_value=[])
        await processor.add_streams(
            image_entities=["camera.a", "camera.b"], duration=5, max_frames=3,
            target_width=1280, include_filename=False, expose_images=False,
            clip_path="clips/x.mp4",
        )
        await processor.clip_task
        assert list(processor._face_clip_plan) == ["camera.a", "camera.b"]
        assert processor._face_clip_plan["camera.a"].endswith("x-a.mp4")


# ------------------------------------------------------------------ video mode


class TestVideoMode:
    async def _add_video(self, processor, path, fps=1, identify=None, frames=None):
        data = frames or (_jpeg(40) + UNDECODABLE + _jpeg(90) + _jpeg(140))
        proc = FakeProcess(FakeStream([data]))
        processor._select_keyframe_index = AsyncMock(return_value=1)
        processor.debug_info = []
        identify = identify or AsyncMock(return_value=_ok_result(t=2.5))
        with (
            patch.object(face_client, "async_identify", identify),
            patch(
                "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
        ):
            await processor.add_video(
                video_path=path, base_url="http://ha", max_frames=None,
                target_width=2048, include_filename=False, expose_images=True,
                fps=fps, debug_polylines=True,
            )
        return identify

    def _clip(self, tmp_path):
        folder = tmp_path / "haustuer_20260926T180111_936970"
        folder.mkdir()
        clip = folder / "clip.mp4"
        clip.write_bytes(MP4 + b"data")
        return str(clip)

    @pytest.mark.asyncio
    async def test_fps_mode_maps_raw_frame_counter(self, processor, tmp_path):
        _enable(processor)
        identify = await self._add_video(processor, self._clip(tmp_path))
        identify.assert_awaited_once()
        assert identify.await_args.args[2] == "haustuer"
        assert identify.await_args.kwargs == {"clip_data": MP4 + b"data"}
        # fps output n = 0, (1 undecodable), 2, 3 shows t = (n + 0.5) / fps
        assert [_green_pixels(img) > 20 for img in _model_images(processor)] == [
            False, True, False,
        ]
        exposed = processor._expose_image.await_args.kwargs["image_data"]
        assert _green_pixels(exposed) == 0
        assert [entry["frame"] for entry in processor.debug_info] == ["clip frame 2"]
        assert processor.debug_info[0]["faces"][0]["name"] == "lea"
        assert processor.face_labeled_names == {"lea"}

    @pytest.mark.asyncio
    async def test_fractional_fps(self, processor, tmp_path):
        _enable(processor)
        # 2 fps: raw 0, 2, 3 -> frames show t = 0.25, 1.25, 1.75 (ffmpeg fps filter keeps
        # the last frame of each slot); a sample at 1.9 labels raw 3 only
        identify = AsyncMock(return_value=_ok_result(t=1.9))
        await self._add_video(processor, self._clip(tmp_path), fps=2, identify=identify)
        assert [_green_pixels(img) > 20 for img in _model_images(processor)] == [
            False, False, True,
        ]

    @pytest.mark.asyncio
    async def test_iframe_mode_facts_only(self, processor, tmp_path):
        _enable(processor)
        identify = await self._add_video(processor, self._clip(tmp_path), fps=None)
        identify.assert_awaited_once()
        assert all(_green_pixels(img) == 0 for img in _model_images(processor))
        assert processor.face_labeled_names == set()
        assert [r.status for r in processor.face_results] == ["ok"]
        text = face_client.build_fact_text(processor.face_results, processor.face_labeled_names)
        assert '"lea"' in text and "green box" not in text

    @pytest.mark.asyncio
    async def test_downloaded_clip_read_before_temp_file_removed(self, processor):
        _enable(processor)
        temp_paths = []

        async def fake_fetch(url, target_file=None, **kwargs):
            temp_paths.append(target_file)
            with open(target_file, "wb") as handle:
                handle.write(MP4 + b"downloaded")

        processor._fetch = AsyncMock(side_effect=fake_fetch)
        identify = await self._add_video(
            processor, "http://frigate.local/api/events/xyz/clip.mp4"
        )
        assert identify.await_args.kwargs == {"clip_data": MP4 + b"downloaded"}
        assert identify.await_args.args[2] == "xyz"
        assert not os.path.exists(temp_paths[0])

    @pytest.mark.asyncio
    async def test_error_keeps_frames_unlabeled(self, processor, tmp_path, caplog):
        _enable(processor)
        failure = FaceResult(camera="haustuer", status="error:http_503")
        await self._add_video(
            processor, self._clip(tmp_path), identify=AsyncMock(return_value=failure)
        )
        assert all(_green_pixels(img) == 0 for img in _model_images(processor))
        assert len(_model_images(processor)) == 3
        assert face_client.build_fact_text(processor.face_results, processor.face_labeled_names) == ""

    @pytest.mark.asyncio
    async def test_missing_local_file_is_no_clip(self, processor, tmp_path):
        _enable(processor)
        identify = AsyncMock()
        await self._add_video(processor, str(tmp_path / "gone.mp4"), identify=identify)
        identify.assert_not_awaited()
        assert [r.status for r in processor.face_results] == ["error:no_clip"]

    @pytest.mark.asyncio
    async def test_option_off_no_request(self, processor, tmp_path):
        processor.face_settings = SETTINGS
        identify = AsyncMock()
        await self._add_video(processor, self._clip(tmp_path), identify=identify)
        identify.assert_not_awaited()
        assert processor.face_results == []
        assert all(_green_pixels(img) == 0 for img in _model_images(processor))

    @pytest.mark.asyncio
    async def test_face_task_cancelled_when_video_fails(self, processor, tmp_path):
        _enable(processor)
        tasks = []
        original = processor._start_face_task

        def spy(*args):
            task = original(*args)
            tasks.append(task)
            return task

        processor._start_face_task = spy

        async def slow_identify(*args, **kwargs):
            await asyncio.Event().wait()

        proc = FakeProcess(FakeStream([b""]), returncode=1)
        with (
            patch.object(face_client, "async_identify", slow_identify),
            patch(
                "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
        ):
            with pytest.raises(Exception):
                await processor.add_video(
                    video_path=self._clip(tmp_path), base_url="http://ha",
                    max_frames=None, target_width=2048, fps=1,
                )
        await asyncio.sleep(0)
        assert len(tasks) == 1 and tasks[0].cancelled()


# ------------------------------------------------------------------ handlers


def _build_data_call(data):
    data_call = Mock()
    data_call.data = Mock()
    data_call.data.get = Mock(side_effect=lambda key, default=None: data.get(key, default))
    return data_call


def _handlers():
    from custom_components.llmvision import setup

    hass = Mock()
    hass.data = {}
    hass.services = Mock()
    hass.services.register = Mock()
    hass.http = Mock()
    assert setup(hass, {}) is True
    return {c.args[1]: c.args[2] for c in hass.services.register.call_args_list}


def _mock_processor(results, labeled=False, method="add_videos"):
    processor = Mock()
    processor.key_frame = ""
    processor.debug_info = None
    processor.clip_paths = []
    processor.clip_task = None
    processor.face_results = []
    processor.face_labeled_names = set()
    request_obj = Mock()
    messages = []

    async def capture(call):
        messages.append(call.message)
        return {"response_text": "ok"}

    request_obj.call = AsyncMock(side_effect=capture)

    async def process(**kwargs):
        if method == "add_videos":
            processor.face_results = list(results)
            processor.face_labeled_names = {"lea"} if labeled else set()
        return request_obj

    setattr(processor, method, AsyncMock(side_effect=process))

    async def identify_recorded():
        processor.face_results = list(results)

    processor.identify_recorded_clips = AsyncMock(side_effect=identify_recorded)
    processor.load_recorded_clips = AsyncMock()
    return processor, request_obj, messages


async def _run(service, data, processor, request_obj, settings=SETTINGS):
    handlers = _handlers()
    call_obj = ServiceCallData(_build_data_call(data))
    memory = Mock()
    memory._update_memory = AsyncMock()
    get_settings = Mock(return_value=settings)
    with (
        patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
        patch("custom_components.llmvision.Request", return_value=request_obj),
        patch("custom_components.llmvision.MediaProcessor", return_value=processor),
        patch("custom_components.llmvision.Memory", return_value=memory),
        patch("custom_components.llmvision._create_event", new=AsyncMock()),
        patch.object(face_client, "get_face_settings", get_settings),
    ):
        result = await handlers[service](_build_data_call(data))
    return result, get_settings


BASE = {"provider": "e", "message": "m"}
VIDEO_PREFIX = "The attached images are frames from a video. m"


class TestHandlers:
    @pytest.mark.asyncio
    async def test_video_identified_persons(self):
        processor, request_obj, messages = _mock_processor(
            [_ok_result(has_unknown=True)], labeled=True
        )
        result, _ = await _run(
            "video_analyzer_pro", {**BASE, "identify_persons": True}, processor, request_obj
        )
        assert processor.identify_persons is True
        assert processor.face_settings is SETTINGS
        assert messages[0].startswith(VIDEO_PREFIX + "\n\n")
        assert 'marked in the images with a green box and their name: "lea".' in messages[0]
        assert "At least one other face could not be identified" in messages[0]
        assert result["persons"] == [{"name": "lea", "score": 0.756}]
        assert result["face_service"] == "ok"
        assert result["face_service_ms"] == 42
        assert "identify_id" not in messages[0]

    @pytest.mark.asyncio
    async def test_video_appearance_reported_but_not_prompted(self, caplog):
        result_obj = _ok_result(has_unknown=True)
        result_obj.appearance = face_client.AppearanceInfo(
            status="ok",
            matches=(
                face_client.AppearanceMatch("ralf", 0.93, "strong", True, 0.61, 300.0, False),
            ),
            person_tracks=2,
            unnamed_tracks=0,
        )
        processor, request_obj, messages = _mock_processor([result_obj], labeled=True)
        result, _ = await _run(
            "video_analyzer_pro", {**BASE, "identify_persons": True}, processor, request_obj
        )
        assert "ralf" not in messages[0] and '"lea"' in messages[0]
        assert result["persons"] == [{"name": "lea", "score": 0.756}]
        assert result["appearance_persons"] == [
            {"name": "ralf", "score": 0.93, "tier": "strong", "qualifies": True,
             "seed_face_score": 0.61, "seed_age_s": 300, "same_camera": False}
        ]
        assert result["person_tracks"] == 2 and result["unnamed_tracks"] == 0
        assert result["reid"] == "ok"
        assert "ralf" not in caplog.text

    @pytest.mark.asyncio
    async def test_video_without_reid_data_reports_null_counts(self):
        processor, request_obj, _ = _mock_processor([_ok_result()])
        result, _ = await _run(
            "video_analyzer_pro", {**BASE, "identify_persons": True}, processor, request_obj
        )
        assert result["appearance_persons"] == [] and result["reid"] == "unavailable"
        assert result["person_tracks"] is None and result["unnamed_tracks"] is None

    @pytest.mark.asyncio
    async def test_video_error_leaves_message_unchanged(self):
        failure = FaceResult(camera="c", status="error:http_503", elapsed_ms=5)
        processor, request_obj, messages = _mock_processor([failure])
        result, _ = await _run(
            "video_analyzer_pro", {**BASE, "identify_persons": True}, processor, request_obj
        )
        assert messages == [VIDEO_PREFIX]
        assert result["face_service"] == "error:http_503"
        assert result["persons"] == []
        assert result["response_text"] == "ok"

    @pytest.mark.asyncio
    async def test_video_disabled(self):
        processor, request_obj, messages = _mock_processor([])
        result, _ = await _run(
            "video_analyzer_pro",
            {**BASE, "identify_persons": True},
            processor,
            request_obj,
            settings=None,
        )
        assert messages == [VIDEO_PREFIX]
        assert result["face_service"] == "disabled"
        assert result["reid"] == "disabled" and result["unnamed_tracks"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [None, False, "true"])
    async def test_option_off_unchanged_response(self, value):
        processor, request_obj, messages = _mock_processor([_ok_result()])
        data = dict(BASE) if value is None else {**BASE, "identify_persons": value}
        result, get_settings = await _run(
            "video_analyzer_pro", data, processor, request_obj
        )
        get_settings.assert_not_called()
        assert messages == [VIDEO_PREFIX]
        assert result == {"response_text": "ok"}

    @pytest.mark.asyncio
    async def test_stream_identifies_recorded_clips_after_capture(self):
        processor, request_obj, messages = _mock_processor(
            [_ok_result()], method="add_streams"
        )
        order = []
        processor.add_streams.side_effect = None
        processor.add_streams.return_value = request_obj

        async def add_streams(**kwargs):
            order.append("capture")
            return request_obj

        async def identify():
            order.append("identify")
            processor.face_results = [_ok_result()]

        async def load():
            order.append("load")

        processor.add_streams = AsyncMock(side_effect=add_streams)
        processor.load_recorded_clips = AsyncMock(side_effect=load)
        processor.identify_recorded_clips = AsyncMock(side_effect=identify)
        result, _ = await _run(
            "stream_analyzer_pro",
            {**BASE, "identify_persons": True, "image_entity": ["camera.a"]},
            processor,
            request_obj,
        )
        assert order == ["capture", "load", "identify"]
        assert 'in this clip: "lea"' in messages[0]
        assert result["face_service"] == "ok"
        assert result["persons"] == [{"name": "lea", "score": 0.756}]

    @pytest.mark.asyncio
    async def test_stream_option_off(self):
        processor, request_obj, messages = _mock_processor([], method="add_streams")
        result, get_settings = await _run(
            "stream_analyzer_pro", dict(BASE), processor, request_obj
        )
        processor.identify_recorded_clips.assert_not_awaited()
        processor.load_recorded_clips.assert_not_awaited()
        get_settings.assert_not_called()
        assert "persons" not in result and "face_service" not in result

    @pytest.mark.asyncio
    async def test_upstream_services_ignore_option(self):
        processor, request_obj, messages = _mock_processor([_ok_result()])
        processor.add_videos.side_effect = None
        processor.add_videos.return_value = request_obj
        result, get_settings = await _run(
            "video_analyzer", {**BASE, "identify_persons": True}, processor, request_obj
        )
        get_settings.assert_not_called()
        assert "face_service" not in result
