"""Unit tests for the stream_analyzer_pro feature."""
import asyncio
import base64
import io
import os

import pytest
from unittest.mock import AsyncMock, Mock, patch
from PIL import Image

from homeassistant.exceptions import ServiceValidationError

from custom_components.llmvision.media_handlers import MediaProcessor
from custom_components.llmvision.providers import OpenAI
from custom_components.llmvision.const import CONF_REASONING_EFFORT, DOMAIN
from custom_components.llmvision import ServiceCallData


def _jpeg_bytes(color, size=(100, 50)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _run_executor(_executor, func, *args):
    """Execute the submitted callable synchronously (test double)."""
    return func(*args)


@pytest.fixture
def processor():
    hass = Mock()
    hass.loop = Mock()
    hass.loop.run_in_executor = AsyncMock(side_effect=_run_executor)
    hass.states = Mock()
    hass.config = Mock()
    hass.config.path = Mock(return_value="/mock/path")
    hass.config.config_dir = "/config"
    hass.async_create_task = Mock(
        side_effect=lambda coro, **kwargs: asyncio.create_task(coro)
    )
    client = Mock()
    client.add_frame = Mock()
    with patch(
        "custom_components.llmvision.media_handlers.async_get_clientsession"
    ):
        return MediaProcessor(hass, client)


class TestComputeInterval:
    def test_fps_sets_interval(self, processor):
        assert processor._compute_interval(10, fps=2) == 0.5
        assert processor._compute_interval(10, fps=0.5) == 2.0

    def test_fps_omitted_uses_legacy_cadence(self, processor):
        assert processor._compute_interval(2) == 1
        assert processor._compute_interval(5) == 2
        assert processor._compute_interval(20) == 3
        assert processor._compute_interval(40) == 5

    def test_fps_zero_falls_back_to_legacy(self, processor):
        assert processor._compute_interval(5, fps=0) == 2


class TestValidatePolylines:
    def test_none_returns_none(self, processor):
        assert processor._validate_polylines(None) is None

    def test_empty_returns_none(self, processor):
        assert processor._validate_polylines([]) is None

    def test_valid_polyline_normalized(self, processor):
        result = processor._validate_polylines([[[0.1, 0.5], [0.9, 0.5]]])
        assert result == [[(0.1, 0.5), (0.9, 0.5)]]

    def test_too_few_points_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._validate_polylines([[[0.1, 0.5]]])

    def test_out_of_range_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._validate_polylines([[[1.5, 0.5], [0.2, 0.2]]])

    def test_non_numeric_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._validate_polylines([[["a", 0.5], [0.2, 0.2]]])

    def test_bool_coordinate_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._validate_polylines([[[True, 0.5], [0.2, 0.2]]])

    def test_not_a_list_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._validate_polylines("not-a-list")


class TestResolveStoragePath:
    def test_empty_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._resolve_storage_path("")

    def test_relative_is_confined_to_media(self, processor):
        resolved = processor._resolve_storage_path("recordings")
        assert resolved == os.path.realpath("/media/llmvision/recordings")

    def test_traversal_escape_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._resolve_storage_path("../../etc")

    def test_absolute_outside_roots_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._resolve_storage_path("/etc/passwd")

    def test_absolute_inside_config_dir_allowed(self, processor, tmp_path):
        processor.hass.config.config_dir = str(tmp_path)
        target = str(tmp_path / "recordings")
        assert processor._resolve_storage_path(target) == os.path.realpath(target)


class TestDrawPolylines:
    @pytest.mark.asyncio
    async def test_polyline_drawn_in_red(self, processor):
        image_data = _jpeg_bytes((0, 0, 0))
        b64, (w, h), resolved = await processor._draw_polylines_on_image(
            image_data=image_data,
            target_width=100,
            polylines=[[(0.0, 0.5), (1.0, 0.5)]],
        )
        assert (w, h) == (100, 50)
        assert resolved == [[(0, 25), (100, 25)]]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        r, g, b = img.getpixel((50, 25))[:3]
        assert r > 150 and g < 100 and b < 100

    @pytest.mark.asyncio
    async def test_keyframe_resize_stays_clean(self, processor):
        image_data = _jpeg_bytes((0, 0, 0))
        b64 = await processor.resize_image(target_width=100, image_data=image_data)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        r, g, b = img.getpixel((50, 25))[:3]
        assert r < 60 and g < 60 and b < 60


class TestSelectStreamFrames:
    def _scored(self, count):
        return [
            (f"camera0-frame-{i}", f"d{i}".encode(), 0.01 * i, 0, i)
            for i in range(count)
        ]

    def test_unbounded_keeps_all_frames(self, processor):
        first = {"camera.a": ("camera0-frame-0", b"first")}
        selected = processor._select_stream_frames(
            ["camera.a"], first, self._scored(20), None
        )
        assert len(selected) == 21

    def test_max_frames_caps_selection(self, processor):
        first = {"camera.a": ("camera0-frame-0", b"first")}
        selected = processor._select_stream_frames(
            ["camera.a"], first, self._scored(20), 5
        )
        assert len(selected) == 5


class TestWriteSnapshot:
    @pytest.mark.asyncio
    async def test_writes_file(self, processor, tmp_path):
        data = _jpeg_bytes((10, 20, 30))
        path = await processor._write_snapshot(str(tmp_path), "frame.jpg", data)
        assert os.path.exists(path)
        with open(path, "rb") as handle:
            assert handle.read() == data

    @pytest.mark.asyncio
    async def test_write_failure_raises(self, processor, tmp_path):
        with patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(ServiceValidationError):
                await processor._write_snapshot(str(tmp_path), "frame.jpg", b"x")


class TestReasoningEffortOverride:
    def test_per_call_override(self, mock_hass):
        mock_hass.data = {
            DOMAIN: {"p": {CONF_REASONING_EFFORT: "low"}}
        }
        call = Mock()
        call.provider = "p"
        call.reasoning_effort = "high"
        call.model_is_glimpse = Mock(return_value=False)
        with patch("custom_components.llmvision.providers.async_get_clientsession"):
            provider = OpenAI(mock_hass, "k", "gpt-5-mini")
            params = provider._get_default_parameters(call)
        assert params["reasoning_effort"] == "high"

    def test_no_override_uses_config(self, mock_hass):
        mock_hass.data = {
            DOMAIN: {"p": {CONF_REASONING_EFFORT: "low"}}
        }
        call = Mock()
        call.provider = "p"
        call.reasoning_effort = None
        call.model_is_glimpse = Mock(return_value=False)
        with patch("custom_components.llmvision.providers.async_get_clientsession"):
            provider = OpenAI(mock_hass, "k", "gpt-5-mini")
            params = provider._get_default_parameters(call)
        assert params["reasoning_effort"] == "low"

    def test_gpt5_mini_clamps_high_to_medium(self, mock_hass):
        with patch("custom_components.llmvision.providers.async_get_clientsession"):
            provider = OpenAI(mock_hass, "k", "gpt-5-mini")
            assert provider._model_supports_thinking("high") == "medium"


def _build_data_call(data):
    data_call = Mock()
    data_call.data = Mock()
    data_call.data.get = Mock(side_effect=lambda key, default=None: data.get(key, default))
    return data_call


def _make_hass():
    hass = Mock()
    hass.data = {}
    hass.config = Mock()
    hass.config.path = Mock(return_value="/tmp")
    hass.services = Mock()
    hass.services.register = Mock()
    hass.http = Mock()
    hass.http.register_view = Mock()
    return hass


class TestStreamAnalyzerProService:
    def _handlers(self, hass):
        from custom_components.llmvision import setup

        assert setup(hass, {}) is True
        handlers = {}
        for call in hass.services.register.call_args_list:
            _, service_name, handler = call.args[:3]
            handlers[service_name] = handler
        return handlers

    def test_service_registered(self):
        hass = _make_hass()
        handlers = self._handlers(hass)
        assert "stream_analyzer_pro" in handlers

    @pytest.mark.asyncio
    async def test_handler_returns_debug_and_key_frame(self):
        hass = _make_hass()
        handlers = self._handlers(hass)

        call_obj = ServiceCallData(
            _build_data_call({"provider": "e", "message": "m"})
        )
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = "frame.jpg"
        processor.debug_info = [{"frame": "f", "width": 10, "height": 10, "polylines": []}]
        processor.clip_paths = []
        processor.add_streams = AsyncMock(return_value=request_obj)

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            result = await handlers["stream_analyzer_pro"](
                _build_data_call({"provider": "e", "message": "m"})
            )

        assert result["key_frame"] == "frame.jpg"
        assert result["debug"] == processor.debug_info

    @pytest.mark.asyncio
    async def test_handler_omits_debug_when_none(self):
        hass = _make_hass()
        handlers = self._handlers(hass)

        call_obj = ServiceCallData(
            _build_data_call({"provider": "e", "message": "m"})
        )
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = ""
        processor.debug_info = None
        processor.clip_paths = []
        processor.add_streams = AsyncMock(return_value=request_obj)

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            result = await handlers["stream_analyzer_pro"](
                _build_data_call({"provider": "e", "message": "m"})
            )

        assert "debug" not in result

    @pytest.mark.asyncio
    async def test_same_camera_capture_is_serialized_but_provider_calls_overlap(self):
        hass = _make_hass()
        handlers = self._handlers(hass)
        motion_state = Mock()
        motion_state.state = "on"
        hass.states = Mock()
        hass.states.get = Mock(return_value=motion_state)

        first_capture_started = asyncio.Event()
        release_first_capture = asyncio.Event()
        first_provider_started = asyncio.Event()
        release_first_provider = asyncio.Event()
        second_capture_started = asyncio.Event()

        request_one = Mock()

        async def call_first_provider(_call):
            first_provider_started.set()
            await release_first_provider.wait()
            return {"response_text": "first"}

        request_one.call = AsyncMock(side_effect=call_first_provider)
        request_two = Mock()
        request_two.call = AsyncMock(return_value={"response_text": "second"})

        processor_one = Mock()
        processor_one.key_frame = ""
        processor_one.debug_info = None
        processor_one.clip_paths = []
        processor_one.clip_task = None

        async def capture_first(**_kwargs):
            first_capture_started.set()
            await release_first_capture.wait()
            return request_one

        processor_one.add_streams = AsyncMock(side_effect=capture_first)

        processor_two = Mock()
        processor_two.key_frame = ""
        processor_two.debug_info = None
        processor_two.clip_paths = []
        processor_two.clip_task = None

        async def capture_second(**_kwargs):
            second_capture_started.set()
            return request_two

        processor_two.add_streams = AsyncMock(side_effect=capture_second)

        call_one = ServiceCallData(
            _build_data_call(
                {
                    "provider": "e",
                    "message": "first",
                    "image_entity": ["camera.front"],
                    "coalesce_while_recording": True,
                    "motion_entity": ["binary_sensor.front_motion"],
                }
            )
        )
        call_two = ServiceCallData(
            _build_data_call(
                {
                    "provider": "e",
                    "message": "second",
                    "image_entity": ["camera.front"],
                    "coalesce_while_recording": True,
                    "motion_entity": ["binary_sensor.front_motion"],
                }
            )
        )
        memories = [Mock(), Mock()]
        for memory in memories:
            memory._update_memory = AsyncMock()

        with (
            patch(
                "custom_components.llmvision.ServiceCallData",
                side_effect=[call_one, call_two],
            ),
            patch(
                "custom_components.llmvision.MediaProcessor",
                side_effect=[processor_one, processor_two],
            ),
            patch(
                "custom_components.llmvision.Memory",
                side_effect=memories,
            ),
            patch("custom_components.llmvision.Request", side_effect=[request_one, request_two]),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            first_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await first_capture_started.wait()

            second_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await asyncio.sleep(0)
            assert not second_capture_started.is_set()

            release_first_capture.set()
            await first_provider_started.wait()
            await asyncio.wait_for(second_capture_started.wait(), timeout=0.1)
            assert not first_task.done()

            release_first_provider.set()
            first_result, second_result = await asyncio.gather(first_task, second_task)

        assert first_result["response_text"] == "first"
        assert second_result["response_text"] == "second"
        assert first_result["capture"]["status"] == "completed"
        assert second_result["capture"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_busy_calls_coalesce_and_skip_pending_capture_when_motion_ends(self):
        hass = _make_hass()
        handlers = self._handlers(hass)
        motion_state = Mock()
        motion_state.state = "on"
        hass.states = Mock()
        hass.states.get = Mock(return_value=motion_state)

        first_capture_started = asyncio.Event()
        release_first_capture = asyncio.Event()
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "first"})

        processor_one = Mock()
        processor_one.key_frame = ""
        processor_one.debug_info = None
        processor_one.clip_paths = []
        processor_one.clip_task = None

        async def capture_first(**_kwargs):
            first_capture_started.set()
            await release_first_capture.wait()
            return request_obj

        processor_one.add_streams = AsyncMock(side_effect=capture_first)

        processor_two = Mock()
        processor_two.clip_task = None
        processor_two.add_streams = AsyncMock()
        processor_three = Mock()
        processor_three.clip_task = None
        processor_three.add_streams = AsyncMock()

        call_data = {
            "provider": "e",
            "message": "m",
            "image_entity": ["camera.front"],
            "coalesce_while_recording": True,
            "motion_entity": ["binary_sensor.front_motion"],
        }
        calls = [
            ServiceCallData(_build_data_call(call_data)),
            ServiceCallData(_build_data_call(call_data)),
            ServiceCallData(_build_data_call(call_data)),
        ]
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()

        with (
            patch("custom_components.llmvision.ServiceCallData", side_effect=calls),
            patch(
                "custom_components.llmvision.MediaProcessor",
                side_effect=[processor_one, processor_two, processor_three],
            ),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            first_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await first_capture_started.wait()
            pending_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await asyncio.sleep(0)

            coalesced = await handlers["stream_analyzer_pro"](_build_data_call({}))
            assert coalesced["capture"]["status"] == "coalesced"
            assert coalesced["capture"]["reason"] == "pending_follow_up_exists"

            motion_state.state = "off"
            release_first_capture.set()
            first_result, pending_result = await asyncio.gather(
                first_task, pending_task
            )

        assert first_result["capture"]["status"] == "completed"
        assert pending_result["capture"]["status"] == "skipped"
        assert pending_result["capture"]["reason"] == "motion_inactive"
        assert pending_result["capture"]["request_id"]
        processor_two.add_streams.assert_not_awaited()
        processor_three.add_streams.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("image_entities", "motion_entities", "message"),
        [
            (
                ["camera.front", "camera.back"],
                ["binary_sensor.front_motion"],
                "exactly one camera",
            ),
            (["camera.front"], [], "motion_entity"),
            (["camera.front"], ["sensor.front_motion"], "motion_entity"),
        ],
    )
    async def test_invalid_coalescing_configuration_raises(
        self, image_entities, motion_entities, message
    ):
        hass = _make_hass()
        handlers = self._handlers(hass)
        call_obj = ServiceCallData(
            _build_data_call(
                {
                    "provider": "e",
                    "message": "m",
                    "image_entity": image_entities,
                    "coalesce_while_recording": True,
                    "motion_entity": motion_entities,
                }
            )
        )

        with patch(
            "custom_components.llmvision.ServiceCallData", return_value=call_obj
        ):
            with pytest.raises(ServiceValidationError, match=message):
                await handlers["stream_analyzer_pro"](_build_data_call({}))

    @pytest.mark.asyncio
    async def test_cancelling_pending_owner_releases_coalescing_slot(self):
        hass = _make_hass()
        handlers = self._handlers(hass)
        motion_state = Mock()
        motion_state.state = "on"
        hass.states = Mock()
        hass.states.get = Mock(return_value=motion_state)

        first_capture_started = asyncio.Event()
        release_first_capture = asyncio.Event()
        replacement_capture_started = asyncio.Event()
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})

        processor_one = Mock()
        processor_one.key_frame = ""
        processor_one.debug_info = None
        processor_one.clip_paths = []
        processor_one.clip_task = None

        async def capture_first(**_kwargs):
            first_capture_started.set()
            await release_first_capture.wait()
            return request_obj

        processor_one.add_streams = AsyncMock(side_effect=capture_first)

        processor_two = Mock()
        processor_two.clip_task = None
        processor_two.add_streams = AsyncMock()

        processor_three = Mock()
        processor_three.key_frame = ""
        processor_three.debug_info = None
        processor_three.clip_paths = []
        processor_three.clip_task = None

        async def capture_replacement(**_kwargs):
            replacement_capture_started.set()
            return request_obj

        processor_three.add_streams = AsyncMock(side_effect=capture_replacement)

        call_data = {
            "provider": "e",
            "message": "m",
            "image_entity": ["camera.front"],
            "coalesce_while_recording": True,
            "motion_entity": ["binary_sensor.front_motion"],
        }
        calls = [
            ServiceCallData(_build_data_call(call_data)),
            ServiceCallData(_build_data_call(call_data)),
            ServiceCallData(_build_data_call(call_data)),
        ]
        memories = [Mock(), Mock()]
        for memory in memories:
            memory._update_memory = AsyncMock()

        with (
            patch("custom_components.llmvision.ServiceCallData", side_effect=calls),
            patch(
                "custom_components.llmvision.MediaProcessor",
                side_effect=[processor_one, processor_two, processor_three],
            ),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.Memory", side_effect=memories),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            first_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await first_capture_started.wait()

            pending_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await asyncio.sleep(0)
            pending_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending_task

            replacement_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await asyncio.sleep(0)
            assert not replacement_task.done()

            release_first_capture.set()
            await asyncio.wait_for(replacement_capture_started.wait(), timeout=0.1)
            first_result, replacement_result = await asyncio.gather(
                first_task, replacement_task
            )

        assert first_result["capture"]["status"] == "completed"
        assert replacement_result["capture"]["status"] == "completed"
        processor_two.add_streams.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_different_camera_captures_can_overlap(self):
        hass = _make_hass()
        handlers = self._handlers(hass)

        first_capture_started = asyncio.Event()
        release_first_capture = asyncio.Event()
        second_capture_started = asyncio.Event()

        request_one = Mock()
        request_one.call = AsyncMock(return_value={"response_text": "first"})
        request_two = Mock()
        request_two.call = AsyncMock(return_value={"response_text": "second"})

        processor_one = Mock()
        processor_one.key_frame = ""
        processor_one.debug_info = None
        processor_one.clip_paths = []
        processor_one.clip_task = None

        async def capture_first(**_kwargs):
            first_capture_started.set()
            await release_first_capture.wait()
            return request_one

        processor_one.add_streams = AsyncMock(side_effect=capture_first)

        processor_two = Mock()
        processor_two.key_frame = ""
        processor_two.debug_info = None
        processor_two.clip_paths = []
        processor_two.clip_task = None

        async def capture_second(**_kwargs):
            second_capture_started.set()
            return request_two

        processor_two.add_streams = AsyncMock(side_effect=capture_second)

        calls = [
            ServiceCallData(
                _build_data_call(
                    {
                        "provider": "e",
                        "message": "first",
                        "image_entity": ["camera.front"],
                    }
                )
            ),
            ServiceCallData(
                _build_data_call(
                    {
                        "provider": "e",
                        "message": "second",
                        "image_entity": ["camera.back"],
                    }
                )
            ),
        ]
        memories = [Mock(), Mock()]
        for memory in memories:
            memory._update_memory = AsyncMock()

        with (
            patch("custom_components.llmvision.ServiceCallData", side_effect=calls),
            patch(
                "custom_components.llmvision.MediaProcessor",
                side_effect=[processor_one, processor_two],
            ),
            patch("custom_components.llmvision.Memory", side_effect=memories),
            patch("custom_components.llmvision.Request", side_effect=[request_one, request_two]),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            first_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await first_capture_started.wait()
            second_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )

            await asyncio.wait_for(second_capture_started.wait(), timeout=0.1)
            release_first_capture.set()
            await asyncio.gather(first_task, second_task)

    @pytest.mark.asyncio
    async def test_duplicate_camera_entities_are_captured_once(self):
        hass = _make_hass()
        handlers = self._handlers(hass)

        call_obj = ServiceCallData(
            _build_data_call(
                {
                    "provider": "e",
                    "message": "m",
                    "image_entity": ["camera.front", "camera.front"],
                }
            )
        )
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = ""
        processor.debug_info = None
        processor.clip_paths = []
        processor.clip_task = None
        processor.add_streams = AsyncMock(return_value=request_obj)

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            await handlers["stream_analyzer_pro"](_build_data_call({}))

        assert processor.add_streams.await_args.kwargs["image_entities"] == [
            "camera.front"
        ]

    @pytest.mark.asyncio
    async def test_clip_finalization_holds_same_camera_lock(self):
        hass = _make_hass()
        handlers = self._handlers(hass)

        clip_started = asyncio.Event()
        release_clip = asyncio.Event()
        second_capture_started = asyncio.Event()

        request_one = Mock()
        request_one.call = AsyncMock(return_value={"response_text": "first"})
        request_two = Mock()
        request_two.call = AsyncMock(return_value={"response_text": "second"})

        processor_one = Mock()
        processor_one.key_frame = ""
        processor_one.debug_info = None
        processor_one.clip_paths = []
        processor_one.clip_task = None

        async def finish_clip():
            clip_started.set()
            await release_clip.wait()

        async def capture_first(**_kwargs):
            processor_one.clip_task = asyncio.create_task(finish_clip())
            return request_one

        processor_one.add_streams = AsyncMock(side_effect=capture_first)

        processor_two = Mock()
        processor_two.key_frame = ""
        processor_two.debug_info = None
        processor_two.clip_paths = []
        processor_two.clip_task = None

        async def capture_second(**_kwargs):
            second_capture_started.set()
            return request_two

        processor_two.add_streams = AsyncMock(side_effect=capture_second)

        calls = [
            ServiceCallData(
                _build_data_call(
                    {
                        "provider": "e",
                        "message": "first",
                        "image_entity": ["camera.front"],
                    }
                )
            ),
            ServiceCallData(
                _build_data_call(
                    {
                        "provider": "e",
                        "message": "second",
                        "image_entity": ["camera.front"],
                    }
                )
            ),
        ]
        memories = [Mock(), Mock()]
        for memory in memories:
            memory._update_memory = AsyncMock()

        with (
            patch("custom_components.llmvision.ServiceCallData", side_effect=calls),
            patch(
                "custom_components.llmvision.MediaProcessor",
                side_effect=[processor_one, processor_two],
            ),
            patch("custom_components.llmvision.Memory", side_effect=memories),
            patch(
                "custom_components.llmvision.Request",
                side_effect=[request_one, request_two],
            ),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            first_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await clip_started.wait()
            second_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await asyncio.sleep(0)
            assert not second_capture_started.is_set()

            release_clip.set()
            await asyncio.wait_for(second_capture_started.wait(), timeout=0.1)
            await asyncio.gather(first_task, second_task)

    @pytest.mark.asyncio
    async def test_cancellation_stops_clip_before_releasing_lock(self):
        hass = _make_hass()
        handlers = self._handlers(hass)

        clip_started = asyncio.Event()
        clip_stopped = asyncio.Event()
        second_capture_started = asyncio.Event()

        request_one = Mock()
        request_one.call = AsyncMock(return_value={"response_text": "first"})
        request_two = Mock()
        request_two.call = AsyncMock(return_value={"response_text": "second"})

        processor_one = Mock()
        processor_one.key_frame = ""
        processor_one.debug_info = None
        processor_one.clip_paths = []
        processor_one.clip_task = None

        async def record_clip_until_cancelled():
            clip_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                clip_stopped.set()

        async def capture_first(**_kwargs):
            processor_one.clip_task = asyncio.create_task(
                record_clip_until_cancelled()
            )
            return request_one

        processor_one.add_streams = AsyncMock(side_effect=capture_first)

        processor_two = Mock()
        processor_two.key_frame = ""
        processor_two.debug_info = None
        processor_two.clip_paths = []
        processor_two.clip_task = None

        async def capture_second(**_kwargs):
            second_capture_started.set()
            return request_two

        processor_two.add_streams = AsyncMock(side_effect=capture_second)

        calls = [
            ServiceCallData(
                _build_data_call(
                    {
                        "provider": "e",
                        "message": "first",
                        "image_entity": ["camera.front"],
                    }
                )
            ),
            ServiceCallData(
                _build_data_call(
                    {
                        "provider": "e",
                        "message": "second",
                        "image_entity": ["camera.front"],
                    }
                )
            ),
        ]
        memories = [Mock(), Mock()]
        for memory in memories:
            memory._update_memory = AsyncMock()

        with (
            patch("custom_components.llmvision.ServiceCallData", side_effect=calls),
            patch(
                "custom_components.llmvision.MediaProcessor",
                side_effect=[processor_one, processor_two],
            ),
            patch("custom_components.llmvision.Memory", side_effect=memories),
            patch(
                "custom_components.llmvision.Request",
                side_effect=[request_one, request_two],
            ),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            first_task = asyncio.create_task(
                handlers["stream_analyzer_pro"](_build_data_call({}))
            )
            await clip_started.wait()
            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task
            assert clip_stopped.is_set()

            await handlers["stream_analyzer_pro"](_build_data_call({}))
            assert second_capture_started.is_set()


class TestAddVideosProWiring:
    @pytest.mark.asyncio
    async def test_validates_and_passes_pro_params(self, processor):
        processor.add_video = AsyncMock()
        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://x",
        ):
            await processor.add_videos(
                video_paths=["/config/a.mp4"],
                event_ids=None,
                max_frames=None,
                target_width=1280,
                include_filename=False,
                expose_images=False,
                fps=2,
                polylines=[[[0.1, 0.5], [0.9, 0.5]]],
                storage_path="recordings",
                debug_polylines=True,
            )
        assert processor.debug_info == []
        processor.add_video.assert_awaited_once()
        kwargs = processor.add_video.await_args.kwargs
        assert kwargs["fps"] == 2
        assert kwargs["polylines"] == [[(0.1, 0.5), (0.9, 0.5)]]
        assert kwargs["resolved_storage"] == os.path.realpath(
            "/media/llmvision/recordings"
        )
        assert kwargs["debug_polylines"] is True

    @pytest.mark.asyncio
    async def test_invalid_polylines_raise(self, processor):
        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://x",
        ):
            with pytest.raises(ServiceValidationError):
                await processor.add_videos(
                    video_paths=["/config/a.mp4"],
                    event_ids=None,
                    max_frames=3,
                    target_width=1280,
                    include_filename=False,
                    expose_images=False,
                    polylines=[[[2.0, 0.5], [0.1, 0.2]]],
                )

    @pytest.mark.asyncio
    async def test_traversal_storage_raises(self, processor):
        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://x",
        ):
            with pytest.raises(ServiceValidationError):
                await processor.add_videos(
                    video_paths=["/config/a.mp4"],
                    event_ids=None,
                    max_frames=3,
                    target_width=1280,
                    include_filename=False,
                    expose_images=False,
                    storage_path="../../etc",
                )


class TestVideoAnalyzerProService:
    def _handlers(self, hass):
        from custom_components.llmvision import setup

        assert setup(hass, {}) is True
        handlers = {}
        for call in hass.services.register.call_args_list:
            _, service_name, handler = call.args[:3]
            handlers[service_name] = handler
        return handlers

    def test_service_registered(self):
        hass = _make_hass()
        handlers = self._handlers(hass)
        assert "video_analyzer_pro" in handlers

    @pytest.mark.asyncio
    async def test_handler_returns_debug_and_key_frame(self):
        hass = _make_hass()
        handlers = self._handlers(hass)

        call_obj = ServiceCallData(
            _build_data_call({"provider": "e", "message": "m"})
        )
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = "frame.jpg"
        processor.debug_info = [
            {"frame": "clip frame 1", "width": 10, "height": 10, "polylines": []}
        ]
        processor.add_videos = AsyncMock(return_value=request_obj)

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            result = await handlers["video_analyzer_pro"](
                _build_data_call({"provider": "e", "message": "m"})
            )

        assert result["key_frame"] == "frame.jpg"
        assert result["debug"] == processor.debug_info


class TestClipRecording:
    def test_resolve_output_file_relative(self, processor):
        assert processor._resolve_output_file("clips/a.mp4") == os.path.realpath(
            "/media/llmvision/clips/a.mp4"
        )

    def test_resolve_output_file_traversal_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._resolve_output_file("../../etc/x.mp4")

    def test_resolve_output_file_empty_raises(self, processor):
        with pytest.raises(ServiceValidationError):
            processor._resolve_output_file("")

    def test_resolve_output_file_absolute_inside_config(self, processor, tmp_path):
        processor.hass.config.config_dir = str(tmp_path)
        target = str(tmp_path / "clips" / "a.mp4")
        assert processor._resolve_output_file(target) == os.path.realpath(target)

    def test_suffix_path(self, processor):
        assert (
            processor._suffix_path("/media/llmvision/clips/a.mp4", "camera.front_door")
            == "/media/llmvision/clips/a-front_door.mp4"
        )

    def test_build_clip_ffmpeg_cmd(self, processor):
        cmd = processor._build_clip_ffmpeg_cmd(
            "rtsp://x", 5, 15, "/media/llmvision/clips/a.mp4", 1920
        )
        assert cmd[0] == "ffmpeg"
        assert "libx264" in cmd
        assert "yuv420p" in cmd
        assert "+faststart" in cmd
        assert "rtsp://x" in cmd
        assert cmd[-1] == "/media/llmvision/clips/a.mp4"
        # duration passed via -t
        assert cmd[cmd.index("-t") + 1] == "5"
        # correct level must be chosen by x264, not hardcoded to a wrong value
        assert "-level" not in cmd
        # short keyframe interval for smooth decode
        assert "-g" in cmd
        assert cmd[cmd.index("-g") + 1] == "30"
        # downscale cap (never upscales), and forced fps when given
        vf = cmd[cmd.index("-vf") + 1]
        assert "scale='min(1920,iw)':-2" in vf
        assert "fps=15" in vf

    def test_build_clip_ffmpeg_cmd_native(self, processor):
        # No record_fps and scale 0 => passthrough timing, native resolution
        cmd = processor._build_clip_ffmpeg_cmd(
            "rtsp://x", 5, None, "/out.mp4", 0
        )
        assert "-vf" not in cmd
        assert "fps=" not in " ".join(cmd)
        assert "-g" in cmd  # keyframe interval still set (default 30)

    @pytest.mark.asyncio
    async def test_record_clip_no_stream_source_is_skipped(self, processor, tmp_path):
        out = str(tmp_path / "clip.mp4")
        with patch(
            "custom_components.llmvision.media_handlers._async_get_stream_source",
            AsyncMock(return_value=None),
        ):
            result = await processor.record_clip(["camera.front"], 5, out)
        assert result == []
        assert processor.clip_paths == []

    @pytest.mark.asyncio
    async def test_record_clip_success(self, processor, tmp_path):
        out = str(tmp_path / "clip.mp4")

        async def _communicate():
            with open(out, "wb") as handle:
                handle.write(b"video")
            return (b"", b"")

        proc = Mock()
        proc.communicate = _communicate
        proc.returncode = 0

        with (
            patch(
                "custom_components.llmvision.media_handlers._async_get_stream_source",
                AsyncMock(return_value="rtsp://x"),
            ),
            patch(
                "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
        ):
            result = await processor.record_clip(["camera.front"], 5, out, record_fps=15)
        assert result == [out]
        assert processor.clip_paths == [out]

    @pytest.mark.asyncio
    async def test_add_streams_runs_record_clip(self, processor):
        processor.record = AsyncMock()
        processor.record_clip = AsyncMock()
        await processor.add_streams(
            image_entities=["camera.a"],
            duration=5,
            max_frames=3,
            target_width=1280,
            include_filename=False,
            expose_images=False,
            clip_path="clips/a.mp4",
            record_fps=15,
        )
        await asyncio.sleep(0)
        processor.record_clip.assert_awaited_once()
        assert processor.clip_task is not None
        assert processor.requested_clip_paths == [
            os.path.realpath("/media/llmvision/clips/a.mp4")
        ]
        kwargs = processor.record_clip.await_args.kwargs
        assert kwargs["resolved_path"] == os.path.realpath(
            "/media/llmvision/clips/a.mp4"
        )
        assert kwargs["record_fps"] == 15

    @pytest.mark.asyncio
    async def test_add_streams_does_not_wait_for_record_clip(self, processor):
        clip_started = asyncio.Event()
        release_clip = asyncio.Event()

        async def _record_clip(**kwargs):
            clip_started.set()
            await release_clip.wait()

        processor.record = AsyncMock()
        processor.record_clip = AsyncMock(side_effect=_record_clip)

        await asyncio.wait_for(
            processor.add_streams(
                image_entities=["camera.a"],
                duration=5,
                max_frames=3,
                target_width=1280,
                include_filename=False,
                expose_images=False,
                clip_path="clips/a.mp4",
            ),
            timeout=0.1,
        )

        await clip_started.wait()
        assert not release_clip.is_set()
        release_clip.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_record_clip_cancellation_kills_process(self, processor, tmp_path):
        out = str(tmp_path / "clip.mp4")
        communicate_started = asyncio.Event()

        async def _communicate():
            communicate_started.set()
            await asyncio.Event().wait()

        proc = Mock()
        proc.communicate = _communicate
        proc.returncode = None
        proc.kill = Mock()
        proc.wait = AsyncMock()

        with (
            patch(
                "custom_components.llmvision.media_handlers._async_get_stream_source",
                AsyncMock(return_value="rtsp://x"),
            ),
            patch(
                "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
        ):
            task = asyncio.create_task(
                processor.record_clip(["camera.front"], 5, out)
            )
            await communicate_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        proc.kill.assert_called_once()
        proc.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_streams_invalid_clip_path_raises(self, processor):
        processor.record = AsyncMock()
        with pytest.raises(ServiceValidationError):
            await processor.add_streams(
                image_entities=["camera.a"],
                duration=5,
                max_frames=3,
                target_width=1280,
                include_filename=False,
                expose_images=False,
                clip_path="../../etc/x.mp4",
            )

    @pytest.mark.asyncio
    async def test_add_streams_without_clip_skips_record_clip(self, processor):
        processor.record = AsyncMock()
        processor.record_clip = AsyncMock()
        await processor.add_streams(
            image_entities=["camera.a"],
            duration=5,
            max_frames=3,
            target_width=1280,
            include_filename=False,
            expose_images=False,
        )
        processor.record_clip.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_handler_returns_clip(self):
        hass = _make_hass()
        from custom_components.llmvision import setup

        assert setup(hass, {}) is True
        handlers = {}
        for call in hass.services.register.call_args_list:
            _, service_name, handler = call.args[:3]
            handlers[service_name] = handler

        call_obj = ServiceCallData(_build_data_call({"provider": "e", "message": "m"}))
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = ""
        processor.debug_info = None
        processor.clip_paths = ["/media/llmvision/clips/a.mp4"]
        processor.add_streams = AsyncMock(return_value=request_obj)

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            result = await handlers["stream_analyzer_pro"](
                _build_data_call({"provider": "e", "message": "m"})
            )

        assert result["clip"] == "/media/llmvision/clips/a.mp4"

    @pytest.mark.asyncio
    async def test_stream_handler_returns_requested_clip_while_recording(self):
        hass = _make_hass()
        from custom_components.llmvision import setup

        assert setup(hass, {}) is True
        handlers = {}
        for call in hass.services.register.call_args_list:
            _, service_name, handler = call.args[:3]
            handlers[service_name] = handler

        call_obj = ServiceCallData(_build_data_call({"provider": "e", "message": "m"}))
        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        memory_obj = Mock()
        memory_obj._update_memory = AsyncMock()
        processor = Mock()
        processor.key_frame = ""
        processor.debug_info = None
        processor.clip_paths = []
        processor.requested_clip_paths = ["/media/llmvision/clips/pending.mp4"]
        processor.add_streams = AsyncMock(return_value=request_obj)

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call_obj),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory_obj),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
        ):
            result = await handlers["stream_analyzer_pro"](
                _build_data_call({"provider": "e", "message": "m"})
            )

        assert result["clip"] == "/media/llmvision/clips/pending.mp4"
