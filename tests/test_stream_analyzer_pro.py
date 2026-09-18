"""Unit tests for the stream_analyzer_pro feature."""
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
