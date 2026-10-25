"""Tests for stream_analyzer_pro's pretrigger_frame option."""
import asyncio
import base64
import io
import logging
import time
from datetime import datetime

import pytest
from unittest.mock import AsyncMock, Mock, patch
from PIL import Image

from homeassistant.exceptions import ServiceValidationError

from custom_components.llmvision import face_client, pretrigger
from custom_components.llmvision.const import CONF_PROVIDER, DOMAIN
from custom_components.llmvision.face_client import (
    AppearanceInfo,
    AppearanceMatch,
    FacePerson,
    FaceResult,
)
from custom_components.llmvision.media_handlers import MediaProcessor
from custom_components.llmvision.pretrigger import PretriggerEntry
from custom_components.llmvision.providers import Request


def _jpeg(color, size=(100, 50)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _run_executor(_executor, func, *args):
    return func(*args)


def _ok(*names, appearance=None):
    return FaceResult(
        camera="a",
        status="ok",
        persons=[FacePerson(name=name, score=0.7) for name in names],
        has_faces=bool(names),
        appearance=appearance,
    )


def _entry(monotonic, names=None, provider="p"):
    return PretriggerEntry(
        frame=b"jpeg",
        captured_at=datetime(2026, 10, 25, 14, 23, 59),
        monotonic=monotonic,
        provider=provider,
        names=names,
    )


class TestModule:
    def test_household_names_face_only(self):
        appearance = AppearanceInfo(
            status="ok",
            matches=(AppearanceMatch("ralf", 0.95, "strong", True, 0.6, 60.0, True),),
        )
        assert pretrigger.household_names([_ok("lea", appearance=appearance)]) == ("lea",)

    def test_household_names_none_on_failure_or_missing(self):
        failed = FaceResult(camera="a", status="error:timeout")
        assert pretrigger.household_names([_ok("lea"), failed]) is None
        assert pretrigger.household_names([]) is None

    def test_household_names_empty_when_nobody(self):
        assert pretrigger.household_names([_ok()]) == ()

    def test_frame_label_uses_sanitized_slug(self):
        assert pretrigger.frame_label("camera.einfahrt", True) == "einfahrt-vortrigger"
        assert pretrigger.frame_label("camera.a/../b c", True) == "a_.._b_c-vortrigger"
        assert pretrigger.frame_label("camera.einfahrt", False) == "camera0-vortrigger"

    @pytest.mark.parametrize(
        "entry_mono,provider,expected",
        [
            (900.0, "p", 100),  # 100 s before the trigger
            (100.0, "p", 900),  # exactly max age
            (99.0, "p", None),  # older than max age
            (1000.5, "p", None),  # captured after the trigger (queued call)
            (900.0, "other", None),  # captured for another provider
        ],
    )
    def test_usable_age(self, entry_mono, provider, expected):
        entry = _entry(entry_mono, provider=provider)
        assert pretrigger.usable_age(entry, 1000.0, 900, "p") == expected

    def test_usable_age_without_entry(self):
        assert pretrigger.usable_age(None, 1000.0, 900, "p") is None

    def test_cache_prune_drops_expired_and_bounds_size(self):
        cache = pretrigger.PretriggerCache()
        cache.publish("camera.old", cache.next_seq(), _entry(0.0), 0.0)
        for i in range(pretrigger.MAX_ENTRIES + 3):
            cache.publish(f"camera.c{i}", cache.next_seq(), _entry(5000.0 + i), 5000.0)
        assert "camera.old" not in cache
        assert len(cache) == pretrigger.MAX_ENTRIES
        assert "camera.c0" not in cache
        assert f"camera.c{pretrigger.MAX_ENTRIES + 2}" in cache

    def test_cache_older_call_never_overwrites(self):
        cache = pretrigger.PretriggerCache()
        older, newer = cache.next_seq(), cache.next_seq()
        assert cache.publish("camera.a", newer, None, 10.0)
        assert not cache.publish("camera.a", older, _entry(9.0), 10.0)
        assert cache.get("camera.a") is None
        newest = cache.next_seq()
        assert cache.publish("camera.a", newest, _entry(11.0), 11.0)
        assert cache.get("camera.a").monotonic == 11.0

    @pytest.mark.parametrize(
        "names,expected",
        [
            (("anke", "lea"), 'identified these household members: "anke", "lea".'),
            ((), "identified no household member."),
            (None, "is unknown."),
        ],
    )
    def test_fact_text(self, names, expected):
        text = pretrigger.fact_text("a-vortrigger", _entry(0.0, names=names), 96)
        assert text.startswith(
            'Pre-trigger frame: the first image, labeled "a-vortrigger", is the last '
            "frame of this camera's previous analysis, captured at 14:23:59 local "
            "time, 96 s before this trigger."
        )
        assert "never list it in any frame field" in text
        assert text.endswith(expected)


class TestRequestPrepend:
    def test_prepend_keeps_images_and_names_aligned(self):
        with patch("custom_components.llmvision.providers.async_get_clientsession"):
            request = Request(Mock(), message="m", max_tokens=10, temperature=0.5)
        request.add_frame("b1", "a-frame-0")
        request.prepend_frame("b0", "a-vortrigger")
        assert request.base64_images == ["b0", "b1"]
        assert request.filenames == ["a-vortrigger", "a-frame-0"]


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
    with patch("custom_components.llmvision.media_handlers.async_get_clientsession"):
        return MediaProcessor(hass, client)


class TestMediaProcessor:
    @pytest.mark.asyncio
    async def test_record_keeps_last_captured_frame(self, processor):
        first = ("a-frame-0", _jpeg("red"))
        frames = {
            f"a-frame-{i}": {
                "frame_data": _jpeg(color), "ssim_score": score,
                "camera_number": 0, "frame_index": i,
            }
            # The last frame has the least movement, so max_frames=2 drops it
            for i, color, score in [(1, "blue", 0.1), (2, "green", 0.9)]
        }
        processor._capture_stream_camera = AsyncMock(return_value=(first, frames))
        before = time.monotonic()
        await processor.record(
            image_entities=["camera.a"], duration=5, max_frames=2,
            target_width=2048, include_filename=True, expose_images=False,
            fps=1, frame_source="stream",
        )
        frame, captured = processor.last_frames["camera.a"]
        assert frame == frames["a-frame-2"]["frame_data"]
        assert before <= captured <= time.monotonic()
        labels = [c.kwargs["filename"] for c in processor.client.add_frame.call_args_list]
        assert "a-frame-2" not in labels

    @pytest.mark.asyncio
    async def test_record_first_frame_only(self, processor):
        first = ("a-frame-0", _jpeg("red"))
        processor._capture_stream_camera = AsyncMock(return_value=(first, {}))
        await processor.record(
            image_entities=["camera.a"], duration=5, max_frames=None,
            target_width=2048, include_filename=True, expose_images=False,
            fps=1, frame_source="stream",
        )
        assert processor.last_frames["camera.a"][0] == first[1]

    @pytest.mark.asyncio
    async def test_add_pretrigger_frame_prepends_with_polyline(self, processor, tmp_path):
        processor.hass.config.config_dir = str(tmp_path)
        storage = tmp_path / "evidence"
        await processor.add_pretrigger_frame(
            image_data=_jpeg((0, 0, 0)),
            label="a-vortrigger",
            target_width=100,
            polylines=[[[0.0, 0.5], [1.0, 0.5]]],
            storage_path=str(storage),
        )
        kwargs = processor.client.prepend_frame.call_args.kwargs
        assert kwargs["filename"] == "a-vortrigger"
        img = Image.open(io.BytesIO(base64.b64decode(kwargs["base64_image"])))
        r, g, b = img.getpixel((50, 25))[:3]
        assert r > 150 and g < 100 and b < 100
        stored = storage / "a-vortrigger.jpg"
        assert stored.read_bytes() == base64.b64decode(kwargs["base64_image"])
        processor.client.add_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_snapshot_mode_stamps_last_fetch(self, processor):
        state = Mock()
        state.attributes = {"entity_picture": "/api/camera_proxy/camera.a"}
        processor.hass.states.get = Mock(return_value=state)
        colors = iter(["red", "blue", "green", "white", "black"] * 20)
        fetches = []

        async def fetch(*args, **kwargs):
            fetches.append(time.monotonic())
            return _jpeg(next(colors))

        processor._fetch = AsyncMock(side_effect=fetch)
        with patch("custom_components.llmvision.media_handlers.get_url", return_value=""):
            await processor.record(
                image_entities=["camera.a"], duration=0.3, max_frames=None,
                target_width=2048, include_filename=True, expose_images=False,
                fps=4,
            )
        _, captured = processor.last_frames["camera.a"]
        # Stamped at the last fetch, not after the trailing interval sleep
        assert 0 <= captured - fetches[-1] < 0.1

    @pytest.mark.asyncio
    async def test_annotation_failure_prepends_nothing(self, processor):
        with pytest.raises(Exception):
            await processor.add_pretrigger_frame(
                image_data=b"not a jpeg", label="a-vortrigger", target_width=100
            )
        processor.client.prepend_frame.assert_not_called()

    @pytest.mark.asyncio
    async def test_unexpected_storage_error_keeps_frame(self, processor, caplog):
        processor._write_snapshot = AsyncMock(side_effect=RuntimeError("disk"))
        processor._resolve_storage_path = Mock(return_value="/media/x")
        await processor.add_pretrigger_frame(
            image_data=_jpeg((0, 0, 0)), label="a-vortrigger", target_width=100,
            storage_path="x",
        )
        processor.client.prepend_frame.assert_called_once()
        assert "Could not store the pre-trigger frame" in caplog.text

    @pytest.mark.asyncio
    async def test_storage_failure_keeps_frame(self, processor, caplog):
        await processor.add_pretrigger_frame(
            image_data=_jpeg((0, 0, 0)),
            label="a-vortrigger",
            target_width=100,
            storage_path="/etc",
        )
        processor.client.prepend_frame.assert_called_once()
        assert "Could not store the pre-trigger frame" in caplog.text


# --------------------------------------------------------------- service handler

SETTINGS = Mock()
BASE = {
    "provider": "p",
    "message": "m",
    "image_entity": ["camera.a"],
    "include_filename": True,
    "target_width": 2048,
    "pretrigger_frame": True,
}


def _data_call(data):
    data_call = Mock()
    data_call.data = Mock()
    data_call.data.get = Mock(side_effect=lambda key, default=None: data.get(key, default))
    return data_call


class Harness:
    """One integration setup (shared pre-trigger memory) and per-call doubles."""

    def __init__(self):
        from custom_components.llmvision import setup

        hass = Mock()
        hass.data = {}
        hass.services = Mock()
        hass.services.register = Mock()
        hass.http = Mock()
        assert setup(hass, {}) is True
        self.hass = hass
        handlers = {c.args[1]: c.args[2] for c in hass.services.register.call_args_list}
        self.handler = handlers["stream_analyzer_pro"]
        self.messages = []
        self.processors = []

    async def run(
        self, data, frame=b"frame", captured=None, results=(), fail=False,
        add_error=None, on_call=None,
    ):
        processor = Mock()
        processor.key_frame = ""
        processor.debug_info = None
        processor.clip_paths = []
        processor.clip_task = None
        processor.face_results = []
        processor.face_labeled_names = set()
        processor.face_settings = None
        processor.last_frames = {}
        processor.add_pretrigger_frame = AsyncMock(side_effect=add_error)
        request_obj = Mock()

        async def add_streams(**kwargs):
            if frame is not None:
                processor.last_frames = {
                    "camera.a": (frame, time.monotonic() if captured is None else captured)
                }
            return request_obj

        async def identify():
            processor.face_results = list(results)

        async def call(call_data):
            self.messages.append(call_data.message)
            if on_call is not None:
                await on_call()
            if fail:
                raise RuntimeError("provider down")
            return {"response_text": "ok"}

        processor.add_streams = AsyncMock(side_effect=add_streams)
        processor.load_recorded_clips = AsyncMock()
        processor.identify_recorded_clips = AsyncMock(side_effect=identify)
        request_obj.call = AsyncMock(side_effect=call)
        self.processors.append(processor)
        memory = Mock()
        memory._update_memory = AsyncMock()
        with (
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory),
            patch("custom_components.llmvision._create_event", new=AsyncMock()),
            patch.object(face_client, "get_face_settings", Mock(return_value=SETTINGS)),
        ):
            return await self.handler(_data_call(data))


class TestHandler:
    @pytest.mark.asyncio
    async def test_second_run_gets_previous_frame_and_names(self):
        harness = Harness()
        data = {**BASE, "identify_persons": True}
        first = await harness.run(data, frame=b"first", results=[_ok("lea")])
        assert first["pretrigger"] == {"used": False, "age_s": None}
        harness.processors[0].add_pretrigger_frame.assert_not_called()
        assert "Pre-trigger frame" not in harness.messages[0]

        second = await harness.run(data, frame=b"second", results=[_ok()])
        assert second["pretrigger"]["used"] is True
        assert 0 <= second["pretrigger"]["age_s"] <= 1
        kwargs = harness.processors[1].add_pretrigger_frame.call_args.kwargs
        assert kwargs["image_data"] == b"first"
        assert kwargs["label"] == "a-vortrigger"
        assert kwargs["target_width"] == 2048
        message = harness.messages[1]
        assert 'the first image, labeled "a-vortrigger"' in message
        assert 'identified these household members: "lea".' in message

        third = await harness.run(data, frame=b"third")
        assert harness.processors[2].add_pretrigger_frame.call_args.kwargs[
            "image_data"
        ] == b"second"
        assert third["pretrigger"]["used"] is True
        assert "identified no household member." in harness.messages[2]

    @pytest.mark.asyncio
    async def test_reid_names_never_carried_over(self):
        harness = Harness()
        appearance = AppearanceInfo(
            status="ok",
            matches=(AppearanceMatch("ralf", 0.95, "strong", True, 0.6, 60.0, True),),
        )
        data = {**BASE, "identify_persons": True}
        await harness.run(data, results=[_ok(appearance=appearance)])
        await harness.run(data, results=[_ok()])
        assert "ralf" not in harness.messages[1]
        assert "identified no household member." in harness.messages[1]

    @pytest.mark.asyncio
    async def test_face_failure_reads_as_unknown(self):
        harness = Harness()
        data = {**BASE, "identify_persons": True}
        await harness.run(data, results=[FaceResult(camera="a", status="error:timeout")])
        await harness.run(data)
        assert "household member in that previous analysis is unknown." in harness.messages[1]

    @pytest.mark.asyncio
    async def test_without_identify_persons_names_unknown(self):
        harness = Harness()
        await harness.run(BASE)
        await harness.run(BASE)
        assert "is unknown." in harness.messages[1]

    @pytest.mark.asyncio
    async def test_failed_model_call_is_not_context(self):
        harness = Harness()
        with pytest.raises(RuntimeError):
            await harness.run(BASE, frame=b"arrival", fail=True)
        result = await harness.run(BASE)
        assert result["pretrigger"] == {"used": False, "age_s": None}

    @pytest.mark.asyncio
    async def test_frame_captured_after_trigger_is_not_used(self):
        harness = Harness()
        await harness.run(BASE, captured=time.monotonic() + 60)
        result = await harness.run(BASE)
        assert result["pretrigger"]["used"] is False

    @pytest.mark.asyncio
    async def test_expired_frame_is_not_used(self):
        harness = Harness()
        await harness.run(BASE, captured=time.monotonic() - 120)
        result = await harness.run({**BASE, "pretrigger_max_age": 60})
        assert result["pretrigger"]["used"] is False
        result = await harness.run(BASE)  # the 120 s old frame was replaced
        assert result["pretrigger"]["used"] is True

    @pytest.mark.asyncio
    async def test_other_provider_never_gets_the_frame(self):
        harness = Harness()
        await harness.run(BASE)
        result = await harness.run({**BASE, "provider": "other"})
        assert result["pretrigger"]["used"] is False

    @pytest.mark.asyncio
    async def test_option_off_clears_and_keeps_response(self):
        harness = Harness()
        await harness.run(BASE)
        off = {k: v for k, v in BASE.items() if k != "pretrigger_frame"}
        result = await harness.run(off)
        assert result == {"response_text": "ok"}
        assert "Pre-trigger frame" not in harness.messages[1]
        harness.processors[1].add_pretrigger_frame.assert_not_called()
        result = await harness.run(BASE)
        assert result["pretrigger"]["used"] is False

    @pytest.mark.asyncio
    async def test_oversized_frame_not_cached(self):
        harness = Harness()
        await harness.run(BASE, frame=b"x" * (pretrigger.MAX_FRAME_BYTES + 1))
        result = await harness.run(BASE)
        assert result["pretrigger"]["used"] is False

    @pytest.mark.asyncio
    async def test_frame_failure_keeps_analysis(self, caplog):
        harness = Harness()
        await harness.run(BASE)
        result = await harness.run(BASE, add_error=OSError("broken"))
        assert result["pretrigger"] == {"used": False, "age_s": None}
        assert "Pre-trigger frame" not in harness.messages[1]
        assert "Pre-trigger frame not used" in caplog.text

    @pytest.mark.asyncio
    async def test_run_without_cacheable_frame_clears_older_frame(self):
        harness = Harness()
        await harness.run(BASE, frame=b"older")
        await harness.run(BASE, frame=None)
        result = await harness.run(BASE)
        assert result["pretrigger"]["used"] is False

    @pytest.mark.asyncio
    async def test_option_off_during_inflight_call_wins(self):
        harness = Harness()
        off = {k: v for k, v in BASE.items() if k != "pretrigger_frame"}

        async def off_call():
            await harness.run(off)

        await harness.run(BASE, frame=b"inflight", on_call=off_call)
        result = await harness.run(BASE)
        assert result["pretrigger"]["used"] is False

    @pytest.mark.asyncio
    async def test_slow_older_call_never_overwrites_newer_frame(self):
        harness = Harness()

        async def newer_call():
            await harness.run(BASE, frame=b"newer")

        await harness.run(BASE, frame=b"older", on_call=newer_call)
        await harness.run(BASE)
        assert harness.processors[2].add_pretrigger_frame.call_args.kwargs[
            "image_data"
        ] == b"newer"

    @pytest.mark.asyncio
    async def test_single_image_provider_skips_frame(self):
        harness = Harness()
        harness.hass.data.setdefault(DOMAIN, {})["p"] = {CONF_PROVIDER: "Groq"}
        await harness.run(BASE)
        result = await harness.run(BASE)
        assert result["pretrigger"]["used"] is False
        harness.processors[1].add_pretrigger_frame.assert_not_called()
        assert "Pre-trigger frame" not in harness.messages[1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra,match",
        [
            ({"image_entity": ["camera.a", "camera.b"]}, "exactly one camera"),
            ({"pretrigger_max_age": 30}, "pretrigger_max_age must be between"),
            ({"pretrigger_max_age": "x"}, "pretrigger_max_age must be a number"),
        ],
    )
    async def test_validation(self, extra, match):
        harness = Harness()
        with pytest.raises(ServiceValidationError, match=match):
            await harness.run({**BASE, **extra})

    @pytest.mark.asyncio
    async def test_logs_contain_no_names_or_image_data(self, caplog):
        caplog.set_level(logging.DEBUG)
        harness = Harness()
        data = {**BASE, "identify_persons": True}
        frame = base64.b64encode(b"secret-frame").decode()
        await harness.run(data, frame=frame.encode(), results=[_ok("lea")])
        await harness.run(data)
        assert "Pre-trigger frame added" in caplog.text
        assert "lea" not in caplog.text and frame not in caplog.text
