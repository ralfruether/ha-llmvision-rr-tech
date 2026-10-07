"""Tests for llmvision.store_analysis_record (analysis.json next to clip and snapshots)."""
import json
import os
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.exceptions import ServiceValidationError

from custom_components.llmvision import analysis_record


@pytest.fixture
def media(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_record, "MEDIA_ROOT", str(tmp_path))
    (tmp_path / "llmvision").mkdir()
    return tmp_path / "llmvision"


def _hass():
    hass = Mock()

    async def run(func, *args):
        return func(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=run)
    return hass


class TestResolve:
    def test_relative_is_placed_below_media_llmvision(self, media):
        assert analysis_record.resolve_record_directory("camera-analysis/a") == os.path.realpath(
            media / "camera-analysis" / "a"
        )

    def test_absolute_inside_is_allowed(self, media):
        target = str(media / "camera-analysis" / "b")
        assert analysis_record.resolve_record_directory(target) == os.path.realpath(target)

    @pytest.mark.parametrize("value", ["", "   ", None, 5, "../x", "../../etc", "/etc", "."])
    def test_outside_or_invalid_rejected(self, media, value):
        with pytest.raises(ServiceValidationError):
            analysis_record.resolve_record_directory(value)

    def test_symlink_out_of_the_root_rejected(self, media, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside, media / "link")
        with pytest.raises(ServiceValidationError):
            analysis_record.resolve_record_directory("link/x")


class TestSerialize:
    def test_dict_and_json_string(self):
        assert json.loads(analysis_record.serialize_record({"a": "ü"})) == {"a": "ü"}
        assert json.loads(analysis_record.serialize_record('{"a": true, "b": null}')) == {
            "a": True,
            "b": None,
        }

    @pytest.mark.parametrize("value", ["[1, 2]", "not json", 3, None, ["a"]])
    def test_non_objects_rejected(self, value):
        with pytest.raises(ServiceValidationError):
            analysis_record.serialize_record(value)

    def test_nan_rejected(self):
        with pytest.raises(ServiceValidationError):
            analysis_record.serialize_record({"x": float("nan")})

    def test_size_limit(self, monkeypatch):
        monkeypatch.setattr(analysis_record, "MAX_RECORD_BYTES", 50)
        with pytest.raises(ServiceValidationError, match="too large"):
            analysis_record.serialize_record({"x": "y" * 100})


class TestStore:
    @pytest.mark.asyncio
    async def test_writes_and_replaces_atomically(self, media):
        hass = _hass()
        first = await analysis_record.async_store_analysis_record(
            hass, "camera-analysis/c", {"notification_level": "alarm"}
        )
        path = media / "camera-analysis" / "c" / "analysis.json"
        assert first == {"path": os.path.realpath(path), "bytes": path.stat().st_size}
        await analysis_record.async_store_analysis_record(
            hass, "camera-analysis/c", json.dumps({"notification_level": "none"})
        )
        assert json.loads(path.read_text()) == {"notification_level": "none"}
        assert sorted(os.listdir(path.parent)) == ["analysis.json"]  # no temp files left

    @pytest.mark.asyncio
    async def test_write_error_is_a_validation_error(self, media, monkeypatch):
        def fail(*_args):
            raise PermissionError("denied")

        monkeypatch.setattr(analysis_record, "write_record", fail)
        with pytest.raises(ServiceValidationError, match="PermissionError"):
            await analysis_record.async_store_analysis_record(_hass(), "camera-analysis/d", {})


def test_service_is_registered():
    from custom_components.llmvision import setup

    hass = Mock()
    hass.data = {}
    hass.services = Mock()
    hass.http = Mock()
    assert setup(hass, {}) is True
    names = {c.args[1] for c in hass.services.register.call_args_list}
    assert "store_analysis_record" in names
