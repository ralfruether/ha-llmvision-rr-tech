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


# ------------------------------------------------------------------ read access

RID = "garten_20261010T152347_966843"


@pytest.fixture
def records(media):
    root = media / "camera-analysis"
    root.mkdir()
    return root


def _record(root, rid=RID, body=b'{"a": 1}', mtime=None):
    folder = root / rid
    folder.mkdir()
    path = folder / "analysis.json"
    path.write_bytes(body)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class TestListRecords:
    def test_lists_only_valid_dirs_with_a_regular_record(self, records, tmp_path):
        _record(records, RID, mtime=2000)
        _record(records, "haustuer_20261010T100000_1", mtime=1000)
        (records / "einfahrt_20261010T100000_2").mkdir()  # no record
        _record(records, "Not-An-Id", mtime=3000)
        (records / "vorgarten_20261010T100000_3").mkdir()
        os.symlink(tmp_path / "x.json", records / "vorgarten_20261010T100000_3" / "analysis.json")
        outside = tmp_path / "outside_20261010T100000_4"
        outside.mkdir()
        (outside / "analysis.json").write_text("{}")
        os.symlink(outside, records / "outside_20261010T100000_4")
        found, truncated = analysis_record.list_records(0, 10)
        assert [r["id"] for r in found] == ["haustuer_20261010T100000_1", RID]  # oldest first
        assert found[1] == {"id": RID, "mtime": 2000, "bytes": 8} and truncated is False

    def test_since_and_limit(self, records):
        for i in range(5):
            _record(records, f"garten_20261010T10000{i}_{i}", mtime=1000 + i)
        found, truncated = analysis_record.list_records(1002, 2)
        assert [r["mtime"] for r in found] == [1002, 1003] and truncated is True

    def test_missing_root_is_empty(self, media):
        assert analysis_record.list_records(0, 10) == ([], False)


class TestReadRecord:
    def test_reads_bytes_unchanged(self, records):
        _record(records, body='{"name": "lea", "ü": 1}'.encode())
        assert analysis_record.read_record(RID) == '{"name": "lea", "ü": 1}'.encode()

    @pytest.mark.parametrize("value", [
        "../x", "a/b", "", "garten", "x_20261010T152347_1/..", None, RID + "\n",
        "garten_２０261010T152347_1", "Garten_20261010T152347_1", "a" * 65 + "_20261010T152347_1",
        "garten_20261010T152347_1%2F..",
    ])
    def test_invalid_ids_rejected(self, records, value):
        with pytest.raises(ValueError):
            analysis_record.read_record(value)

    def test_missing_is_none(self, records):
        assert analysis_record.read_record(RID) is None
        (records / RID).mkdir()
        assert analysis_record.read_record(RID) is None

    def test_symlinked_dir_and_file_are_not_followed(self, records, tmp_path):
        outside = tmp_path / "out"
        outside.mkdir()
        (outside / "analysis.json").write_text('{"secret": 1}')
        os.symlink(outside, records / RID)
        assert analysis_record.read_record(RID) is None
        other = "haustuer_20261010T100000_1"
        (records / other).mkdir()
        os.symlink(outside / "analysis.json", records / other / "analysis.json")
        assert analysis_record.read_record(other) is None

    def test_too_large(self, records, monkeypatch):
        _record(records, body=b"x" * 21)
        monkeypatch.setattr(analysis_record, "MAX_RECORD_BYTES", 20)
        with pytest.raises(analysis_record.RecordTooLarge):
            analysis_record.read_record(RID)

    def test_exactly_the_limit_is_read(self, records, monkeypatch):
        _record(records, body=b"x" * 20)
        monkeypatch.setattr(analysis_record, "MAX_RECORD_BYTES", 20)
        assert analysis_record.read_record(RID) == b"x" * 20

    def test_fifo_or_directory_named_analysis_json_is_none_without_hanging(self, records):
        (records / RID).mkdir()
        os.mkfifo(records / RID / "analysis.json")
        assert analysis_record.read_record(RID) is None
        other = "haustuer_20261010T100000_1"
        (records / other / "analysis.json").mkdir(parents=True)
        assert analysis_record.read_record(other) is None
        assert analysis_record.list_records(0, 10) == ([], False)

    def test_symlinked_records_root_is_followed_only_as_configured_root(self, media, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        os.symlink(real, media / "camera-analysis")
        _record(real)
        assert analysis_record.read_record(RID) == b'{"a": 1}'


def _request(query=None, admin=True, user=True):
    request = {}
    if user:
        request["hass_user"] = Mock(is_admin=admin)
    req = Mock()
    req.get = request.get
    req.query = query or {}
    req.app = {"hass": _hass()}
    return req


class TestViews:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("admin,user", [(False, True), (False, False)])
    async def test_admin_required(self, records, admin, user):
        from custom_components.llmvision.api import AnalysisRecordsView, AnalysisRecordView

        _record(records)
        req = _request(admin=admin, user=user)
        assert (await AnalysisRecordsView().get(req)).status == 403
        assert (await AnalysisRecordView().get(req, RID)).status == 403
        req.app["hass"].async_add_executor_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_list(self, records):
        from custom_components.llmvision.api import AnalysisRecordsView

        _record(records, mtime=1_791_600_000)
        response = await AnalysisRecordsView().get(_request({"since": "2026-10-10T00:00:00+02:00"}))
        body = json.loads(response.body)
        assert response.status == 200 and response.headers["Cache-Control"] == "no-store"
        assert [r["id"] for r in body["records"]] == [RID] and body["truncated"] is False
        later = await AnalysisRecordsView().get(_request({"since": "1791600001"}))
        assert json.loads(later.body)["records"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        {"since": "yesterday"}, {"since": "nan"}, {"since": "inf"},
        {"limit": "x"}, {"limit": "0"}, {"limit": "1001"},
    ])
    async def test_list_bad_query(self, records, query):
        from custom_components.llmvision.api import AnalysisRecordsView

        assert (await AnalysisRecordsView().get(_request(query))).status == 400

    @pytest.mark.asyncio
    async def test_get(self, records):
        from custom_components.llmvision.api import AnalysisRecordView

        _record(records, body=b'{"level": "alarm"}')
        response = await AnalysisRecordView().get(_request(), RID)
        assert response.status == 200 and response.body == b'{"level": "alarm"}'
        assert response.content_type == "application/json"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert (await AnalysisRecordView().get(_request(), "haustuer_20261010T100000_1")).status == 404
        assert (await AnalysisRecordView().get(_request(), "..")).status == 400

    @pytest.mark.asyncio
    async def test_get_too_large(self, records, monkeypatch):
        from custom_components.llmvision.api import AnalysisRecordView

        _record(records, body=b'{"a": "' + b"x" * 50 + b'"}')
        monkeypatch.setattr(analysis_record, "MAX_RECORD_BYTES", 20)
        assert (await AnalysisRecordView().get(_request(), RID)).status == 413


def test_record_views_are_registered_in_setup():
    from custom_components.llmvision import setup
    from custom_components.llmvision.api import AnalysisRecordsView, AnalysisRecordView

    hass = Mock()
    hass.data = {}
    hass.services = Mock()
    hass.http = Mock()
    setup(hass, {})
    registered = [c.args[0] for c in hass.http.register_view.call_args_list]
    assert AnalysisRecordsView in registered and AnalysisRecordView in registered
