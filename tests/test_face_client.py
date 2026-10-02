"""Unit tests for the optional local face service client."""
import asyncio
import json
import logging
import math

import aiohttp
import pytest
from unittest.mock import AsyncMock, Mock, patch

from custom_components.llmvision import face_client
from custom_components.llmvision.const import (
    CONF_FACE_SERVICE_TOKEN,
    CONF_FACE_SERVICE_URL,
    CONF_PROVIDER,
)
from custom_components.llmvision.face_client import (
    FacePerson,
    FaceResult,
    FaceSample,
    FaceServiceError,
    FaceSettings,
)

TOKEN = "s3cret-Token_value.0123456789"
SETTINGS = FaceSettings(url="http://192.168.9.230:8770", token=TOKEN)
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"mdat-bytes"

REAL_RESPONSE = {
    "identify_id": "abc",
    "has_faces": True,
    "has_unknown": True,
    "uncertain": 0,
    "persons": [
        {
            "name": "lea",
            "score": 0.7563,
            "samples": [{"t": 11.36, "box": [0.7355, 0.5115, 0.7699, 0.5664]}],
        },
        {
            "name": "anke",
            "score": 0.4838,
            "samples": [{"t": 15.0, "box": [0.9076, 0.2214, 0.9785, 0.3318]}],
        },
    ],
    "processing_ms": 9741,
    "versions": {"model": "m", "gallery": "g", "policy": "p"},
}


def _payload(persons, has_faces=True, has_unknown=False):
    return {"has_faces": has_faces, "has_unknown": has_unknown, "persons": persons}


def _person(name="lea", score=0.8, samples=None):
    if samples is None:
        samples = [{"t": 1.0, "box": [0.1, 0.1, 0.2, 0.2]}]
    return {"name": name, "score": score, "samples": samples}


# ------------------------------------------------------------------ validation


class TestParsePayload:
    def test_real_response(self):
        persons, has_faces, has_unknown = face_client.parse_identify_payload(
            REAL_RESPONSE
        )
        assert [p.name for p in persons] == ["lea", "anke"]
        assert persons[0].samples == (
            FaceSample(t=11.36, box=(0.7355, 0.5115, 0.7699, 0.5664)),
        )
        assert has_faces is True and has_unknown is True

    @pytest.mark.parametrize(
        "name",
        ["lea\n", "Lea", "a" * 41, "1lea", " lea", "lea2", "", "léa", "lea;rm", None, 5],
    )
    def test_invalid_names_dropped(self, name):
        persons, _, _ = face_client.parse_identify_payload(
            _payload([_person(name=name), _person(name="anke")])
        )
        assert [p.name for p in persons] == ["anke"]

    @pytest.mark.parametrize("name", ["a" * 40, "mary-jane", "o'neil", "anna maria", "dr. who"])
    def test_valid_names_kept(self, name):
        persons, _, _ = face_client.parse_identify_payload(_payload([_person(name=name)]))
        assert [p.name for p in persons] == [name]

    @pytest.mark.parametrize(
        "score", [math.nan, math.inf, -0.1, 1.01, True, "0.5", None]
    )
    def test_invalid_score_drops_person(self, score):
        persons, _, _ = face_client.parse_identify_payload(
            _payload([_person(score=score)])
        )
        assert persons == []

    @pytest.mark.parametrize(
        "sample",
        [
            {"t": math.nan, "box": [0.1, 0.1, 0.2, 0.2]},
            {"t": -0.1, "box": [0.1, 0.1, 0.2, 0.2]},
            {"t": 3600.5, "box": [0.1, 0.1, 0.2, 0.2]},
            {"t": True, "box": [0.1, 0.1, 0.2, 0.2]},
            {"t": 1.0, "box": [0.1, 0.1, 1.2, 0.2]},
            {"t": 1.0, "box": [-0.1, 0.1, 0.2, 0.2]},
            {"t": 1.0, "box": [0.3, 0.1, 0.2, 0.2]},
            {"t": 1.0, "box": [0.1, 0.3, 0.2, 0.2]},
            {"t": 1.0, "box": [0.1, 0.1, 0.1, 0.2]},
            {"t": 1.0, "box": [0.1, 0.1, 0.2]},
            {"t": 1.0, "box": [0.1, 0.1, 0.2, math.inf]},
            {"t": 1.0, "box": [0.1, 0.1, 0.2, True]},
            {"t": 1.0, "box": "0.1,0.1,0.2,0.2"},
            "sample",
        ],
    )
    def test_invalid_samples_dropped(self, sample):
        valid = {"t": 2.0, "box": [0.1, 0.1, 0.2, 0.2]}
        persons, _, _ = face_client.parse_identify_payload(
            _payload([_person(samples=[sample, valid])])
        )
        assert persons[0].samples == (FaceSample(t=2.0, box=(0.1, 0.1, 0.2, 0.2)),)

    def test_boundaries_accepted(self):
        persons, _, _ = face_client.parse_identify_payload(
            _payload(
                [
                    _person(
                        score=1,
                        samples=[
                            {"t": 0, "box": [0, 0, 1, 1]},
                            {"t": 3600, "box": [0.0, 0.0, 0.5, 0.5]},
                        ],
                    )
                ]
            )
        )
        assert len(persons[0].samples) == 2

    @pytest.mark.parametrize(
        "body",
        [
            [],
            "x",
            {"has_faces": "true", "has_unknown": False, "persons": []},
            {"has_faces": True, "has_unknown": 1, "persons": []},
            {"has_faces": True, "has_unknown": False},
            {"has_faces": True, "has_unknown": False, "persons": {}},
            {"has_unknown": False, "persons": []},
        ],
    )
    def test_structurally_invalid_body(self, body):
        with pytest.raises(FaceServiceError) as err:
            face_client.parse_identify_payload(body)
        assert err.value.reason == "invalid_response"

    def test_duplicates_dropped_and_limits(self):
        many = [_person(name=chr(ord("a") + i)) for i in range(15)]
        persons, _, _ = face_client.parse_identify_payload(
            _payload([_person(name="a", score=0.1)] + many)
        )
        assert len(persons) == 10
        assert persons[0].score == 0.1
        assert len({p.name for p in persons}) == 10

        samples = [{"t": i / 10, "box": [0.1, 0.1, 0.2, 0.2]} for i in range(600)]
        persons, _, _ = face_client.parse_identify_payload(
            _payload([_person(samples=samples)])
        )
        assert len(persons[0].samples) == 500

    def test_unknown_fields_ignored(self):
        body = _payload([dict(_person(), extra={"x": 1})])
        body["future"] = [1, 2]
        persons, _, _ = face_client.parse_identify_payload(body)
        assert [p.name for p in persons] == ["lea"]

    @pytest.mark.parametrize(
        "raw",
        [
            b'{"has_faces": true, "has_unknown": false, "persons": [], "x": NaN}',
            b'{"has_faces": true, "has_unknown": false, "persons": [], "x": Infinity}',
            b"not json",
            b"\xff\xfe\x00",
            b"[" * 100000,
        ],
    )
    def test_decode_rejects_bad_json(self, raw):
        with pytest.raises(FaceServiceError) as err:
            face_client.decode_identify_body(raw)
        assert err.value.reason == "invalid_response"

    def test_decode_huge_integer_dropped(self):
        raw = (
            b'{"has_faces": true, "has_unknown": false, "persons": [{"name": "lea", '
            b'"score": 1' + b"0" * 400 + b', "samples": []}]}'
        )
        persons, _, _ = face_client.decode_identify_body(raw)
        assert persons == []


# ------------------------------------------------------------------ mapping


class TestLabelsAndBoxes:
    def _person(self, *times):
        return FacePerson(
            name="lea",
            score=0.9,
            samples=tuple(FaceSample(t=t, box=(0.1, 0.1, 0.2, 0.2)) for t in times),
        )

    def test_nearest_sample_within_window(self):
        person = self._person(0.4, 1.2, 3.0)
        assert face_client.nearest_sample(person.samples, 1.0).t == 1.2
        assert face_client.nearest_sample(person.samples, 2.8).t == 3.0
        assert face_client.nearest_sample(person.samples, 3.25).t == 3.0

    def test_no_sample_outside_window(self):
        person = self._person(0.0, 3.0)
        assert face_client.nearest_sample(person.samples, 1.5) is None
        assert face_client.nearest_sample(person.samples, 3.26) is None
        assert face_client.nearest_sample(person.samples, 0.5) is None  # one slot away
        assert face_client.labels_for_time([person], 1.5) == []

    def test_labels_for_time(self):
        anke = FacePerson("anke", 0.5, (FaceSample(5.0, (0.5, 0.5, 0.6, 0.6)),))
        labels = face_client.labels_for_time([self._person(1.0), anke], 1.2)
        assert labels == [("lea", (0.1, 0.1, 0.2, 0.2))]

    def test_head_box_scaled_around_centre(self):
        assert face_client.head_box((0.4, 0.4, 0.6, 0.6), 1000, 500) == (
            340,
            170,
            660,
            330,
        )

    def test_head_box_clamped(self):
        assert face_client.head_box((0.0, 0.0, 0.1, 0.1), 100, 100) == (0, 0, 13, 13)
        assert face_client.head_box((0.9, 0.9, 1.0, 1.0), 100, 100) == (87, 87, 99, 99)


class TestFactsAndResponse:
    def _ok(self, *names, has_unknown=False, scores=None, elapsed=10):
        scores = scores or {}
        return FaceResult(
            camera="c",
            status="ok",
            persons=[FacePerson(n, scores.get(n, 0.5)) for n in names],
            has_faces=bool(names) or has_unknown,
            has_unknown=has_unknown,
            elapsed_ms=elapsed,
        )

    def test_labeled_names_sorted_and_quoted(self):
        text = face_client.build_fact_text(
            [self._ok("lea", "anke")], labeled_names={"anke", "lea", "other"}
        )
        assert text == (
            "Local face recognition identified household members, marked in the "
            'images with a green box and their name: "anke", "lea". Treat these '
            "persons as confirmed known persons. Do not assign names to any other "
            "person."
        )

    def test_unlabeled_with_unknown(self):
        text = face_client.build_fact_text(
            [self._ok("lea", has_unknown=True)]
        )
        assert 'identified household members in this clip: "lea".' in text
        assert "green box" not in text
        assert text.endswith(
            "At least one other face could not be identified (it may still be a "
            "household member seen at a bad angle)."
        )

    def test_only_drawn_names_called_marked(self):
        text = face_client.build_fact_text(
            [self._ok("lea", "anke")], labeled_names={"lea"}
        )
        assert (
            'marked in the images with a green box and their name: "lea". It also '
            'identified these household members in this clip: "anke". Treat these'
        ) in text

    def test_unknown_only(self):
        text = face_client.build_fact_text([self._ok(has_unknown=True)])
        assert text.startswith("Local face recognition could not identify")
        assert '"' not in text

    def test_nothing_added(self):
        assert face_client.build_fact_text([self._ok()]) == ""
        assert face_client.build_fact_text([]) == ""
        error = FaceResult(camera="c", status="error:timeout")
        assert face_client.build_fact_text([error], {"lea"}) == ""

    def test_response_fields(self):
        results = [
            self._ok("lea", scores={"lea": 0.75634}, elapsed=30),
            self._ok("lea", "anke", scores={"lea": 0.5, "anke": 0.48381}, elapsed=50),
        ]
        assert face_client.service_response_fields(True, results) == {
            "persons": [
                {"name": "lea", "score": 0.756},
                {"name": "anke", "score": 0.484},
            ],
            "face_service": "ok",
            "face_service_ms": 50,
        }

    def test_response_fields_error_and_disabled(self):
        error = FaceResult(camera="c", status="error:http_503", elapsed_ms=7)
        fields = face_client.service_response_fields(True, [self._ok("lea"), error])
        assert fields["face_service"] == "error:http_503"
        assert fields["persons"] == [{"name": "lea", "score": 0.5}]
        assert face_client.service_response_fields(True, [])["face_service"] == (
            "error:no_clip"
        )
        assert face_client.service_response_fields(False, [error]) == {
            "persons": [],
            "face_service": "disabled",
            "face_service_ms": 0,
        }


# ------------------------------------------------------------------ settings


class TestSettings:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://192.168.9.230:8770", "http://192.168.9.230:8770"),
            (" http://192.168.9.230:8770/ ", "http://192.168.9.230:8770"),
            ("http://10.0.0.5/face/", "http://10.0.0.5/face"),
            ("http://127.0.0.1:8770", "http://127.0.0.1:8770"),
            ("http://[fd00::1]:8770", "http://[fd00::1]:8770"),
            ("http://mac-mini.local:8770", "http://mac-mini.local:8770"),
            ("http://facebox.lan", "http://facebox.lan"),
            ("https://face.example.com", "https://face.example.com"),
            ("HTTPS://8.8.8.8:443", "HTTPS://8.8.8.8:443"),
            ("", ""),
            (None, ""),
            ("   ", ""),
        ],
    )
    def test_valid_urls(self, url, expected):
        assert face_client.normalize_face_service_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "http://8.8.8.8:8770",
            "http://face.example.com",
            "http://local",
            "http://.lan",
            "http://0.0.0.0:8770",
            "ftp://192.168.1.2",
            "192.168.1.2:8770",
            "http://user:pw@192.168.1.2",
            "http://@192.168.1.2",
            "http://192.168.1.2/?x=1",
            "http://192.168.1.2#frag",
            "http://",
            "https://",
            "http://192.168.1.2:99999",
            "http://192.168.1.2:abc",
            "http://192.168.1 .2",
            "javascript:alert(1)",
        ],
    )
    def test_invalid_urls(self, url):
        with pytest.raises(ValueError):
            face_client.normalize_face_service_url(url)

    @pytest.mark.parametrize(
        "token,valid",
        [
            (TOKEN, True),
            ("a" * 16, True),
            ("a" * 512, True),
            ("a" * 15, False),
            ("a" * 513, False),
            ("abcdefghijklmnop\n", False),
            ("abcdefgh ijklmnop", False),
            ("abcdefghijklmnop:", False),
            ("", False),
            (None, False),
        ],
    )
    def test_token_validation(self, token, valid):
        assert face_client.is_valid_face_service_token(token) is valid

    def _hass(self, data):
        hass = Mock()
        entry = Mock()
        entry.data = data
        hass.config_entries.async_entries = Mock(return_value=[entry])
        return hass

    def test_old_entry_disabled(self):
        hass = self._hass({CONF_PROVIDER: "Settings", "retention_time": 7})
        assert face_client.get_face_settings(hass) is None

    def test_configured_entry(self):
        hass = self._hass(
            {
                CONF_PROVIDER: "Settings",
                CONF_FACE_SERVICE_URL: "http://192.168.9.230:8770/",
                CONF_FACE_SERVICE_TOKEN: TOKEN,
            }
        )
        settings = face_client.get_face_settings(hass)
        assert settings.url == "http://192.168.9.230:8770"
        assert settings.token == TOKEN
        assert TOKEN not in repr(settings)

    @pytest.mark.parametrize(
        "url,token",
        [
            ("", TOKEN),
            ("http://192.168.9.230:8770", ""),
            ("http://8.8.8.8", TOKEN),
            ("http://192.168.9.230:8770", "short"),
        ],
    )
    def test_incomplete_or_invalid_entry_disabled(self, url, token):
        hass = self._hass(
            {
                CONF_PROVIDER: "Settings",
                CONF_FACE_SERVICE_URL: url,
                CONF_FACE_SERVICE_TOKEN: token,
            }
        )
        assert face_client.get_face_settings(hass) is None

    def test_no_settings_entry(self):
        hass = self._hass({CONF_PROVIDER: "OpenAI"})
        assert face_client.get_face_settings(hass) is None


class TestCameraSlug:
    @pytest.mark.parametrize(
        "entity,slug",
        [
            ("camera.haustuer", "haustuer"),
            ("camera.front door", "front_door"),
            ("camera.", "camera"),
            ("camera." + "x" * 80, "x" * 64),
        ],
    )
    def test_entity_slug(self, entity, slug):
        assert face_client.camera_slug_for_entity(entity) == slug

    @pytest.mark.parametrize(
        "path,slug",
        [
            (
                "/media/camera-analysis/haustuer_20260926T180111_936970/clip.mp4",
                "haustuer",
            ),
            ("/media/clips/garten.mp4", "garten"),
            ("/media/clips/Front Door.mp4", "Front_Door"),
            ("/media/x/recording.mp4", "x"),
            ("/video.mp4", "camera"),
            ("http://ha.local/api/clips/einfahrt.mp4?authSig=abc", "einfahrt"),
            (
                "/api/frigate/notifications/1700000000.1-abc/clip.mp4",
                "1700000000.1-abc",
            ),
            ("", "camera"),
        ],
    )
    def test_video_slug(self, path, slug):
        assert face_client.camera_slug_for_video(path) == slug


# ------------------------------------------------------------------ HTTP


class FakeContent:
    def __init__(self, body, chunk_size=1000):
        self._body = body
        self._chunk_size = chunk_size

    async def read(self, n=-1):
        size = self._chunk_size if n < 0 else min(n, self._chunk_size)
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk


class FakeResponse:
    def __init__(self, status=200, body=b"", content_type="application/json", block=False):
        self.status = status
        self.content_type = content_type
        self.content = FakeContent(body)
        self._block = block

    async def __aenter__(self):
        if self._block:
            await asyncio.Event().wait()
        return self

    async def __aexit__(self, *exc):
        return False


class RaisingContext:
    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _hass():
    hass = Mock()
    hass.loop = Mock()
    hass.loop.run_in_executor = AsyncMock(
        side_effect=lambda _executor, func, *args: func(*args)
    )
    return hass


async def _identify(response, caplog, **kwargs):
    session = FakeSession(response)
    caplog.set_level(logging.DEBUG)
    with patch.object(face_client, "async_get_clientsession", return_value=session):
        result = await face_client.async_identify(
            _hass(), SETTINGS, "haustuer", clip_data=b"clip-bytes", **kwargs
        )
    return result, session


class TestIdentifyRequest:
    @pytest.mark.asyncio
    async def test_success(self, caplog):
        body = json.dumps(REAL_RESPONSE).encode()
        result, session = await _identify(FakeResponse(body=body), caplog)
        assert result.status == "ok" and result.ok
        assert [p.name for p in result.persons] == ["lea", "anke"]
        assert result.has_unknown is True
        url, kwargs = session.calls[0]
        assert url == "http://192.168.9.230:8770/v1/identify"
        assert kwargs["params"] == {"camera": "haustuer"}
        assert kwargs["data"] == b"clip-bytes"
        assert kwargs["headers"] == {
            "Content-Type": "video/mp4",
            "Authorization": f"Bearer {TOKEN}",
        }
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"].total == 12
        assert TOKEN not in caplog.text
        assert "lea" not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403, 413, 415, 422, 500, 503, 504])
    async def test_http_errors(self, status, caplog):
        result, _ = await _identify(
            FakeResponse(status=status, body=b'{"detail": "lea secret"}'), caplog
        )
        assert result.status == f"error:http_{status}"
        assert result.persons == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert f"error:http_{status}" in warnings[0].getMessage()
        assert TOKEN not in caplog.text
        assert "lea secret" not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    async def test_redirect_is_error(self, status, caplog):
        result, _ = await _identify(FakeResponse(status=status), caplog)
        assert result.status == "error:redirect"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,status",
        [
            (asyncio.TimeoutError(), "error:timeout"),
            (aiohttp.ServerTimeoutError(), "error:timeout"),
            (aiohttp.ClientConnectionError(f"Bearer {TOKEN}"), "error:connection"),
            (aiohttp.ClientPayloadError("x"), "error:connection"),
            (ConnectionRefusedError(), "error:connection"),
            (RuntimeError(f"boom {TOKEN}"), "error:invalid_response"),
        ],
    )
    async def test_exceptions_mapped(self, exc, status, caplog):
        result, _ = await _identify(RaisingContext(exc), caplog)
        assert result.status == status
        assert TOKEN not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            FakeResponse(body=json.dumps(REAL_RESPONSE).encode(), content_type="text/html"),
            FakeResponse(body=json.dumps(REAL_RESPONSE).encode(), content_type=None),
            FakeResponse(body=b"{not json"),
            FakeResponse(body=b'{"has_faces": NaN, "has_unknown": false, "persons": []}'),
            FakeResponse(body=b'{"persons": []}'),
            FakeResponse(body=b"[" + b" " * (300 * 1024) + b"]"),
        ],
    )
    async def test_invalid_responses(self, response, caplog):
        result, _ = await _identify(response, caplog)
        assert result.status == "error:invalid_response"

    @pytest.mark.asyncio
    async def test_oversized_body_reads_at_most_limit_plus_one(self, caplog):
        response = FakeResponse(body=b"x" * (1024 * 1024))
        result, _ = await _identify(response, caplog)
        assert result.status == "error:invalid_response"
        # Only MAX+1 bytes were consumed from the stream
        assert len(response.content._body) == 1024 * 1024 - (256 * 1024 + 1)

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self):
        session = FakeSession(FakeResponse(block=True))
        with patch.object(face_client, "async_get_clientsession", return_value=session):
            task = asyncio.ensure_future(
                face_client.async_identify(
                    _hass(), SETTINGS, "haustuer", clip_data=b"clip"
                )
            )
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


class TestClipLoading:
    @pytest.mark.asyncio
    async def test_reads_clip_from_path(self, tmp_path, caplog):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(MP4)
        session = FakeSession(
            FakeResponse(body=json.dumps(_payload([])).encode())
        )
        with patch.object(face_client, "async_get_clientsession", return_value=session):
            result = await face_client.async_identify(
                _hass(), SETTINGS, "cam", clip_path=str(clip)
            )
        assert result.status == "ok"
        assert session.calls[0][1]["data"] == MP4

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", [None, "missing.mp4", "empty.mp4", "dir"])
    async def test_missing_clip(self, tmp_path, name, caplog):
        (tmp_path / "empty.mp4").write_bytes(b"")
        (tmp_path / "dir").mkdir()
        path = str(tmp_path / name) if name else None
        session = FakeSession(FakeResponse())
        with patch.object(face_client, "async_get_clientsession", return_value=session):
            result = await face_client.async_identify(
                _hass(), SETTINGS, "cam", clip_path=path
            )
        assert result.status == "error:no_clip"
        assert session.calls == []

    @pytest.mark.asyncio
    async def test_non_video_file_not_uploaded(self, tmp_path):
        secrets = tmp_path / "secrets.yaml"
        secrets.write_text("api_key: super-secret-value\n")
        session = FakeSession(FakeResponse())
        with patch.object(face_client, "async_get_clientsession", return_value=session):
            result = await face_client.async_identify(
                _hass(), SETTINGS, "cam", clip_path=str(secrets)
            )
        assert result.status == "error:no_clip"
        assert session.calls == []

    @pytest.mark.asyncio
    async def test_too_large_clip_not_uploaded(self, tmp_path):
        clip = tmp_path / "big.mp4"
        with open(clip, "wb") as handle:
            handle.truncate(face_client.MAX_CLIP_BYTES + 1)
        session = FakeSession(FakeResponse())
        with patch.object(face_client, "async_get_clientsession", return_value=session):
            result = await face_client.async_identify(
                _hass(), SETTINGS, "cam", clip_path=str(clip)
            )
        assert result.status == "error:too_large"
        assert session.calls == []

    @pytest.mark.asyncio
    async def test_load_runs_in_executor(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(MP4)
        hass = _hass()
        assert await face_client.async_load_clip(hass, str(clip)) == MP4
        hass.loop.run_in_executor.assert_awaited_once()


def test_fps_frame_time_matches_ffmpeg_fps_filter():
    # ffmpeg's fps filter keeps the last input frame of each output slot, so output
    # frame n shows the scene at about (n + 0.5) / rate (measured +0.47 s at fps=1).
    assert face_client.fps_frame_time(0, 1.0) == 0.5
    assert face_client.fps_frame_time(6, 1.0) == 6.5
    assert face_client.fps_frame_time(3, 2.0) == 1.75
