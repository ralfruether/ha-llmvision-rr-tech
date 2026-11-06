"""Unit tests for calendar.py module."""
import pytest
import datetime
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from zoneinfo import ZoneInfo
from homeassistant.components.calendar import CalendarEvent
from homeassistant.util import dt as dt_util


class MockTimelineEvent:
    """Mock timeline event."""
    def __init__(self, uid, title, start, end, description):
        self.uid = uid
        self.title = title
        self.start = start
        self.end = end
        self.description = description


class TestCalendar:
    """Test Calendar class."""

    @pytest.fixture
    def mock_timeline(self):
        """Create a mock Timeline."""
        timeline = Mock()
        timeline.get_all_events = AsyncMock(return_value=[])
        timeline.load_events = AsyncMock()
        timeline.delete_event = AsyncMock()
        return timeline

    @pytest.fixture
    def calendar_instance(self, mock_hass, mock_config_entry, mock_timeline):
        """Create a Calendar instance with mocked dependencies."""
        from custom_components.llmvision.calendar import Calendar

        with patch('custom_components.llmvision.calendar.Timeline', return_value=mock_timeline):
            with patch('os.makedirs'):
                with patch('os.path.join', return_value="/mock/path"):
                    calendar = Calendar(mock_hass, mock_config_entry)
                    return calendar

    def test_init(self, calendar_instance):
        """Test Calendar initialization."""
        assert calendar_instance._attr_name == "LLM Vision Timeline"
        assert calendar_instance._attr_unique_id == "llm_vision_timeline"
        assert calendar_instance._events == []
        assert calendar_instance._current_event is None

    def test_icon_property(self, calendar_instance):
        """Test icon property."""
        assert calendar_instance.icon == "mdi:timeline-outline"

    def test_event_property(self, calendar_instance):
        """Test event property."""
        assert calendar_instance.event is None

        # Set an event
        test_event = Mock()
        calendar_instance._current_event = test_event
        assert calendar_instance.event == test_event

    def test_extra_state_attributes_returns_latest_event_data(self, calendar_instance):
        """Test extra_state_attributes returns values from the most recent event."""
        older_event = Mock()
        older_event.summary = "Older event"
        older_event.description = "Older description"
        older_event.start = datetime.datetime(
            2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc
        )
        older_event.end = datetime.datetime(
            2024, 1, 1, 11, 0, tzinfo=datetime.timezone.utc
        )
        older_event.location = "older_frame.jpg,Older Camera"

        latest_event = Mock()
        latest_event.summary = "Latest event"
        latest_event.description = "Latest description"
        latest_event.start = datetime.datetime(
            2024, 1, 2, 10, 0, tzinfo=datetime.timezone.utc
        )
        latest_event.end = datetime.datetime(
            2024, 1, 2, 11, 0, tzinfo=datetime.timezone.utc
        )
        latest_event.location = "latest_frame.jpg,Front Door"

        # Deliberately unsorted to verify the property sorts by start descending.
        calendar_instance._events = [older_event, latest_event]

        attributes = calendar_instance.extra_state_attributes

        assert attributes["title"] == "Latest event"
        assert attributes["description"] == "Latest description"
        assert attributes["starts"] == latest_event.start
        assert attributes["ends"] == latest_event.end
        assert attributes["key_frame"] == "latest_frame.jpg"
        assert attributes["camera_name"] == "Front Door"

    def test_ensure_datetime_with_datetime(self, calendar_instance):
        """Test _ensure_datetime with datetime input."""
        dt = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        result = calendar_instance._ensure_datetime(dt)

        assert result == dt
        assert result.tzinfo is not None

    def test_ensure_datetime_with_date(self, calendar_instance):
        """Test _ensure_datetime with date input."""
        dt = datetime.date(2024, 1, 1)
        result = calendar_instance._ensure_datetime(dt)

        assert isinstance(result, datetime.datetime)
        assert result.tzinfo is not None

    def test_ensure_datetime_without_timezone(self, calendar_instance):
        """Test _ensure_datetime adds timezone if missing."""
        dt = datetime.datetime(2024, 1, 1, 12, 0, 0)
        result = calendar_instance._ensure_datetime(dt)

        assert result.tzinfo is not None
        assert result.tzinfo == datetime.timezone.utc

    @pytest.mark.asyncio
    async def test_async_update_empty(self, calendar_instance, mock_timeline):
        """Test async_update with no events."""
        mock_timeline.get_all_events = AsyncMock(return_value=[])

        await calendar_instance.async_update()

        assert calendar_instance._events == []

    @pytest.mark.parametrize("method", ["async_update", "async_get_events"])
    @patch.object(dt_util, "DEFAULT_TIME_ZONE", ZoneInfo("Europe/Berlin"))
    async def test_rollback_and_invalid_history(
        self, calendar_instance, mock_timeline, caplog, method
    ):
        """Real HA local-time validation must not discard a valid fold crossing."""
        berlin = ZoneInfo("Europe/Berlin")
        start = datetime.datetime(2026, 10, 25, 2, 59, 18, 801037, tzinfo=berlin)
        end = datetime.datetime(2026, 10, 25, 2, 1, 21, 387204,
                                tzinfo=berlin, fold=1)
        healthy_start = datetime.datetime(2026, 10, 25, 2, 30,
                                          tzinfo=berlin, fold=1)
        events = [
            MockTimelineEvent("rollback", "Rollback", start, end, "private text"),
            MockTimelineEvent("invalid", "Invalid", start,
                              end.replace(fold=0), "private text"),
            MockTimelineEvent("healthy", "Healthy", healthy_start,
                              healthy_start.astimezone(datetime.timezone.utc)
                              + datetime.timedelta(minutes=1), ""),
            MockTimelineEvent("missing", "Missing", None, None, ""),
            MockTimelineEvent("zero", "Zero", end, end, ""),
        ]
        original_endpoints = [(e.start, e.end) for e in events]
        mock_timeline.get_all_events.return_value = events
        if method == "async_update":
            await calendar_instance.async_update()
            result = calendar_instance._events
            # Local wall times would put rollback first; UTC puts healthy first.
            assert result[0].uid == "healthy"
            assert calendar_instance.extra_state_attributes["title"] == "Healthy"
        else:
            result = await calendar_instance.async_get_events(
                calendar_instance.hass,
                datetime.datetime(2026, 10, 25, 2, 0, tzinfo=berlin, fold=1),
                datetime.datetime(2026, 10, 25, 3, 0, tzinfo=berlin),
            )
        assert {e.uid for e in result} == {"rollback", "healthy", "zero"}
        rollback = next(e for e in result if e.uid == "rollback")
        assert isinstance(rollback, CalendarEvent)
        assert rollback.end - rollback.start == datetime.timedelta(
            minutes=2, seconds=2, microseconds=586167
        )
        assert rollback.as_dict()["start"] == start.astimezone(
            datetime.timezone.utc
        ).isoformat()
        assert [(e.start, e.end) for e in events] == original_endpoints
        mock_timeline.load_events.assert_awaited_once_with(purge_expired=False)
        mock_timeline.delete_event.assert_not_awaited()
        assert caplog.text.count("Skipping timeline event with invalid interval") == 2
        assert "private text" not in caplog.text

    async def test_rollback_query_uses_absolute_exclusive_boundaries(
        self, calendar_instance, mock_timeline
    ):
        berlin = ZoneInfo("Europe/Berlin")
        start = datetime.datetime(2026, 10, 25, 2, 59, tzinfo=berlin)
        end = datetime.datetime(2026, 10, 25, 2, 1, tzinfo=berlin, fold=1)
        mock_timeline.get_all_events.return_value = [
            MockTimelineEvent("rollback", "Rollback", start, end, "")
        ]
        # Query ends exactly at the event start: no overlap.
        result = await calendar_instance.async_get_events(
            calendar_instance.hass,
            datetime.datetime(2026, 10, 25, 2, 30, tzinfo=berlin), start,
        )
        assert result == []
        # Query starts exactly at the event end: no overlap.
        result = await calendar_instance.async_get_events(
            calendar_instance.hass, end,
            datetime.datetime(2026, 10, 25, 2, 30, tzinfo=berlin, fold=1),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_async_update_with_events(self, calendar_instance, mock_timeline):
        """Test async_update with events."""
        mock_events = [
            MockTimelineEvent(
                uid="1",
                title="Event 1",
                start=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
                end=datetime.datetime(2024, 1, 1, 1, 0, tzinfo=datetime.timezone.utc),
                description="Description 1"
            ),
            MockTimelineEvent(
                uid="2",
                title="Event 2",
                start=datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc),
                end=datetime.datetime(2024, 1, 2, 1, 0, tzinfo=datetime.timezone.utc),
                description="Description 2"
            )
        ]
        mock_timeline.get_all_events = AsyncMock(return_value=mock_events)

        await calendar_instance.async_update()

        assert len(calendar_instance._events) == 2
        assert all(isinstance(e, CalendarEvent) for e in calendar_instance._events)
        # Events should be sorted by start time in reverse
        assert calendar_instance._events[0].uid == "2"

    @pytest.mark.asyncio
    async def test_async_get_events_in_range(self, calendar_instance, mock_timeline):
        """Test async_get_events returns events within date range."""
        mock_events = [
            MockTimelineEvent(
                uid="1",
                title="Event 1",
                start=datetime.datetime(2024, 1, 15, tzinfo=datetime.timezone.utc),
                end=datetime.datetime(2024, 1, 15, 1, 0, tzinfo=datetime.timezone.utc),
                description="Description 1"
            )
        ]
        mock_timeline.get_all_events = AsyncMock(return_value=mock_events)

        start_date = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        end_date = datetime.datetime(2024, 1, 31, tzinfo=datetime.timezone.utc)

        result = await calendar_instance.async_get_events(
            calendar_instance.hass, start_date, end_date
        )

        assert len(result) == 1
        assert result[0].uid == "1"

    @pytest.mark.asyncio
    async def test_async_get_events_outside_range(self, calendar_instance, mock_timeline):
        """Test async_get_events excludes events outside date range."""
        mock_events = [
            MockTimelineEvent(
                uid="1",
                title="Event 1",
                start=datetime.datetime(2024, 2, 15, tzinfo=datetime.timezone.utc),
                end=datetime.datetime(2024, 2, 15, 1, 0, tzinfo=datetime.timezone.utc),
                description="Description 1"
            )
        ]
        mock_timeline.get_all_events = AsyncMock(return_value=mock_events)

        start_date = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        end_date = datetime.datetime(2024, 1, 31, tzinfo=datetime.timezone.utc)

        result = await calendar_instance.async_get_events(
            calendar_instance.hass, start_date, end_date
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_async_delete_event(self, calendar_instance, mock_timeline):
        """Test async_delete_event."""
        uid = "test_uid"
        calendar_instance.async_schedule_update_ha_state = Mock()

        await calendar_instance.async_delete_event(uid)

        mock_timeline.delete_event.assert_called_once_with(uid)
        calendar_instance.async_schedule_update_ha_state.assert_called_once_with(
            force_refresh=True
        )

        mock_timeline.delete_event.assert_called_once_with(uid)

    @pytest.mark.asyncio
    async def test_timeline_updates_are_debounced(self, calendar_instance):
        """Timeline-updated signals refresh via a debouncer, not a direct reload.

        A burst of camera events (one signal each) must coalesce into a single
        reload instead of one full DB reload per event, which would otherwise
        saturate the event loop.
        """
        calendar_instance.async_on_remove = Mock()
        with patch(
            "custom_components.llmvision.calendar.Debouncer"
        ) as mock_debouncer, patch(
            "custom_components.llmvision.calendar.async_dispatcher_connect"
        ) as mock_connect:
            await calendar_instance.async_added_to_hass()

        # A debouncer was created with the refresh coroutine and a non-zero cooldown.
        mock_debouncer.assert_called_once()
        _, kwargs = mock_debouncer.call_args
        assert kwargs["function"] == calendar_instance._async_refresh_state
        assert kwargs["cooldown"] > 0

        # The signal handler routes through the debouncer instead of refreshing inline.
        mock_connect.assert_called_once()
        signal_handler = mock_connect.call_args[0][2]
        signal_handler()
        calendar_instance.hass.async_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_refresh_state_forces_full_reload(self, calendar_instance):
        """The debounced refresh performs a force_refresh state update."""
        calendar_instance.async_update_ha_state = AsyncMock()

        await calendar_instance._async_refresh_state()

        calendar_instance.async_update_ha_state.assert_awaited_once_with(
            force_refresh=True
        )

    @pytest.mark.asyncio
    async def test_pending_refresh_cancelled_on_removal(self, calendar_instance):
        """Removing the entity shuts the debouncer down to cancel a pending refresh."""
        debouncer = AsyncMock()
        calendar_instance._refresh_debouncer = debouncer

        await calendar_instance.async_will_remove_from_hass()

        debouncer.async_shutdown.assert_awaited_once()


class TestCalendarAdvanced:
    """Advanced tests for Calendar class."""

    @pytest.fixture
    def mock_timeline_with_events(self):
        """Create a mock Timeline with events."""
        timeline = Mock()
        events = [
            MockTimelineEvent(
                uid=f"event_{i}",
                title=f"Event {i}",
                start=datetime.datetime(2024, 1, i+1, tzinfo=datetime.timezone.utc),
                end=datetime.datetime(2024, 1, i+1, 1, 0, tzinfo=datetime.timezone.utc),
                description=f"Description {i}"
            )
            for i in range(5)
        ]
        timeline.get_all_events = AsyncMock(return_value=events)
        timeline.load_events = AsyncMock()
        timeline.delete_event = AsyncMock()
        return timeline

    @pytest.fixture
    def calendar_with_events(self, mock_hass, mock_config_entry, mock_timeline_with_events):
        """Create a Calendar instance with events."""
        from custom_components.llmvision.calendar import Calendar
        
        with patch('custom_components.llmvision.calendar.Timeline', return_value=mock_timeline_with_events):
            with patch('os.makedirs'):
                with patch('os.path.join', return_value="/mock/path"):
                    calendar = Calendar(mock_hass, mock_config_entry)
                    return calendar

    @pytest.mark.asyncio
    async def test_async_update_sorts_events(self, calendar_with_events, mock_timeline_with_events):
        """Test async_update sorts events by start time in reverse."""
        await calendar_with_events.async_update()
        
        # Events should be sorted in reverse chronological order
        assert len(calendar_with_events._events) == 5
        assert calendar_with_events._events[0].uid == "event_4"
        assert calendar_with_events._events[-1].uid == "event_0"

    @pytest.mark.asyncio
    async def test_async_get_events_partial_overlap(self, calendar_with_events, mock_timeline_with_events):
        """Test async_get_events with partial date range overlap."""
        start_date = datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc)
        end_date = datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc)
        
        result = await calendar_with_events.async_get_events(
            calendar_with_events.hass, start_date, end_date
        )
        
        # Should include events 1, 2, 3 (indices 1, 2, 3)
        assert len(result) >= 2

    def test_ensure_datetime_preserves_timezone(self):
        """Test _ensure_datetime preserves existing timezone."""
        from custom_components.llmvision.calendar import Calendar
        
        mock_hass = Mock()
        mock_config_entry = Mock()
        
        with patch('custom_components.llmvision.calendar.Timeline'):
            with patch('os.makedirs'):
                with patch('os.path.join', return_value="/mock/path"):
                    calendar = Calendar(mock_hass, mock_config_entry)
                    
                    tz = datetime.timezone(datetime.timedelta(hours=5))
                    dt = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz)
                    
                    result = calendar._ensure_datetime(dt)
                    
                    assert result.tzinfo == tz
