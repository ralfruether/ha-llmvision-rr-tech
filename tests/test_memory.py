"""Unit tests for memory.py module."""
import base64
import io
import json
import os

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
from homeassistant.exceptions import ServiceValidationError
from custom_components.llmvision.memory import (
    MAX_MEMORY_IMAGE_BYTES,
    MAX_MEMORY_IMAGE_PIXELS,
    MAX_MEMORY_IMAGES_BYTES,
    Memory,
)
from custom_components.llmvision.const import (
    CONF_MEMORY_IMAGES_CACHE_KEY,
    CONF_MEMORY_IMAGES_ENCODED,
    CONF_MEMORY_PATHS,
    CONF_MEMORY_STRINGS,
    CONF_PROVIDER,
    CONF_RESIZE_MEMORY_IMAGES,
    DOMAIN,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TITLE_PROMPT,
)


class TestMemory:
    """Test Memory class."""

    def test_init_without_entry(self, mock_hass):
        """Test Memory initialization without config entry."""
        memory = Memory(mock_hass, strings=["test"], paths=[])
        
        assert memory.hass == mock_hass
        assert memory.memory_strings == ["test"]
        assert memory.memory_paths == []
        assert memory.memory_images == []
        assert memory._system_prompt == DEFAULT_SYSTEM_PROMPT
        assert memory._title_prompt == DEFAULT_TITLE_PROMPT

    def test_init_with_custom_prompt(self, mock_hass):
        """Test Memory initialization with custom system prompt."""
        custom_prompt = "Custom prompt"
        memory = Memory(mock_hass, system_prompt=custom_prompt)
        
        assert memory._system_prompt == custom_prompt

    def test_init_with_entry(self, mock_hass, mock_config_entry):
        """Test Memory initialization with config entry."""
        mock_hass.config_entries.async_entries = Mock(return_value=[mock_config_entry])
        
        memory = Memory(mock_hass)
        
        assert memory.entry == mock_config_entry
        assert memory._system_prompt == "Test system prompt"
        assert memory._title_prompt == "Test title prompt"

    def test_system_prompt_property(self, mock_hass):
        """Test system_prompt property."""
        memory = Memory(mock_hass)
        assert memory.system_prompt.startswith("System prompt: ")

    def test_title_prompt_property(self, mock_hass):
        """Test title_prompt property."""
        memory = Memory(mock_hass)
        assert memory.title_prompt == DEFAULT_TITLE_PROMPT

    def test_get_memory_images_openai_empty(self, mock_hass):
        """Test _get_memory_images with OpenAI format and no images."""
        memory = Memory(mock_hass)
        result = memory._get_memory_images(memory_type="OpenAI")
        
        assert result == []

    def test_get_memory_images_openai_with_images(self, mock_hass):
        """Test _get_memory_images with OpenAI format and images."""
        memory = Memory(mock_hass, strings=["Image 1"], paths=[])
        memory.memory_images = ["base64_encoded_image"]
        
        result = memory._get_memory_images(memory_type="OpenAI")
        
        assert len(result) == 3  # prompt + text + image
        assert result[0]["type"] == "text"
        assert "reference" in result[0]["text"]
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Image 1:"
        assert result[2]["type"] == "image_url"

    def test_get_memory_images_anthropic(self, mock_hass):
        """Test _get_memory_images with Anthropic format."""
        memory = Memory(mock_hass, strings=["Test"], paths=[])
        memory.memory_images = ["base64_image"]
        
        result = memory._get_memory_images(memory_type="Anthropic")
        
        assert len(result) == 3
        assert result[2]["type"] == "image"
        assert result[2]["source"]["type"] == "base64"

    def test_get_memory_images_google(self, mock_hass):
        """Test _get_memory_images with Google format."""
        memory = Memory(mock_hass, strings=["Test"], paths=[])
        memory.memory_images = ["base64_image"]
        
        result = memory._get_memory_images(memory_type="Google")
        
        assert len(result) == 3
        assert "inline_data" in result[2]

    def test_get_memory_images_ollama(self, mock_hass):
        """Test _get_memory_images with Ollama format."""
        memory = Memory(mock_hass, strings=["Test"], paths=[])
        memory.memory_images = ["base64_image"]
        
        result = memory._get_memory_images(memory_type="Ollama")
        
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert "images" in result[1]

    def test_get_memory_images_aws(self, mock_hass):
        """Test _get_memory_images with AWS format."""
        import base64
        memory = Memory(mock_hass, strings=["Test"], paths=[])
        # Use valid base64 encoded data
        memory.memory_images = [base64.b64encode(b"test_image_data").decode("utf-8")]
        
        result = memory._get_memory_images(memory_type="AWS")
        
        assert len(result) == 3
        assert "image" in result[2]

    def test_get_memory_images_unknown_type(self, mock_hass):
        """Test _get_memory_images with unknown memory type."""
        memory = Memory(mock_hass)
        result = memory._get_memory_images(memory_type="Unknown")
        
        assert result == []

    def test_str_representation(self, mock_hass):
        """Test __str__ method."""
        memory = Memory(mock_hass, strings=["test"], paths=["path"])
        memory.memory_images = ["img1", "img2"]
        
        result = str(memory)
        
        assert "Memory" in result
        assert "['test']" in result
        assert "['path']" in result
        assert "2" in result

    def test_find_memory_entry_not_found(self, mock_hass):
        """Test _find_memory_entry when no Settings entry exists."""
        memory = Memory(mock_hass)
        assert memory.entry is None

    def test_find_memory_entry_found(self, mock_hass, mock_config_entry):
        """Test _find_memory_entry when Settings entry exists."""
        mock_hass.config_entries.async_entries = Mock(return_value=[mock_config_entry])
        
        memory = Memory(mock_hass)
        
        assert memory.entry == mock_config_entry



class TestMemoryAdvanced:
    """Advanced tests for Memory class."""

    def test_get_memory_images_openai_legacy(self, mock_hass):
        """Test _get_memory_images with OpenAI-legacy format."""
        memory = Memory(mock_hass, strings=["Test"], paths=[])
        memory.memory_images = ["base64_image"]
        
        result = memory._get_memory_images(memory_type="OpenAI-legacy")
        
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[2]["type"] == "image_url"

    def test_get_memory_images_multiple_images(self, mock_hass):
        """Test _get_memory_images with multiple images."""
        memory = Memory(mock_hass, strings=["Image 1", "Image 2"], paths=[])
        memory.memory_images = ["base64_1", "base64_2"]
        
        result = memory._get_memory_images(memory_type="OpenAI")
        
        # Should have: prompt + (text + image) * 2 = 5 items
        assert len(result) == 5

    def test_memory_with_empty_strings(self, mock_hass):
        """Test Memory with empty strings list."""
        memory = Memory(mock_hass, strings=[], paths=[])
        
        assert memory.memory_strings == []
        assert memory.memory_paths == []

    def test_memory_with_paths_no_images(self, mock_hass):
        """Test Memory with paths but no encoded images."""
        memory = Memory(mock_hass, strings=["test"], paths=["/path/to/image.jpg"])
        
        assert len(memory.memory_paths) == 1
        assert len(memory.memory_images) == 0

    def test_system_prompt_includes_prefix(self, mock_hass):
        """Test system_prompt property includes prefix."""
        memory = Memory(mock_hass)
        
        result = memory.system_prompt
        
        assert result.startswith("System prompt: ")

    def test_title_prompt_no_prefix(self, mock_hass):
        """Test title_prompt property has no prefix."""
        memory = Memory(mock_hass)
        
        result = memory.title_prompt
        
        assert not result.startswith("System prompt:")
        assert isinstance(result, str)

    def test_find_memory_entry_multiple_entries(self, mock_hass, mock_config_entry):
        """Test _find_memory_entry with multiple entries."""
        other_entry = Mock()
        other_entry.data = {"provider": "OpenAI"}
        
        mock_hass.config_entries.async_entries = Mock(
            return_value=[other_entry, mock_config_entry]
        )
        
        memory = Memory(mock_hass)
        
        assert memory.entry == mock_config_entry

    def test_memory_initialization_with_all_params(self, mock_hass):
        """Test Memory initialization with all parameters."""
        memory = Memory(
            mock_hass,
            strings=["test1", "test2"],
            paths=["/path1", "/path2"],
            system_prompt="Custom prompt"
        )
        
        assert len(memory.memory_strings) == 2
        assert len(memory.memory_paths) == 2
        assert memory._system_prompt == "Custom prompt"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("resize_memory_images", "expected_size"),
        [(True, (512, 128)), (False, (1024, 256))],
    )
    async def test_encode_images_respects_resize_setting(
        self, mock_hass, tmp_path, resize_memory_images, expected_size
    ):
        """Memory encoding should optionally preserve source dimensions."""
        source = tmp_path / "reference.jpg"
        Image.new("RGB", (1024, 256), color="navy").save(source, format="JPEG")
        mock_hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        memory = Memory(mock_hass)
        memory.resize_memory_images = resize_memory_images

        encoded = await memory._encode_images([str(source)])

        with Image.open(io.BytesIO(base64.b64decode(encoded[0]))) as image:
            assert image.size == expected_size

    @pytest.mark.asyncio
    async def test_update_memory_persists_and_reuses_encoded_cache(
        self, mock_hass, tmp_path
    ):
        """Encoded images should use the canonical cache key and be reusable."""
        first = tmp_path / "first.jpg"
        second = tmp_path / "second.jpg"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        entry = Mock(
            data={
                CONF_PROVIDER: "Settings",
                CONF_MEMORY_PATHS: [str(first), str(second)],
                CONF_MEMORY_STRINGS: ["First", "Second"],
                CONF_RESIZE_MEMORY_IMAGES: True,
                CONF_MEMORY_IMAGES_ENCODED: [],
            }
        )
        mock_hass.config_entries.async_entries.return_value = [entry]
        mock_hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        memory = Memory(mock_hass)
        memory._encode_images = AsyncMock(return_value=["encoded-1", "encoded-2"])

        await memory._update_memory()

        updated = mock_hass.config_entries.async_update_entry.call_args.kwargs["data"]
        assert updated[CONF_MEMORY_IMAGES_ENCODED] == ["encoded-1", "encoded-2"]
        cache_key = json.loads(updated[CONF_MEMORY_IMAGES_CACHE_KEY])
        assert cache_key == {
            "resize": True,
            "sources": [
                [os.path.realpath(first), first.stat().st_size, first.stat().st_mtime_ns],
                [
                    os.path.realpath(second),
                    second.stat().st_size,
                    second.stat().st_mtime_ns,
                ],
            ],
        }
        assert "images" not in updated

        entry.data = updated
        cached_memory = Memory(mock_hass)
        cached_memory._encode_images = AsyncMock()
        await cached_memory._update_memory()
        cached_memory._encode_images.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_memory_invalidates_cache_when_resize_mode_changes(
        self, mock_hass, tmp_path
    ):
        """Changing only resize mode should regenerate equally sized caches."""
        source = tmp_path / "reference.jpg"
        source.write_bytes(b"reference")
        entry = Mock(
            data={
                CONF_PROVIDER: "Settings",
                CONF_MEMORY_PATHS: [str(source)],
                CONF_MEMORY_STRINGS: ["Reference"],
                CONF_RESIZE_MEMORY_IMAGES: True,
                CONF_MEMORY_IMAGES_ENCODED: [],
            }
        )
        mock_hass.config_entries.async_entries.return_value = [entry]
        mock_hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        memory = Memory(mock_hass)
        memory._encode_images = AsyncMock(return_value=["resized"])

        await memory._update_memory()

        updated = mock_hass.config_entries.async_update_entry.call_args.kwargs["data"]
        updated[CONF_RESIZE_MEMORY_IMAGES] = False
        entry.data = updated
        original_memory = Memory(mock_hass)
        original_memory._encode_images = AsyncMock(return_value=["original-size"])
        await original_memory._update_memory()

        original_memory._encode_images.assert_awaited_once_with(
            [str(source)]
        )
        refreshed = mock_hass.config_entries.async_update_entry.call_args.kwargs["data"]
        assert refreshed[CONF_MEMORY_IMAGES_ENCODED] == ["original-size"]

    @pytest.mark.asyncio
    async def test_update_memory_invalidates_cache_when_source_changes(
        self, mock_hass, tmp_path
    ):
        """Replacing an image at the same path should regenerate its cache."""
        source = tmp_path / "reference.jpg"
        source.write_bytes(b"first")
        entry = Mock(
            data={
                CONF_PROVIDER: "Settings",
                CONF_MEMORY_PATHS: [str(source)],
                CONF_MEMORY_STRINGS: ["Reference"],
                CONF_RESIZE_MEMORY_IMAGES: True,
                CONF_MEMORY_IMAGES_ENCODED: [],
            }
        )
        mock_hass.config_entries.async_entries.return_value = [entry]
        mock_hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        memory = Memory(mock_hass)
        memory._encode_images = AsyncMock(return_value=["first-encoded"])
        await memory._update_memory()
        entry.data = mock_hass.config_entries.async_update_entry.call_args.kwargs[
            "data"
        ]

        source.write_bytes(b"second-content")
        refreshed_memory = Memory(mock_hass)
        refreshed_memory._encode_images = AsyncMock(return_value=["second-encoded"])
        await refreshed_memory._update_memory()

        refreshed_memory._encode_images.assert_awaited_once_with([str(source)])

    @pytest.mark.asyncio
    async def test_encode_images_rejects_excessive_pixel_count(self, mock_hass):
        """Oversized decoded images should be rejected before loading pixels."""
        image = MagicMock()
        image.__enter__.return_value = image
        image.size = (MAX_MEMORY_IMAGE_PIXELS + 1, 1)
        mock_hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        memory = Memory(mock_hass)

        with patch(
            "custom_components.llmvision.memory.Image.open", return_value=image
        ), pytest.raises(ServiceValidationError, match="exceeds"):
            await memory._encode_images(["/media/oversized.jpg"])

        image.load.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_images_rejects_excessive_jpeg_size(
        self, mock_hass, tmp_path
    ):
        """Encoded memory images should respect the provider payload guard."""
        source = tmp_path / "reference.jpg"
        Image.new("RGB", (16, 16), color="navy").save(source, format="JPEG")
        mock_hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        memory = Memory(mock_hass)

        with patch(
            "custom_components.llmvision.memory.MAX_MEMORY_IMAGE_BYTES", 1
        ), pytest.raises(ServiceValidationError, match="Encoded memory image exceeds"):
            await memory._encode_images([str(source)])

    @pytest.mark.asyncio
    async def test_encode_images_rejects_excessive_aggregate_size(
        self, mock_hass, tmp_path
    ):
        """The combined JPEG payload should remain within its configured limit."""
        source = tmp_path / "reference.jpg"
        Image.new("RGB", (16, 16), color="navy").save(source, format="JPEG")
        mock_hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        memory = Memory(mock_hass)

        with patch(
            "custom_components.llmvision.memory.MAX_MEMORY_IMAGES_BYTES", 1
        ), pytest.raises(ServiceValidationError, match="combined size limit"):
            await memory._encode_images([str(source)])

    def test_memory_image_limits_are_conservative(self):
        """Deployed memory limits should remain explicit and reviewable."""
        assert MAX_MEMORY_IMAGE_PIXELS == 40_000_000
        assert MAX_MEMORY_IMAGE_BYTES == 15 * 1024 * 1024
        assert MAX_MEMORY_IMAGES_BYTES == 20 * 1024 * 1024
