from .const import (
    DOMAIN,
    CONF_MEMORY_PATHS,
    CONF_MEMORY_IMAGES_CACHE_KEY,
    CONF_MEMORY_IMAGES_ENCODED,
    CONF_MEMORY_STRINGS,
    CONF_RESIZE_MEMORY_IMAGES,
    CONF_SYSTEM_PROMPT,
    CONF_TITLE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TITLE_PROMPT,
)
import base64
import io
import json
import os
from PIL import Image
import logging
from homeassistant.exceptions import ServiceValidationError

_LOGGER = logging.getLogger(__name__)

MAX_MEMORY_IMAGE_PIXELS = 40_000_000
MAX_MEMORY_IMAGE_BYTES = 15 * 1024 * 1024
MAX_MEMORY_IMAGES_BYTES = 20 * 1024 * 1024


class Memory:
    def __init__(self, hass, strings=[], paths=[], system_prompt=None):
        self.hass = hass
        self.entry = self._find_memory_entry()
        if self.entry is None:

            self._system_prompt = (
                system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
            )
            self._title_prompt = DEFAULT_TITLE_PROMPT
            self.memory_strings = strings
            self.memory_paths = paths
            self.memory_images = []
            self.memory_images_cache_key = ""
            self.resize_memory_images = True

        else:
            self._system_prompt = (
                system_prompt
                if system_prompt
                else self.entry.data.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
            )
            self._title_prompt = self.entry.data.get(
                CONF_TITLE_PROMPT, DEFAULT_TITLE_PROMPT
            )
            self.memory_strings = self.entry.data.get(CONF_MEMORY_STRINGS, strings)
            self.memory_paths = self.entry.data.get(CONF_MEMORY_PATHS, paths)
            self.memory_images = self.entry.data.get(CONF_MEMORY_IMAGES_ENCODED, [])
            self.memory_images_cache_key = self.entry.data.get(
                CONF_MEMORY_IMAGES_CACHE_KEY, ""
            )
            self.resize_memory_images = self.entry.data.get(
                CONF_RESIZE_MEMORY_IMAGES, True
            )

        _LOGGER.debug(self)

    def _get_memory_images(self, memory_type="OpenAI") -> list:
        content = []
        memory_prompt = "The following images along with descriptions serve as reference. They are not to be mentioned in the response."

        if memory_type == "OpenAI":
            if self.memory_images:
                content.append({"type": "text", "text": memory_prompt})
            for image in self.memory_images:
                tag = self.memory_strings[self.memory_images.index(image)]

                content.append({"type": "text", "text": tag + ":"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        elif memory_type == "OpenAI-legacy":
            if self.memory_images:
                content.append({"type": "text", "text": memory_prompt})
            for image in self.memory_images:
                tag = self.memory_strings[self.memory_images.index(image)]

                content.append({"type": "text", "text": tag + ":"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        elif memory_type == "Ollama":
            if self.memory_images:
                content.append({"role": "user", "content": memory_prompt})
            for image in self.memory_images:
                tag = self.memory_strings[self.memory_images.index(image)]

                content.append(
                    {"role": "user", "content": tag + ":", "images": [image]}
                )

        elif memory_type == "Anthropic":
            if self.memory_images:
                content.append({"type": "text", "text": memory_prompt})
            for image in self.memory_images:
                tag = self.memory_strings[self.memory_images.index(image)]

                content.append({"type": "text", "text": tag + ":"})
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": f"{image}",
                        },
                    }
                )
        elif memory_type == "Google":
            if self.memory_images:
                content.append({"text": memory_prompt})
            for image in self.memory_images:
                tag = self.memory_strings[self.memory_images.index(image)]

                content.append({"text": tag + ":"})
                content.append(
                    {"inline_data": {"mime_type": "image/jpeg", "data": image}}
                )
        elif memory_type == "AWS":
            if self.memory_images:
                content.append({"text": memory_prompt})
            for image in self.memory_images:
                tag = self.memory_strings[self.memory_images.index(image)]

                content.append({"text": tag + ":"})
                content.append(
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": base64.b64decode(image)},
                        }
                    }
                )
        else:
            return []

        return content

    @property
    def system_prompt(self) -> str:
        return "System prompt: " + self._system_prompt

    @property
    def title_prompt(self) -> str:
        return self._title_prompt

    def _find_memory_entry(self):
        memory_entry = None
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            # Check if the config entry is empty
            if entry.data["provider"] == "Settings":
                memory_entry = entry
                break

        return memory_entry

    async def _encode_images(self, image_paths):
        """Encode images as base64"""
        encoded_images = []
        encoded_bytes = 0

        def _encode_image(image_path):
            with Image.open(image_path) as img:
                width, height = img.size
                if width * height > MAX_MEMORY_IMAGE_PIXELS:
                    raise ServiceValidationError(
                        f"Memory image exceeds {MAX_MEMORY_IMAGE_PIXELS:,} pixels: "
                        f"{image_path}"
                    )
                img.load()
                if self.resize_memory_images:
                    aspect_ratio = width / height
                    if aspect_ratio > 1:
                        new_width = 512
                        new_height = int(512 / aspect_ratio)
                    else:
                        new_height = 512
                        new_width = int(512 * aspect_ratio)
                    img = img.resize((new_width, new_height))

                # Convert Memory Images to RGB mode if needed
                if img.mode == "RGBA":
                    img = img.convert("RGB")

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="JPEG")
                image_bytes = img_byte_arr.getvalue()
                if len(image_bytes) > MAX_MEMORY_IMAGE_BYTES:
                    raise ServiceValidationError(
                        f"Encoded memory image exceeds {MAX_MEMORY_IMAGE_BYTES:,} "
                        f"bytes: {image_path}"
                    )
                return (
                    base64.b64encode(image_bytes).decode("utf-8"),
                    len(image_bytes),
                )

        for image_path in image_paths:
            base64_image, image_size = await self.hass.loop.run_in_executor(
                None, _encode_image, image_path
            )
            encoded_bytes += image_size
            if encoded_bytes > MAX_MEMORY_IMAGES_BYTES:
                raise ServiceValidationError(
                    "Encoded memory images exceed the combined size limit of "
                    f"{MAX_MEMORY_IMAGES_BYTES:,} bytes"
                )
            encoded_images.append(base64_image)

        return encoded_images

    async def _update_memory(self):
        """Manage encoded images"""
        # check if len(memory_paths) != len(memory_images)
        if self.entry is None:
            _LOGGER.debug("Memory entry not found; skipping memory update.")
            return

        def _source_metadata():
            metadata = []
            for path in self.memory_paths:
                try:
                    stat = os.stat(path)
                    metadata.append(
                        [os.path.realpath(path), stat.st_size, stat.st_mtime_ns]
                    )
                except OSError:
                    metadata.append([path, None, None])
            return metadata

        source_metadata = await self.hass.loop.run_in_executor(None, _source_metadata)
        cache_key = json.dumps(
            {
                "sources": source_metadata,
                "resize": self.resize_memory_images,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if (
            len(self.memory_paths) != len(self.memory_images)
            or self.memory_images_cache_key != cache_key
        ):
            self.memory_images = await self._encode_images(self.memory_paths)
            self.memory_images_cache_key = cache_key

            memory = self.entry.data.copy()
            memory[CONF_MEMORY_IMAGES_ENCODED] = self.memory_images
            memory[CONF_MEMORY_IMAGES_CACHE_KEY] = self.memory_images_cache_key
            memory.pop("images", None)
            self.hass.config_entries.async_update_entry(self.entry, data=memory)

    def __str__(self):
        return f"Memory({self.memory_strings}, {self.memory_paths}, {len(self.memory_images)})"
