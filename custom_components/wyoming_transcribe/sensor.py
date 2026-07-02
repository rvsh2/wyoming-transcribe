"""Status sensors for the Wyoming Transcribe integration."""

from __future__ import annotations

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import base_url
from .const import DOMAIN


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities(
        [
            ModelStatusSensor(coordinator, entry),
            EnrolledSpeakersSensor(coordinator, entry),
            PendingVoicesSensor(coordinator, entry),
        ]
    )


class WyomingTranscribeSensor(CoordinatorEntity, SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator, entry: ConfigEntry, key: str) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Wyoming Transcribe",
            manufacturer="wyoming-transcribe",
            configuration_url=base_url(entry),
        )


class ModelStatusSensor(WyomingTranscribeSensor):
    _attr_translation_key = "model_status"
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "model_status")

    @property
    def native_value(self):
        health = self.coordinator.data.get("health") or {}
        return health.get("status")

    @property
    def extra_state_attributes(self):
        health = self.coordinator.data.get("health") or {}
        return {
            "model": health.get("model"),
            "device": health.get("device"),
            "ready": health.get("ready"),
        }


class EnrolledSpeakersSensor(WyomingTranscribeSensor):
    _attr_translation_key = "enrolled_speakers"
    _attr_icon = "mdi:account-group"

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "enrolled_speakers")

    @property
    def native_value(self):
        speakers = self.coordinator.data.get("speakers")
        if speakers is None:
            return None
        return len(speakers.get("speakers", []))

    @property
    def extra_state_attributes(self):
        speakers = self.coordinator.data.get("speakers") or {}
        return {
            "names": [s.get("name") for s in speakers.get("speakers", [])],
            "roles": {
                s.get("name"): s.get("role") for s in speakers.get("speakers", [])
            },
        }


class PendingVoicesSensor(WyomingTranscribeSensor):
    _attr_translation_key = "pending_voices"
    _attr_icon = "mdi:account-question"

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry, "pending_voices")

    @property
    def native_value(self):
        pending = self.coordinator.data.get("pending")
        if pending is None:
            return None
        return pending.get("count", 0)

    @property
    def extra_state_attributes(self):
        pending = self.coordinator.data.get("pending") or {}
        return {"clusters": len(pending.get("clusters", []))}
