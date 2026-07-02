"""Wyoming Transcribe integration.

Embeds the enrollment/management UI as a sidebar panel (iframe) and polls the
management API for status sensors (model readiness, enrolled speakers,
unrecognized pending voices).
"""

from __future__ import annotations

import logging
from datetime import timedelta

import aiohttp

from homeassistant.components import frontend
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_API_TOKEN,
    CONF_HOST,
    CONF_PORT,
    DOMAIN,
    PANEL_ICON,
    PANEL_TITLE,
    PANEL_URL_PATH,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["sensor"]
SCAN_INTERVAL = timedelta(seconds=60)
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


def base_url(entry: ConfigEntry) -> str:
    return f"http://{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    base = base_url(entry)
    token = entry.data.get(CONF_API_TOKEN) or None
    headers = {"X-API-Token": token} if token else {}
    session = aiohttp_client.async_get_clientsession(hass)

    async def _async_update() -> dict:
        data: dict = {"health": None, "pending": None, "speakers": None}
        try:
            async with session.get(f"{base}/health", timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                data["health"] = await response.json()
        except Exception as error:
            raise UpdateFailed(f"Cannot reach {base}/health: {error}") from error

        # Token-protected endpoints; sensors degrade gracefully without them.
        for key, path in (("pending", "/pending"), ("speakers", "/speakers")):
            try:
                async with session.get(
                    f"{base}{path}", headers=headers, timeout=REQUEST_TIMEOUT
                ) as response:
                    if response.status == 200:
                        data[key] = await response.json()
                    else:
                        _LOGGER.debug("%s returned %s", path, response.status)
            except Exception as error:
                _LOGGER.debug("Could not fetch %s: %s", path, error)
        return data

    coordinator = DataUpdateCoordinator(
        hass,
        _LOGGER,
        name=DOMAIN,
        update_method=_async_update,
        update_interval=SCAN_INTERVAL,
    )
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    # The iframe URL is opened by the *browser*, so the configured host must be
    # reachable from clients (LAN address), not just from Home Assistant.
    frontend.async_register_built_in_panel(
        hass,
        "iframe",
        sidebar_title=PANEL_TITLE,
        sidebar_icon=PANEL_ICON,
        frontend_url_path=PANEL_URL_PATH,
        config={"url": base},
        require_admin=True,
        update=True,
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        if not hass.data[DOMAIN]:
            frontend.async_remove_panel(hass, PANEL_URL_PATH)
    return unload_ok
