"""Wyoming Transcribe integration.

Embeds the enrollment/management UI as a sidebar panel (iframe), polls the
management API for status sensors, fires an event when a new unrecognized
voice lands in the pending buffer, and exposes services for the "who are
you?" enrollment flow (claim_utterance, set_role).
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import aiohttp
import voluptuous as vol
from aiohttp import web

from homeassistant.components import frontend
from homeassistant.components.http import HomeAssistantView, StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers import config_validation as cv
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
# Also the latency ceiling for the new-pending event that drives the
# "who are you?" flow — keep it short enough that the visitor is still around.
SCAN_INTERVAL = timedelta(seconds=15)
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
# Uploads (voice samples, backup archives) can take longer than status calls.
PROXY_TIMEOUT = aiohttp.ClientTimeout(total=120)

PANEL_STATIC_PATH = "/wyoming_transcribe_static"

EVENT_NEW_PENDING = f"{DOMAIN}_new_pending"

SERVICE_CLAIM_UTTERANCE = "claim_utterance"
SERVICE_CLAIM_LATEST = "claim_latest"
SERVICE_CHECK_LATEST_VOICE = "check_latest_voice"
SERVICE_SET_ROLE = "set_role"

CLAIM_UTTERANCE_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Required("utterance_id"): cv.string,
        vol.Optional("include_cluster", default=True): cv.boolean,
        vol.Optional("host"): cv.string,
    }
)
CLAIM_LATEST_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Optional("include_cluster", default=True): cv.boolean,
        vol.Optional("max_age_seconds", default=300): vol.Coerce(float),
        vol.Optional("anchor_utterance_id"): cv.string,
        vol.Optional("host"): cv.string,
    }
)
CHECK_LATEST_VOICE_SCHEMA = vol.Schema(
    {
        vol.Optional("min_utterances", default=3): vol.Coerce(int),
        vol.Optional("min_seconds", default=8): vol.Coerce(float),
        vol.Optional("max_age_seconds", default=300): vol.Coerce(float),
        vol.Optional("host"): cv.string,
    }
)
SET_ROLE_SCHEMA = vol.Schema(
    {
        vol.Required("name"): cv.string,
        vol.Required("role"): vol.In(["admin", "user", "guest"]),
        vol.Optional("host"): cv.string,
    }
)


def base_url(entry: ConfigEntry) -> str:
    return f"http://{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}"


def _first_runtime(hass: HomeAssistant) -> dict:
    """First configured server — used by the proxy/panel, which manages a
    single server by design (the panel has no server switcher yet)."""
    runtimes = hass.data.get(DOMAIN) or {}
    if not runtimes:
        raise HomeAssistantError("No Wyoming Transcribe server is configured")
    return next(iter(runtimes.values()))


def _runtime_for(hass: HomeAssistant, host: str | None) -> dict:
    """Resolve the target server for a service call.

    With one entry the choice is obvious; with several the caller must name
    the server via the optional `host` field — silently picking the first one
    could claim a pending voice from the wrong house/server.
    """
    runtimes = hass.data.get(DOMAIN) or {}
    if not runtimes:
        raise HomeAssistantError("No Wyoming Transcribe server is configured")
    if host:
        for runtime in runtimes.values():
            if runtime.get("host") == host:
                return runtime
        raise HomeAssistantError(f"No Wyoming Transcribe server configured for host '{host}'")
    if len(runtimes) > 1:
        hosts = ", ".join(sorted(r.get("host", "?") for r in runtimes.values()))
        raise HomeAssistantError(
            f"Multiple Wyoming Transcribe servers are configured ({hosts}); "
            "pass the 'host' field to select one"
        )
    return next(iter(runtimes.values()))


class WyomingTranscribeProxyView(HomeAssistantView):
    """Authenticated proxy to the management API.

    The panel talks to this view with the user's HA credentials
    (hass.fetchWithAuth); the view forwards to the server adding the API
    token, so the token never reaches the browser and port 8580 only needs
    to be reachable from the HA host.
    """

    url = "/api/wyoming_transcribe/proxy/{path:.+}"
    name = "api:wyoming_transcribe:proxy"
    requires_auth = True

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    async def _proxy(self, request: web.Request, path: str) -> web.Response:
        user = request.get("hass_user")
        if user is None or not user.is_admin:
            raise Unauthorized()

        runtime = _first_runtime(self.hass)
        headers = dict(runtime["headers"])
        if request.content_type:
            headers["Content-Type"] = request.headers.get("Content-Type", "")
        body = await request.read() if request.method in ("POST", "PUT") else None

        try:
            async with runtime["session"].request(
                request.method,
                f"{runtime['base']}/{path}",
                params=request.query,
                data=body,
                headers=headers,
                timeout=PROXY_TIMEOUT,
            ) as response:
                payload = await response.read()
                return web.Response(
                    body=payload,
                    status=response.status,
                    content_type=response.content_type,
                )
        except HomeAssistantError:
            raise
        except Exception as error:
            return web.json_response(
                {"detail": f"Wyoming Transcribe server unreachable: {error}"},
                status=502,
            )

    async def get(self, request: web.Request, path: str) -> web.Response:
        return await self._proxy(request, path)

    async def post(self, request: web.Request, path: str) -> web.Response:
        return await self._proxy(request, path)

    async def delete(self, request: web.Request, path: str) -> web.Response:
        return await self._proxy(request, path)


async def _api_post(runtime: dict, path: str, fields: dict) -> dict:
    form = aiohttp.FormData()
    for key, value in fields.items():
        form.add_field(key, str(value))
    async with runtime["session"].post(
        f"{runtime['base']}{path}",
        data=form,
        headers=runtime["headers"],
        timeout=REQUEST_TIMEOUT,
    ) as response:
        body = await response.text()
        if response.status >= 400:
            raise HomeAssistantError(
                f"Wyoming Transcribe API error {response.status} on {path}: {body}"
            )
    await runtime["coordinator"].async_request_refresh()
    try:
        return json.loads(body)
    except ValueError:
        return {}


def _async_register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_CLAIM_UTTERANCE):
        return

    async def handle_claim_utterance(call: ServiceCall) -> dict:
        """Enroll a pending utterance (and its voice cluster) as a person.

        Returns the server response ({claimed: [...], samples: [...]}) so an
        LLM agent can confirm what was actually enrolled.
        """
        runtime = _runtime_for(hass, call.data.get("host"))
        name = quote(call.data["name"], safe="")
        utterance_id = quote(call.data["utterance_id"], safe="")
        return await _api_post(
            runtime,
            f"/speakers/{name}/samples/from-utterance/{utterance_id}",
            {"include_cluster": "true" if call.data["include_cluster"] else "false"},
        )

    async def handle_claim_latest(call: ServiceCall) -> dict:
        """Enroll the newest unrecognized utterance (voice-anchored claim).

        Designed as an LLM tool for the "who are you?" flow. Pass
        anchor_utterance_id (from check_latest_voice) to claim exactly the
        voice that triggered the question — immune to another unknown voice
        interjecting between the answer and this call. Returns the server
        response ({claimed: [...], samples: [...]}).
        """
        runtime = _runtime_for(hass, call.data.get("host"))
        name = quote(call.data["name"], safe="")
        fields = {
            "include_cluster": "true" if call.data["include_cluster"] else "false",
            "max_age_seconds": call.data["max_age_seconds"],
        }
        if call.data.get("anchor_utterance_id"):
            fields["anchor_utterance_id"] = call.data["anchor_utterance_id"]
        return await _api_post(runtime, f"/speakers/{name}/samples/from-latest", fields)

    async def handle_check_latest_voice(call: ServiceCall) -> dict:
        """Regular-visitor check for the LLM "who are you?" flow.

        Returns stats of the newest unrecognized voice's cluster plus a ready
        should_ask verdict, so the agent only interrogates voices that have
        actually talked to the system a few times.
        """
        runtime = _runtime_for(hass, call.data.get("host"))
        silent = {
            "should_ask": False,
            "utterances": 0,
            "seconds": 0.0,
            "newest_age_seconds": None,
            "utterance_id": None,
            "reason": "no_pending_voice",
        }
        try:
            async with runtime["session"].get(
                f"{runtime['base']}/pending/latest-voice",
                headers=runtime["headers"],
                timeout=REQUEST_TIMEOUT,
            ) as response:
                if response.status == 404:
                    return silent
                if response.status >= 400:
                    raise HomeAssistantError(
                        f"latest-voice check failed: HTTP {response.status}"
                    )
                stats = await response.json()
        except HomeAssistantError:
            raise
        except Exception as error:
            raise HomeAssistantError(f"latest-voice check failed: {error}") from error

        fresh = stats["newest_age_seconds"] <= call.data["max_age_seconds"]
        regular = (
            stats["utterances"] >= call.data["min_utterances"]
            and stats["seconds"] >= call.data["min_seconds"]
        )
        reason = "ok" if (fresh and regular) else ("stale" if not fresh else "not_a_regular_yet")
        return {
            "should_ask": fresh and regular,
            "utterances": stats["utterances"],
            "seconds": stats["seconds"],
            "newest_age_seconds": stats["newest_age_seconds"],
            # Anchor for claim_latest: enrolls exactly this voice's cluster
            # even if another unknown voice speaks before the claim happens.
            "utterance_id": stats.get("utterance_id"),
            "reason": reason,
        }

    async def handle_set_role(call: ServiceCall) -> None:
        runtime = _runtime_for(hass, call.data.get("host"))
        name = quote(call.data["name"], safe="")
        await _api_post(runtime, f"/speakers/{name}/role", {"role": call.data["role"]})

    hass.services.async_register(
        DOMAIN,
        SERVICE_CLAIM_UTTERANCE,
        handle_claim_utterance,
        schema=CLAIM_UTTERANCE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CLAIM_LATEST,
        handle_claim_latest,
        schema=CLAIM_LATEST_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CHECK_LATEST_VOICE,
        handle_check_latest_voice,
        schema=CHECK_LATEST_VOICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN, SERVICE_SET_ROLE, handle_set_role, schema=SET_ROLE_SCHEMA
    )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    base = base_url(entry)
    token = entry.data.get(CONF_API_TOKEN) or None
    headers = {"X-API-Token": token} if token else {}
    session = aiohttp_client.async_get_clientsession(hass)

    # Baseline of pending clip ids; None until the first successful fetch so
    # a restart does not re-fire events for clips that were already waiting.
    known_pending_ids: set[str] | None = None

    async def _async_update() -> dict:
        nonlocal known_pending_ids
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

        if data["pending"] is not None:
            clusters = data["pending"].get("clusters", [])
            clips = [clip for cluster in clusters for clip in cluster.get("clips", [])]
            cluster_size = {
                clip["id"]: len(cluster.get("clips", []))
                for cluster in clusters
                for clip in cluster.get("clips", [])
            }
            current_ids = {clip["id"] for clip in clips}
            if known_pending_ids is not None:
                for clip in clips:
                    if clip["id"] not in known_pending_ids:
                        hass.bus.async_fire(
                            EVENT_NEW_PENDING,
                            {
                                "utterance_id": clip["id"],
                                "text": clip.get("text"),
                                "seconds": clip.get("seconds"),
                                "created": clip.get("created"),
                                # Size of this voice's cluster: 1 = one-off
                                # visitor, more = a regular worth identifying.
                                "voice_utterances": cluster_size.get(clip["id"], 1),
                            },
                        )
            known_pending_ids = current_ids

        return data

    coordinator = DataUpdateCoordinator(
        hass,
        _LOGGER,
        name=DOMAIN,
        update_method=_async_update,
        update_interval=SCAN_INTERVAL,
    )
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {
        "coordinator": coordinator,
        "base": base,
        "host": entry.data[CONF_HOST],
        "headers": headers,
        "session": session,
    }

    _async_register_services(hass)

    # Serve the panel module and register it as a custom sidebar panel. The
    # panel talks to the server exclusively through the authenticated proxy
    # view, so browsers never need direct access to port 8580.
    if not hass.data.get(f"{DOMAIN}_http_registered"):
        await hass.http.async_register_static_paths(
            [
                StaticPathConfig(
                    PANEL_STATIC_PATH,
                    str(Path(__file__).parent / "frontend"),
                    cache_headers=False,
                )
            ]
        )
        hass.http.register_view(WyomingTranscribeProxyView(hass))
        hass.data[f"{DOMAIN}_http_registered"] = True

    frontend.async_register_built_in_panel(
        hass,
        "custom",
        sidebar_title=PANEL_TITLE,
        sidebar_icon=PANEL_ICON,
        frontend_url_path=PANEL_URL_PATH,
        config={
            "_panel_custom": {
                "name": "wyoming-transcribe-panel",
                "module_url": f"{PANEL_STATIC_PATH}/panel.js",
                "embed_iframe": False,
                "trust_external": False,
            }
        },
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
            for service in (
                SERVICE_CLAIM_UTTERANCE,
                SERVICE_CLAIM_LATEST,
                SERVICE_CHECK_LATEST_VOICE,
                SERVICE_SET_ROLE,
            ):
                if hass.services.has_service(DOMAIN, service):
                    hass.services.async_remove(DOMAIN, service)
    return unload_ok
