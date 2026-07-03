"""Config flow for the Cohere-Transcribe-Diarize integration."""

from __future__ import annotations

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.helpers import aiohttp_client

from .const import CONF_API_TOKEN, CONF_HOST, CONF_PORT, DEFAULT_PORT, DOMAIN

DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HOST): str,
        vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
        vol.Optional(CONF_API_TOKEN, default=""): str,
    }
)


class CohereTranscribeDiarizeConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Ask for host/port/token and verify the management API answers."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors: dict[str, str] = {}
        if user_input is not None:
            host = user_input[CONF_HOST].strip()
            port = user_input[CONF_PORT]
            token = user_input.get(CONF_API_TOKEN, "").strip()

            await self.async_set_unique_id(f"{host}:{port}")
            self._abort_if_unique_id_configured()

            session = aiohttp_client.async_get_clientsession(self.hass)
            try:
                async with session.get(
                    f"http://{host}:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    response.raise_for_status()
                    await response.json()
            except Exception:
                errors["base"] = "cannot_connect"
            else:
                # Always exercise a token-protected endpoint: /health is open,
                # so a blank token against a protected server would otherwise
                # create an entry whose every real call 401s.
                try:
                    async with session.get(
                        f"http://{host}:{port}/settings",
                        headers={"X-API-Token": token} if token else {},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status == 401:
                            errors["base"] = "invalid_auth"
                except Exception:
                    errors["base"] = "cannot_connect"

                if not errors:
                    return self.async_create_entry(
                        title=f"Cohere-Transcribe-Diarize ({host})",
                        data={CONF_HOST: host, CONF_PORT: port, CONF_API_TOKEN: token},
                    )

        return self.async_show_form(
            step_id="user", data_schema=DATA_SCHEMA, errors=errors
        )
