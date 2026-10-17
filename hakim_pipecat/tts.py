"""Hakim TTS service for Pipecat.

Speaks the public `WSS /v1/audio/speech/stream` API directly (see
https://tryhakim.ai/docs) via the `websockets` library Pipecat already
depends on — no dependency on the `hakim` Python package.

Wire contract implemented here:
  - Client -> server: `session.update`, `speech.create`, `session.close`.
  - Server -> client: `session.created`, `speech.started`, binary
    PCM-S16LE @ 24 kHz mono chunks, `speech.done`, `session.usage`, `error`.
  - `response_format` / `sample_rate` are locked to `pcm` / `24000` on this
    surface — the service's declared sample rate is fixed to match.

Base class: Hakim's realtime TTS has no documented mid-utterance cancel
message and no word-level timestamps, so this subclasses
`InterruptibleTTSService` — the framework's base for exactly that shape
(handles interruption by reconnecting rather than sending a cancel frame).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.settings import TTSSettings

# NOT_GIVEN, the sentinel type and is_given moved to pipecat.utils.types in
# pipecat 1.8, where the type also lost its underscore. Importing them from
# pipecat.services.settings raises ImportError on 1.8 and later.
try:
    from pipecat.utils.types import NOT_GIVEN, is_given
    from pipecat.utils.types import NotGiven as _NotGiven
except ImportError:  # pipecat < 1.8
    from pipecat.services.settings import NOT_GIVEN, _NotGiven, is_given
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from ._common import HakimStreamError, Region, resolve_api_key, resolve_ws_url

SAMPLE_RATE = 24000
DEFAULT_MODEL = "hakim-fast-v1"


@dataclass
class HakimTTSSettings(TTSSettings):
    """Runtime-updatable settings for `HakimTTSService`.

    Parameters:
        cfg: Classifier-free-guidance weight, `0.0`-`10.0`.
        voice_prompt: Free-form voice-character description. Only honoured
            on `model="hakim-v3"` — dropped elsewhere.
    """

    cfg: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice_prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class HakimTTSService(InterruptibleTTSService):
    """Hakim realtime text-to-speech service for Pipecat.

    Example::

        tts = HakimTTSService(api_key=os.environ["HAKIM_API_KEY"], voice="cmokbc2b1001pvu39wmj61b7h")
    """

    Settings = HakimTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        voice: str,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cfg: float = 3.0,
        voice_prompt: str | None = None,
        region: Region = "auto",
        base_url: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Hakim TTS service.

        Args:
            voice: Hakim voice id or slug (e.g. `"cmokbc2b1001pvu39wmj61b7h"`).
                Required — there is no bundled default voice.
            api_key: Hakim API key (`hk_live_...`). Falls back to the
                `HAKIM_API_KEY` env var.
            model: `"hakim-fast-v1"` (default, sub-120ms TTFB), `"hakim-v2"`
                (higher quality + non-verbal tags), or `"hakim-v3"` (adds
                free-form `voice_prompt` control).
            cfg: Classifier-free-guidance weight, `0.0`-`10.0` (default `3.0`).
            voice_prompt: Free-form voice-character description, only
                honoured on `model="hakim-v3"` — dropped elsewhere.
            region: `"auto" | "de" | "uae" | "ksa"`.
            base_url: Overrides the region table entirely (staging /
                self-hosted). Takes precedence over `region`.
            settings: Runtime-updatable `HakimTTSSettings`. Values here take
                precedence over the shorthand kwargs above.
            **kwargs: Additional arguments passed to `InterruptibleTTSService`.
        """
        default_settings = self.Settings(model=model, voice=voice, cfg=cfg, voice_prompt=voice_prompt)
        if settings:
            for f in ("model", "voice", "cfg", "voice_prompt"):
                value = getattr(settings, f)
                if is_given(value):
                    setattr(default_settings, f, value)

        super().__init__(sample_rate=SAMPLE_RATE, settings=default_settings, **kwargs)

        self._api_key = resolve_api_key(api_key)
        self._region = region
        self._base_url = base_url

        self._websocket = None
        self._receive_task = None
        # Hakim's protocol has no client-supplied context id on binary audio
        # frames — only one utterance is ever in flight per socket, so we
        # track "whichever context is currently playing" with a single
        # pointer rather than the multi-context routing Cartesia needs.
        self._active_context_id: str | None = None

    @property
    def provider(self) -> str:
        """Get the name of the TTS provider.

        Returns:
            The provider name, `"Hakim"`.
        """
        return "Hakim"

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics.
        """
        return True

    async def start(self, frame: StartFrame) -> None:
        """Start the service and connect the websocket.

        Args:
            frame: The start frame that triggers service initialization.
        """
        await super().start(frame)
        await self._connect()

    async def _connect(self) -> None:
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self) -> None:
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self}: connecting to Hakim TTS")
            url = resolve_ws_url("/v1/audio/speech/stream", region=self._region, base_url=self._base_url)
            self._websocket = await websocket_connect(
                url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            session_update = {
                "model": self._settings.model,
                "voice": self._settings.voice,
                "cfg": self._settings.cfg,
            }
            if is_given(self._settings.voice_prompt) and self._settings.voice_prompt:
                session_update["voice_prompt"] = self._settings.voice_prompt
            await self._websocket.send(json.dumps({"type": "session.update", "session": session_update}))
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to Hakim: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self) -> None:
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Hakim TTS")
                await self._websocket.send(json.dumps({"type": "session.close"}))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            self._active_context_id = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise RuntimeError(f"{self}: websocket not connected")

    async def _receive_messages(self) -> None:
        async for message in self._get_websocket():
            if isinstance(message, (bytes, bytearray)):
                if self._active_context_id and self.audio_context_available(self._active_context_id):
                    frame = TTSAudioRawFrame(
                        audio=bytes(message),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=self._active_context_id,
                    )
                    await self.append_to_audio_context(self._active_context_id, frame)
                continue

            data = json.loads(message)
            mtype = data.get("type")

            if mtype in ("session.created", "session.usage", None):
                continue
            elif mtype == "speech.started":
                # Hakim echoes back the `request_id` we sent on `speech.create`
                # (we pass Pipecat's own context_id as that request_id in
                # `run_tts`), so we can use it directly instead of guessing
                # from playback state.
                self._active_context_id = data.get("request_id") or self.get_active_audio_context_id()
            elif mtype == "speech.done":
                await self.stop_ttfb_metrics()
                if self._active_context_id:
                    await self.append_to_audio_context(
                        self._active_context_id, TTSStoppedFrame(context_id=self._active_context_id)
                    )
                    await self.remove_audio_context(self._active_context_id)
                self._active_context_id = None
            elif mtype == "error":
                code = data.get("code", "unknown_error")
                message_text = data.get("message", "")
                retryable = bool(data.get("retryable", False))
                fatal = bool(data.get("fatal", False))
                if self._active_context_id:
                    await self.push_frame(TTSStoppedFrame(context_id=self._active_context_id))
                await self.stop_all_metrics()
                if retryable and not fatal:
                    await self.push_error(error_msg=f"Hakim TTS error [{code}]: {message_text}")
                else:
                    self.reset_active_audio_context()
                    raise HakimStreamError(code, message_text, retryable=retryable)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send one `speech.create` request for this context.

        Audio chunks and the terminal `TTSStoppedFrame` are pushed from the
        background receive loop (`_receive_messages`), not from this
        generator — the base class's `push_start_frame=False` default means
        `run_tts` owns opening the audio context here.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this synthesis context.

        Returns:
            An async generator that yields no frames directly; audio and
            terminal frames are pushed from the background receive loop.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            await self.create_audio_context(context_id)

            try:
                await self._get_websocket().send(
                    json.dumps({"type": "speech.create", "input": text, "request_id": context_id})
                )
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
