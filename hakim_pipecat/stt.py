"""Hakim STT service for Pipecat.

Speaks the public `WSS /v1/audio/transcriptions/stream` API directly (see
https://tryhakim.ai/docs) via the `websockets` library Pipecat already
depends on — no dependency on the `hakim` Python package.

Wire contract implemented here:
  - Client -> server: `session.update`, `input_audio_buffer.append`,
    `input_audio_buffer.commit`, `session.close`.
  - Server -> client: `session.created`, `transcription.delta`
    (interim), `transcription.done` (final, for the most recent commit
    window), `error`.
  - Model is pinned to `hakim-arab-v2` — the only accepted value today.

Turn detection: Hakim's realtime STT doesn't do its own end-of-turn
detection — it transcribes continuously and finalizes whatever it has
buffered when it receives `input_audio_buffer.commit`. This service sends
that commit whenever Pipecat's own VAD reports the user stopped speaking,
so turn-taking is fully owned by Pipecat (VAD / Smart Turn), matching how
most STT plugins in this framework operate when the provider has no
built-in turn model.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings

# NOT_GIVEN, the sentinel type and is_given moved to pipecat.utils.types in
# pipecat 1.8, where the type also lost its underscore. Importing them from
# pipecat.services.settings raises ImportError on 1.8 and later.
try:
    from pipecat.utils.types import NOT_GIVEN, is_given
    from pipecat.utils.types import NotGiven as _NotGiven
except ImportError:  # pipecat < 1.8
    from pipecat.services.settings import NOT_GIVEN, _NotGiven, is_given
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from ._common import HakimStreamError, Region, resolve_api_key, resolve_ws_url

STT_MODEL = "hakim-arab-v2"

# Hakim recommends 50-250ms audio chunks over the wire; Pipecat delivers
# smaller frames from the transport, so we buffer up to this size before
# sending each `input_audio_buffer.append`.
_CHUNK_MS = 100


@dataclass
class HakimSTTSettings(STTSettings):
    """Runtime-updatable settings for `HakimSTTService`.

    Parameters:
        timestamps: `"word" | "segment" | "none"`.
        diarize: Speaker diarization. Only useful for stereo audio with one
            speaker per channel (e.g. call recordings) — mono `diarize=True`
            transcribes without speaker labels.
        partials: Whether to emit interim (non-final) transcripts.
    """

    timestamps: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarize: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    partials: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class HakimSTTService(WebsocketSTTService):
    """Hakim realtime speech-to-text service for Pipecat.

    Example::

        stt = HakimSTTService(api_key=os.environ["HAKIM_API_KEY"], language=Language.AR)
    """

    Settings = HakimSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: Language | str = Language.AR,
        input_audio_format: str = "pcm16",
        sample_rate: int = 16000,
        region: Region = "auto",
        base_url: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Hakim STT service.

        Args:
            api_key: Hakim API key (`hk_live_...`). Falls back to the
                `HAKIM_API_KEY` env var.
            language: Language hint for transcription. Defaults to Arabic.
            input_audio_format: `"pcm16" | "opus" | "mulaw"`. Pipecat's
                transport delivers PCM16 by default, so the default matches
                without any client-side transcoding.
            sample_rate: Must match the sample rate of frames actually sent
                to `run_stt`.
            region: `"auto" | "de" | "uae" | "ksa"` — pins the session to a
                specific regional endpoint. `"auto"` picks the closest
                healthy region.
            base_url: Overrides the region table entirely (staging /
                self-hosted). Takes precedence over `region`.
            settings: Runtime-updatable `HakimSTTSettings`. Values here take
                precedence over the `language` shorthand above.
            **kwargs: Additional arguments passed to `WebsocketSTTService`
                (e.g. `keepalive_timeout`, `vad_force_turn_endpoint` is not
                applicable here — see the module docstring).
        """
        default_settings = self.Settings(
            model=STT_MODEL,
            language=language,
            timestamps="segment",
            diarize=False,
            partials=True,
        )
        if settings:
            for f in ("language", "timestamps", "diarize", "partials"):
                value = getattr(settings, f)
                if is_given(value):
                    setattr(default_settings, f, value)

        super().__init__(sample_rate=sample_rate, settings=default_settings, **kwargs)

        self._api_key = resolve_api_key(api_key)
        self._input_audio_format = input_audio_format
        self._region = region
        self._base_url = base_url

        self._chunk_size_bytes = int(_CHUNK_MS * sample_rate * 2 / 1000)
        self._audio_buffer = bytearray()

        self._websocket = None
        self._receive_task = None
        self._finalize_requested_at_commit = False

    @property
    def provider(self) -> str:
        """Get the name of the STT provider.

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

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Forward audio to Hakim over the persistent websocket.

        Results arrive asynchronously via the receive loop (`_handle_message`),
        not from this generator — matches how other continuous-streaming STT
        plugins in this framework (e.g. AssemblyAI, Deepgram) are structured.

        Args:
            audio: Raw PCM audio bytes to transcribe.

        Returns:
            An async generator that yields no frames directly; results are
            pushed from the background receive loop instead.
        """
        self._audio_buffer.extend(audio)

        if self._websocket and self._websocket.state is State.OPEN:
            while len(self._audio_buffer) >= self._chunk_size_bytes:
                chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
                self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
                try:
                    await self._websocket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }
                        )
                    )
                except Exception as e:
                    logger.warning(f"{self}: send failed: {e}")
                    break

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Send `input_audio_buffer.commit` when Pipecat's VAD says the turn ended.

        Args:
            frame: The frame to process.
            direction: The direction the frame is travelling in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                self.request_finalize()
                await self._websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await self.start_processing_metrics()

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

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._disconnect()

    async def _connect_websocket(self) -> None:
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self}: connecting to Hakim STT")
            url = resolve_ws_url(
                "/v1/audio/transcriptions/stream", region=self._region, base_url=self._base_url
            )
            self._websocket = await websocket_connect(
                url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            await self._websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "model": STT_MODEL,
                            "language": str(self._settings.language),
                            "timestamps": self._settings.timestamps,
                            "diarize": self._settings.diarize,
                            "partials": self._settings.partials,
                            "input_audio_format": self._input_audio_format,
                            "input_sample_rate": self.sample_rate,
                        },
                    }
                )
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Hakim: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self) -> None:
        try:
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Hakim STT")
                await self._websocket.send(json.dumps({"type": "session.close"}))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise RuntimeError(f"{self}: websocket not connected")

    async def _receive_messages(self) -> None:
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON message: {message}")
                continue
            await self._handle_message(data)

    async def _handle_message(self, message: dict) -> None:
        mtype = message.get("type")

        if mtype in ("session.created", None):
            return

        if mtype == "transcription.delta":
            text = message.get("text", "")
            if not text:
                return
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    "",
                    time_now_iso8601(),
                    self._settings.language if isinstance(self._settings.language, Language) else None,
                    message,
                )
            )
        elif mtype == "transcription.done":
            self.confirm_finalize()
            text = message.get("text", "")
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    "",
                    time_now_iso8601(),
                    self._settings.language if isinstance(self._settings.language, Language) else None,
                    message,
                    finalized=True,
                )
            )
            await self.stop_processing_metrics()
        elif mtype == "error":
            code = message.get("code", "unknown_error")
            error_message = message.get("message", "")
            retryable = bool(message.get("retryable", False))
            if retryable:
                await self.push_error(error_msg=f"Hakim STT error [{code}]: {error_message}")
            else:
                raise HakimStreamError(code, error_message, retryable=retryable)
