#
# Copyright (c) 2026, Rehan Sanjay Venkatesan
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serializer for EnableX Voice Media Streaming.

EnableX opens a websocket to your server once a call is connected and you have
called ``PUT /voice/v1/call/{voice_id}/stream``. Every message is JSON, and the
audio is 8 kHz mono G.711 mu-law, base64-encoded. See
https://developer.enablex.io/voice/media-streaming.html.

Three things differ from Twilio's Media Streams and are the reason this is its
own serializer rather than Twilio's with a different URL:

- The events are ``connected``, ``start_media``, ``media`` and ``stop_media``,
  and barge-in is ``clear_media``.
- Every ``media`` and ``clear_media`` event sent back to EnableX must carry the
  ``voice_id`` and ``stream_id`` from ``start_media``, and outbound ``media``
  carries a ``seq`` that EnableX uses to play frames in order.
- EnableX only accepts mu-law at 8 kHz back. Its docs say anything else plays
  as noise or is dropped silently, so the serializer always converts.

The two identifiers arrive in ``start_media``, which comes through the same
websocket the transport reads, so the serializer learns them itself. You can
still pass them in if your application has already read ``start_media``.
"""

import base64
import json
import time

from loguru import logger

from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer

ENABLEX_API_BASE = "https://api.enablex.io"
# Fixed by EnableX: the stream is always G.711 mu-law, 8 kHz, mono, both ways.
ENABLEX_SAMPLE_RATE = 8000


class EnableXFrameSerializer(FrameSerializer):
    """Serializer for the EnableX Voice Media Streaming websocket protocol.

    When ``auto_hang_up`` is on (the default), an ``EndFrame`` or
    ``CancelFrame`` hangs up the call through ``DELETE /voice/v1/call/{voice_id}``,
    which needs your EnableX App ID and App Key.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration for EnableXFrameSerializer.

        Parameters:
            sample_rate: Override for the pipeline's input sample rate. Leave
                as None to use the rate the pipeline was set up with.
            auto_hang_up: Hang up the EnableX call when the pipeline ends.
                Defaults to True.
        """

        sample_rate: int | None = None
        auto_hang_up: bool = True

    def __init__(
        self,
        voice_id: str | None = None,
        stream_id: str | None = None,
        app_id: str | None = None,
        app_key: str | None = None,
        api_base: str = ENABLEX_API_BASE,
        params: InputParams | None = None,
    ):
        """Initialize the EnableXFrameSerializer.

        Args:
            voice_id: The call's ``voice_id``. Optional: it is read from
                ``start_media`` when it arrives.
            stream_id: The stream's ``stream_id``. Optional, likewise.
            app_id: EnableX App ID, needed only for ``auto_hang_up``.
            app_key: EnableX App Key, needed only for ``auto_hang_up``.
            api_base: Base URL of the EnableX REST API.
            params: Serializer configuration.
        """
        params = params or EnableXFrameSerializer.InputParams()
        super().__init__(params)
        self._params: EnableXFrameSerializer.InputParams = params

        if self._params.auto_hang_up and not (app_id and app_key):
            raise ValueError(
                "auto_hang_up is on but app_id and app_key are not both set. Pass your "
                "EnableX App ID and App Key, or set auto_hang_up=False."
            )

        self._voice_id = voice_id
        self._stream_id = stream_id
        self._app_id = app_id
        self._app_key = app_key
        self._api_base = api_base.rstrip("/")

        self._sample_rate = 0  # pipeline input rate, set in setup()
        self._seq = 0
        self._warned_no_stream = False
        self._hangup_attempted = False

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )

    @property
    def voice_id(self) -> str | None:
        """The call's voice_id, once known."""
        return self._voice_id

    @property
    def stream_id(self) -> str | None:
        """The stream's stream_id, once known."""
        return self._stream_id

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the serializer with the pipeline's configuration."""
        self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate

    def _have_stream(self) -> bool:
        if self._voice_id and self._stream_id:
            return True
        # Bot audio produced before start_media has nowhere to go, since
        # EnableX rejects media without both identifiers. Say so once rather
        # than for every 20 ms frame.
        if not self._warned_no_stream:
            self._warned_no_stream = True
            logger.warning(
                f"{self}: dropping output until start_media arrives with voice_id and stream_id"
            )
        return False

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a Pipecat frame to an EnableX websocket message."""
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            await self._hang_up_call()
            return None

        if isinstance(frame, InterruptionFrame):
            # The output resampler holds back the last ~100 ms of the speech
            # that was just cut off. Left alone, it plays at the start of the
            # bot's next reply.
            await self._output_resampler.reset()
            if not self._have_stream():
                return None
            return json.dumps(
                {"event": "clear_media", "stream_id": self._stream_id, "voice_id": self._voice_id}
            )

        if isinstance(frame, AudioRawFrame):
            if not self._have_stream():
                return None
            ulaw = await pcm_to_ulaw(
                frame.audio, frame.sample_rate, ENABLEX_SAMPLE_RATE, self._output_resampler
            )
            if not ulaw:
                return None
            self._seq += 1
            return json.dumps(
                {
                    "event": "media",
                    "voice_id": self._voice_id,
                    "stream_id": self._stream_id,
                    "media": {
                        "seq": self._seq,
                        "timestamp": int(time.time() * 1000),
                        "format": {
                            "encoding": "ulaw",
                            "sample_rate": ENABLEX_SAMPLE_RATE,
                            "channels": 1,
                        },
                        "payload": base64.b64encode(ulaw).decode("ascii"),
                    },
                }
            )

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize an EnableX websocket message to a Pipecat frame."""
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning(f"{self}: ignoring a message that is not JSON")
            return None
        if not isinstance(message, dict):
            return None

        event = message.get("event")

        if event == "start_media":
            start = message.get("start") or {}
            self._stream_id = message.get("stream_id") or start.get("stream_id") or self._stream_id
            self._voice_id = start.get("voice_id") or self._voice_id
            logger.debug(f"{self}: stream {self._stream_id} started for call {self._voice_id}")
            return None

        if event == "media":
            payload = (message.get("media") or {}).get("payload")
            if not payload:
                return None
            try:
                ulaw = base64.b64decode(payload)
            except (ValueError, TypeError):
                logger.warning(f"{self}: ignoring media with a payload that is not base64")
                return None
            pcm = await ulaw_to_pcm(ulaw, ENABLEX_SAMPLE_RATE, self._sample_rate, self._input_resampler)
            if not pcm:
                return None
            return InputAudioRawFrame(audio=pcm, num_channels=1, sample_rate=self._sample_rate)

        # connected and stop_media carry nothing the pipeline acts on.
        return None

    async def _hang_up_call(self):
        """Hang up the call with EnableX's REST API."""
        if not self._voice_id:
            logger.warning(f"{self}: cannot hang up, no voice_id is known for this call")
            return
        try:
            import aiohttp

            url = f"{self._api_base}/voice/v1/call/{self._voice_id}"
            # Built by hand: aiohttp.BasicAuth is deprecated in recent aiohttp,
            # and its replacement is missing from the versions older Pipecat pins.
            token = base64.b64encode(f"{self._app_id}:{self._app_key}".encode()).decode()
            headers = {"Authorization": f"Basic {token}"}
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status == 200:
                        logger.info(f"{self}: hung up EnableX call {self._voice_id}")
                    elif response.status == 404:
                        # The caller hung up first, or the call already ended.
                        logger.debug(f"{self}: EnableX call {self._voice_id} had already ended")
                    else:
                        body = await response.text()
                        logger.error(
                            f"{self}: hanging up EnableX call {self._voice_id} failed: "
                            f"HTTP {response.status}, {body}"
                        )
        except Exception as e:
            logger.error(f"{self}: hanging up EnableX call {self._voice_id} failed: {e}")
