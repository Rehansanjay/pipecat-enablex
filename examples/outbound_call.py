"""Live test: a Pipecat bot calls a phone over EnableX and holds a conversation.

The flow follows EnableX's Media Streaming guide:

1. ``POST /voice/v1/call`` dials the number, with ``event_url`` pointing here.
2. On the ``connected`` webhook, ``PUT /voice/v1/call/{voice_id}/stream`` tells
   EnableX to open a websocket to ``/ws`` on this server.
3. ``/ws`` runs a Pipecat pipeline (Deepgram STT, Gemini, Deepgram TTS) through
   ``EnableXFrameSerializer``, at 8 kHz so nothing is resampled.

At the end it prints a timeline of everything that happened, which is the
result of the test.

Needs a public HTTPS host that forwards to this server, for example
``ngrok http 8765``. Then:

    python examples/outbound_call.py --to 91XXXXXXXXXX --host abcd.ngrok-free.app

Reads ENABLEX_APP_ID, ENABLEX_APP_KEY, ENABLEX_FROM_NUMBER, DEEPGRAM_API_KEY
and GEMINI_API_KEY from the environment or a .env file.
"""

import argparse
import asyncio
import base64
import os
import time

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner
from pipecat_enablex import EnableXFrameSerializer

load_dotenv()

API = "https://api.enablex.io/voice/v1/call"
APP_ID = os.environ["ENABLEX_APP_ID"]
APP_KEY = os.environ["ENABLEX_APP_KEY"]
FROM_NUMBER = os.environ["ENABLEX_FROM_NUMBER"]
RATE = 8000  # EnableX's own rate, so the serializer never resamples

SYSTEM = (
    "You are a friendly test bot on a phone call. Keep every reply to one or two short "
    "sentences, with no lists or symbols, because it is spoken aloud."
)
GREETING = "Hi, this is the Pipecat test bot on EnableX. Say something and I'll answer."

HEADERS = {
    "Authorization": "Basic " + base64.b64encode(f"{APP_ID}:{APP_KEY}".encode()).decode(),
    "Content-Type": "application/json",
}

START = time.monotonic()
timeline: list[str] = []
call_over = asyncio.Event()
public_host = ""


def mark(text: str):
    line = f"{time.monotonic() - START:7.2f}s  {text}"
    timeline.append(line)
    logger.info(line)


class Tap(FrameProcessor):
    """Counts audio and logs transcripts as they pass, without changing anything."""

    def __init__(self, label: str, kind: type[Frame]):
        super().__init__()
        self.label = label
        self.kind = kind  # the caller's audio flows on past the bot's tap too
        self.audio_frames = 0
        self.audio_secs = 0.0
        self.rates: set[int] = set()
        self.first_audio_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, self.kind):
            if self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
                mark(f"first {self.label} audio")
            self.audio_frames += 1
            self.audio_secs += len(frame.audio) / (frame.sample_rate * 2 * frame.num_channels)
            self.rates.add(frame.sample_rate)
        elif isinstance(frame, TranscriptionFrame):
            mark(f"caller said: {frame.text!r}")
        await self.push_frame(frame, direction)

    def summary(self) -> str:
        rates = ", ".join(f"{r} Hz" for r in sorted(self.rates)) or "none"
        return f"{self.label} audio: {self.audio_frames} frames, {self.audio_secs:.1f} s at {rates}"


app = FastAPI()


async def enablex(method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
    async with aiohttp.ClientSession() as http:
        async with http.request(method, f"{API}{path}", json=body, headers=HEADERS) as r:
            try:
                return r.status, await r.json(content_type=None)
            except Exception:
                return r.status, await r.text()


@app.post("/event")
async def event(request: Request):
    body = await request.json()
    state = body.get("state")
    voice_id = body.get("voice_id")
    extra = {k: v for k, v in body.items() if k in ("disconnect_reason", "call_answered_by", "call_duration")}
    mark(f"webhook: {state} {extra or ''}")

    if state == "connected":
        status, reply = await enablex("PUT", f"/{voice_id}/stream", {"wss_host": f"wss://{public_host}/ws"})
        mark(f"stream API -> HTTP {status}: {reply}")
        started = status < 300 and isinstance(reply, dict) and reply.get("state") != "stream_failed"
        if not started:
            # Don't leave the phone ringing with nothing on the line.
            status, reply = await enablex("DELETE", f"/{voice_id}")
            mark(f"streaming did not start, hung up -> HTTP {status}: {reply}")
    elif state == "disconnected":
        call_over.set()
    return {}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    mark("EnableX opened the websocket")

    serializer = EnableXFrameSerializer(app_id=APP_ID, app_key=APP_KEY)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = GoogleLLMService(
        api_key=os.environ["GEMINI_API_KEY"],
        settings=GoogleLLMService.Settings(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), system_instruction=SYSTEM
        ),
    )
    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramTTSService.Settings(voice="aura-2-helena-en"),
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    heard, spoke = Tap("caller", InputAudioRawFrame), Tap("bot", TTSAudioRawFrame)

    worker = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                stt,  # passes the caller's audio on as well as the transcripts
                heard,
                aggregators.user(),
                llm,
                tts,
                spoke,
                transport.output(),
                aggregators.assistant(),
            ]
        ),
        params=PipelineParams(audio_in_sample_rate=RATE, audio_out_sample_rate=RATE),
    )

    async def greet_once_stream_is_known():
        # Audio sent before start_media has no voice_id/stream_id to carry, and
        # the serializer drops it. Wait for the identifiers, then greet.
        while serializer.stream_id is None:
            await asyncio.sleep(0.02)
        mark(f"start_media: stream {serializer.stream_id}")
        await worker.queue_frames([TTSSpeakFrame(GREETING)])

    greeter = asyncio.create_task(greet_once_stream_is_known())

    @transport.event_handler("on_client_disconnected")
    async def _gone(_transport, _client):
        mark("EnableX closed the websocket")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
    greeter.cancel()
    timeline.append(f"         {heard.summary()}")
    timeline.append(f"         {spoke.summary()}")


async def main():
    global public_host
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--to", required=True, help="number to call, digits only, e.g. 91XXXXXXXXXX")
    parser.add_argument("--host", required=True, help="public host forwarding here, e.g. abcd.ngrok-free.app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-minutes", type=float, default=5)
    args = parser.parse_args()
    public_host = args.host.removeprefix("https://").rstrip("/")

    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    mark(f"server up on :{args.port}, public host {public_host}")

    status, reply = await enablex(
        "POST",
        "",
        {
            "name": "pipecat-enablex-live-test",
            "owner_ref": "pipecat-enablex",
            "from": FROM_NUMBER,
            "to": args.to,
            "event_url": f"https://{public_host}/event",
        },
    )
    mark(f"dial-out API -> HTTP {status}: {reply}")

    if status < 300:
        try:
            await asyncio.wait_for(call_over.wait(), timeout=args.max_minutes * 60)
        except TimeoutError:
            mark(f"no disconnected webhook after {args.max_minutes} min, stopping")
        await asyncio.sleep(2)  # let the websocket side finish and log

    server.should_exit = True
    await serving
    print("\n==== EnableX live test timeline ====")
    print("\n".join(timeline))


if __name__ == "__main__":
    asyncio.run(main())
