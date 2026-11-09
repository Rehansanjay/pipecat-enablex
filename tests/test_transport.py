"""End to end through Pipecat's FastAPI websocket transport.

A local client plays EnableX: it connects, sends ``connected`` and
``start_media``, then 1 s of mu-law audio in 20 ms ``media`` events, paced in
real time. The bot echoes what it hears. The test checks what comes back is
what EnableX would accept: ``media`` events carrying both identifiers, a
``seq`` that counts up from 1 with no gaps, and 8 kHz mu-law.
"""

import asyncio
import audioop
import base64
import json
import math
import socket
import struct

import pytest
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat_enablex import EnableXFrameSerializer

VOICE_ID = "f1aa71c0-8f2a-4fe8-b5ef-9a330454ef58"
STREAM_ID = "9b2e4c1a-0d3f-4e5a-8b7c-6d1e2f3a4b5c"
FRAMES = 50  # 1 s of 20 ms frames


class Echo(FrameProcessor):
    """Sends the caller's audio straight back, so output exists without an LLM."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            frame = OutputAudioRawFrame(
                audio=frame.audio, sample_rate=frame.sample_rate, num_channels=frame.num_channels
            )
        await self.push_frame(frame, direction)


def build_app(pipeline_rate: int) -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        serializer = EnableXFrameSerializer(
            params=EnableXFrameSerializer.InputParams(auto_hang_up=False)
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=serializer,
            ),
        )
        worker = PipelineWorker(
            Pipeline([transport.input(), Echo(), transport.output()]),
            params=PipelineParams(
                audio_in_sample_rate=pipeline_rate, audio_out_sample_rate=pipeline_rate
            ),
        )

        @transport.event_handler("on_client_disconnected")
        async def _gone(_transport, _client):
            await worker.cancel()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await runner.run()

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tone_ulaw_frames(n: int) -> list[bytes]:
    pcm = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 8000)))
        for i in range(160 * n)
    )
    ulaw = audioop.lin2ulaw(pcm, 2)
    return [ulaw[i * 160 : (i + 1) * 160] for i in range(n)]


@pytest.mark.parametrize("pipeline_rate", [16000, 8000])
async def test_a_call_round_trip_through_the_websocket_transport(pipeline_rate):
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(build_app(pipeline_rate), host="127.0.0.1", port=port, log_level="warning")
    )
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)

    received = []
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:

            async def read():
                async for raw in ws:
                    received.append(json.loads(raw))

            reader = asyncio.create_task(read())
            await ws.send(json.dumps({"event": "connected"}))
            await ws.send(
                json.dumps(
                    {
                        "event": "start_media",
                        "stream_id": STREAM_ID,
                        "start": {"stream_id": STREAM_ID, "voice_id": VOICE_ID, "from": "91X", "to": "91Y"},
                    }
                )
            )
            for seq, chunk in enumerate(tone_ulaw_frames(FRAMES), start=1):
                await ws.send(
                    json.dumps(
                        {
                            "event": "media",
                            "voice_id": VOICE_ID,
                            "stream_id": STREAM_ID,
                            "media": {
                                "seq": seq,
                                "timestamp": "2026-11-09T10:00:00Z",
                                "format": {"encoding": "ulaw", "sample_rate": 8000, "channels": 1},
                                "payload": base64.b64encode(chunk).decode(),
                            },
                        }
                    )
                )
                await asyncio.sleep(0.02)
            await asyncio.sleep(1.0)  # let the echo drain
            await ws.send(
                json.dumps({"event": "stop_media", "stream_id": STREAM_ID, "stop": {"voice_id": VOICE_ID}})
            )
            reader.cancel()
    finally:
        server.should_exit = True
        await serving

    media = [m for m in received if m.get("event") == "media"]
    assert media, f"nothing came back; received {received!r}"
    assert all(m["voice_id"] == VOICE_ID and m["stream_id"] == STREAM_ID for m in media)
    assert [m["media"]["seq"] for m in media] == list(range(1, len(media) + 1))
    assert all(
        m["media"]["format"] == {"encoding": "ulaw", "sample_rate": 8000, "channels": 1} for m in media
    )
    returned = sum(len(base64.b64decode(m["media"]["payload"])) for m in media)
    sent = 160 * FRAMES
    # At 16 kHz the audio crosses two stream resamplers, each holding ~100 ms;
    # at 8 kHz it crosses none.
    floor = 0.7 if pipeline_rate != 8000 else 0.95
    assert floor * sent <= returned <= sent, f"returned {returned} of {sent} bytes"
