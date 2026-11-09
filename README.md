# pipecat-enablex

A [Pipecat](https://github.com/pipecat-ai/pipecat) frame serializer for
[EnableX Voice Media Streaming](https://developer.enablex.io/voice/media-streaming.html),
so a Pipecat voice agent can answer and place calls on an EnableX number.

```python
from pipecat_enablex import EnableXFrameSerializer

serializer = EnableXFrameSerializer(app_id=APP_ID, app_key=APP_KEY)
```

## Why it isn't Twilio's serializer with a different URL

EnableX's protocol looks like Twilio Media Streams: JSON messages carrying
base64 G.711 mu-law at 8 kHz. Three differences mean Pipecat's Twilio
serializer can't speak it:

- **Different events.** EnableX sends `connected`, `start_media`, `media` and
  `stop_media`, and barge-in is `clear_media`.
- **Every message back must carry two identifiers.** Outbound `media` and
  `clear_media` events need the `voice_id` and `stream_id` from `start_media`,
  and outbound `media` carries a `seq` that EnableX uses to play frames in order.
- **Only 8 kHz mu-law is accepted back.** EnableX's docs say anything else plays
  as noise or is dropped silently, so the serializer always converts.

The two identifiers arrive in `start_media`, over the same websocket the
transport reads, so the serializer picks them up itself. You don't have to read
the first messages before creating the transport, as you do with Twilio.

## Usage

The call setup is EnableX's REST API, not the websocket, so your app does two
things: get a connected call, then point EnableX at your websocket.

```python
import aiohttp
from fastapi import FastAPI, Request, WebSocket

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner
from pipecat_enablex import EnableXFrameSerializer

API = "https://api.enablex.io/voice/v1/call"
AUTH = aiohttp.BasicAuth(APP_ID, APP_KEY)
app = FastAPI()


@app.post("/event")  # your event_url
async def event(request: Request):
    body = await request.json()
    voice_id = body["voice_id"]
    async with aiohttp.ClientSession(auth=AUTH) as http:
        if body["state"] == "incomingcall":
            await http.put(f"{API}/{voice_id}/accept")
        elif body["state"] == "connected":
            await http.put(f"{API}/{voice_id}/stream", json={"wss_host": "wss://your-host/ws"})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=EnableXFrameSerializer(app_id=APP_ID, app_key=APP_KEY),
        ),
    )
    worker = PipelineWorker(
        Pipeline([transport.input(), stt, llm, tts, transport.output()]),
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
```

For outbound calls you can skip the `connected` step: put a `stream` object
under `action_on_connect` in `POST /voice/v1/call`, and EnableX starts streaming
as soon as the call is answered.

### Run the pipeline at 8 kHz if you can

EnableX's audio is 8 kHz both ways. At any other pipeline rate the serializer
resamples, and a streaming resampler has a start-up delay. Measured here at a
16 kHz pipeline: nothing comes out for the first 100 ms of a reply, then it
catches up in a burst, with 95% of the audio delivered after two seconds. At
8 kHz there is no resampling and 20 ms in is 160 bytes out, every frame.

### Interruptions

On barge-in the serializer sends `clear_media`, so EnableX drops the audio it
has queued, and it also discards what its own resampler is still holding.
Without the second step, the last ~100 ms of the reply that was cut off plays
at the start of the next one.

### Hanging up

With `auto_hang_up` on (the default), the end of the pipeline hangs up the call
through `DELETE /voice/v1/call/{voice_id}`, using your App ID and App Key. A
call that has already ended is not an error. Set
`EnableXFrameSerializer.InputParams(auto_hang_up=False)` to manage that yourself.

### Not handled

- **DTMF.** EnableX reports keypresses through your webhook, not the media
  websocket, so they never reach the serializer.
- **Call control.** Accepting, dialling and starting the stream are REST calls
  your app makes, as in the example.

EnableX streaming is a subscription feature that has to be switched on for your
account, and works on transactional calls only.

## Testing

```
pip install -e ".[dev]"
pytest
```

20 tests:

- every event in both directions, against EnableX's documented payloads
- the `seq` count, both identifiers on everything sent back, and nothing sent
  before `start_media`
- that interrupted speech does not leak into the next reply
- hang-up against a local stand-in for EnableX's REST API: called once, with
  the right credentials, and quiet when the call has already ended
- a full call round trip through Pipecat's `FastAPIWebsocketTransport`, with a
  local client playing EnableX in real time, at 16 kHz and at 8 kHz

Run against Pipecat **1.15.0** and against **main**. Each behaviour was checked
by breaking it and confirming a test fails.

Not yet tested on a live EnableX call, which needs streaming enabled on an
EnableX account.

## License

BSD 2-Clause, the same as Pipecat.
