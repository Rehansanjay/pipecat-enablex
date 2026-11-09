"""Unit tests for EnableXFrameSerializer, against the documented protocol.

https://developer.enablex.io/voice/media-streaming.html
"""

import base64
import json
import math
import struct
from types import SimpleNamespace

import pytest
from aiohttp import web

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
)
from pipecat_enablex import EnableXFrameSerializer

VOICE_ID = "f1aa71c0-8f2a-4fe8-b5ef-9a330454ef58"
STREAM_ID = "9b2e4c1a-0d3f-4e5a-8b7c-6d1e2f3a4b5c"
PIPELINE_RATE = 16000


def start_media(voice_id=VOICE_ID, stream_id=STREAM_ID):
    return json.dumps(
        {
            "event": "start_media",
            "stream_id": stream_id,
            "start": {"stream_id": stream_id, "voice_id": voice_id, "from": "91X", "to": "91Y"},
        }
    )


def tone_pcm16(ms: int, rate: int, freq: float = 440.0) -> bytes:
    n = rate * ms // 1000
    return b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)
    )


def media(ulaw: bytes, seq: int = 1):
    return json.dumps(
        {
            "event": "media",
            "voice_id": VOICE_ID,
            "stream_id": STREAM_ID,
            "media": {
                "seq": seq,
                "timestamp": "2026-11-09T10:00:00Z",
                "format": {"encoding": "ulaw", "sample_rate": 8000, "channels": 1},
                "payload": base64.b64encode(ulaw).decode(),
            },
        }
    )


async def make(auto_hang_up=False, **kwargs):
    params = EnableXFrameSerializer.InputParams(auto_hang_up=auto_hang_up)
    s = EnableXFrameSerializer(params=params, **kwargs)
    await s.setup(SimpleNamespace(audio_in_sample_rate=PIPELINE_RATE))
    return s


# -- inbound ---------------------------------------------------------------


async def test_start_media_supplies_both_identifiers():
    s = await make()
    assert await s.deserialize(start_media()) is None
    assert s.voice_id == VOICE_ID
    assert s.stream_id == STREAM_ID


async def test_inbound_media_becomes_pcm_at_the_pipeline_rate():
    s = await make()
    await s.deserialize(start_media())
    # 1 s of 8 kHz mu-law in 20 ms chunks, which is 160 bytes each.
    ulaw = b"\x7f\xff" * 4000
    total = 0
    for i in range(50):
        frame = await s.deserialize(media(ulaw[i * 160 : (i + 1) * 160], seq=i + 1))
        if frame is None:
            continue
        assert isinstance(frame, InputAudioRawFrame)
        assert frame.sample_rate == PIPELINE_RATE
        assert frame.num_channels == 1
        total += len(frame.audio)
    # 1 s at 16 kHz, 16-bit, is 32000 bytes; the stream resampler may hold a
    # few milliseconds back, so allow a small shortfall.
    assert 30000 <= total <= 32000


async def test_connected_and_stop_media_produce_no_frames():
    s = await make()
    assert await s.deserialize(json.dumps({"event": "connected"})) is None
    stop = {"event": "stop_media", "stream_id": STREAM_ID, "stop": {"voice_id": VOICE_ID}}
    assert await s.deserialize(json.dumps(stop)) is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        b"\xff\xfe\x00",
        json.dumps(["media"]),
        json.dumps({"event": "media", "media": {}}),
        json.dumps({"event": "media", "media": {"payload": "%%%not base64%%%"}}),
    ],
)
async def test_malformed_input_is_ignored(raw):
    s = await make()
    await s.deserialize(start_media())
    assert await s.deserialize(raw) is None


# -- outbound --------------------------------------------------------------


async def test_outbound_audio_carries_both_ids_and_an_increasing_seq():
    s = await make()
    await s.deserialize(start_media())
    sent = []
    # 2 s, because the stream resampler releases nothing for its first ~100 ms
    # and then catches up in a burst.
    for _ in range(100):
        frame = OutputAudioRawFrame(audio=tone_pcm16(20, PIPELINE_RATE), sample_rate=PIPELINE_RATE, num_channels=1)
        out = await s.serialize(frame)
        if out is not None:
            sent.append(json.loads(out))

    assert sent, "no audio was sent"
    assert [m["media"]["seq"] for m in sent] == list(range(1, len(sent) + 1))
    for m in sent:
        assert m["event"] == "media"
        assert m["voice_id"] == VOICE_ID
        assert m["stream_id"] == STREAM_ID
        assert m["media"]["format"] == {"encoding": "ulaw", "sample_rate": 8000, "channels": 1}
        assert isinstance(m["media"]["timestamp"], int)
    ulaw_total = sum(len(base64.b64decode(m["media"]["payload"])) for m in sent)
    # 2 s at 8 kHz mu-law is 16000 bytes, less what the resampler still holds.
    assert 0.9 * 16000 <= ulaw_total <= 16000


async def test_audio_already_at_8khz_is_sent_as_is():
    s = await make()
    await s.deserialize(start_media())
    frame = OutputAudioRawFrame(audio=tone_pcm16(20, 8000), sample_rate=8000, num_channels=1)
    out = json.loads(await s.serialize(frame))
    assert len(base64.b64decode(out["media"]["payload"])) == 160


async def test_interruption_sends_clear_media_with_both_ids():
    s = await make()
    await s.deserialize(start_media())
    out = json.loads(await s.serialize(InterruptionFrame()))
    assert out == {"event": "clear_media", "stream_id": STREAM_ID, "voice_id": VOICE_ID}


async def test_interrupted_speech_does_not_leak_into_the_next_reply():
    import audioop

    s = await make()
    await s.deserialize(start_media())
    for _ in range(25):  # 0.5 s of tone, so the resampler is holding some back
        await s.serialize(
            OutputAudioRawFrame(audio=tone_pcm16(20, PIPELINE_RATE), sample_rate=PIPELINE_RATE, num_channels=1)
        )
    await s.serialize(InterruptionFrame())

    silence = b"\x00\x00" * (PIPELINE_RATE // 50)
    after = b""
    for _ in range(25):
        out = await s.serialize(OutputAudioRawFrame(audio=silence, sample_rate=PIPELINE_RATE, num_channels=1))
        if out is not None:
            after += base64.b64decode(json.loads(out)["media"]["payload"])
    assert after, "no audio came out after the interruption"
    # The tone is RMS ~5600. Anything near that means the cut-off speech came back.
    assert audioop.rms(audioop.ulaw2lin(after, 2), 2) < 200


async def test_nothing_is_sent_before_start_media():
    s = await make()
    frame = OutputAudioRawFrame(audio=tone_pcm16(20, PIPELINE_RATE), sample_rate=PIPELINE_RATE, num_channels=1)
    assert await s.serialize(frame) is None
    assert await s.serialize(InterruptionFrame()) is None


async def test_ids_passed_in_work_without_start_media():
    s = await make(voice_id=VOICE_ID, stream_id=STREAM_ID)
    out = json.loads(await s.serialize(InterruptionFrame()))
    assert out["voice_id"] == VOICE_ID and out["stream_id"] == STREAM_ID


# -- hang-up -----------------------------------------------------------------


def test_auto_hang_up_needs_credentials():
    with pytest.raises(ValueError):
        EnableXFrameSerializer(app_id="only-the-id")


class FakeEnableX:
    """A local stand-in for EnableX's REST API that records hang-ups."""

    def __init__(self, status=200):
        self.status = status
        self.requests = []

    async def __aenter__(self):
        app = web.Application()
        app.router.add_route("DELETE", "/voice/v1/call/{voice_id}", self._delete)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc):
        await self._runner.cleanup()

    async def _delete(self, request):
        self.requests.append((request.path, request.headers.get("Authorization")))
        return web.json_response({"voice_id": request.match_info["voice_id"], "state": "success"}, status=self.status)


async def test_end_frame_hangs_up_the_call_once():
    async with FakeEnableX() as api:
        s = await make(auto_hang_up=True, app_id="APP", app_key="KEY", api_base=api.base)
        await s.deserialize(start_media())
        assert await s.serialize(EndFrame()) is None
        assert await s.serialize(CancelFrame()) is None
    expected_auth = "Basic " + base64.b64encode(b"APP:KEY").decode()
    assert api.requests == [(f"/voice/v1/call/{VOICE_ID}", expected_auth)]


async def test_a_call_that_already_ended_is_not_an_error():
    async with FakeEnableX(status=404) as api:
        s = await make(auto_hang_up=True, app_id="APP", app_key="KEY", api_base=api.base)
        await s.deserialize(start_media())
        await s.serialize(EndFrame())
    assert len(api.requests) == 1


async def test_hang_up_without_a_voice_id_does_not_call_the_api():
    async with FakeEnableX() as api:
        s = await make(auto_hang_up=True, app_id="APP", app_key="KEY", api_base=api.base)
        await s.serialize(EndFrame())
    assert api.requests == []
