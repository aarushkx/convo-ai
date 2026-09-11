"""
WebSocket endpoint for realtime voice sessions.

The client sends 30 ms frames of 16-bit mono PCM audio at 16 kHz.
It also sends JSON control messages such as {"type": "end_call"}.

The server sends JSON events such as `ready`, `speech_start`,
`user_transcript`, `assistant_partial`, `assistant_done`,
`interrupted`, `call_ended`, and `error`.

Generated speech is sent as binary data containing the turn ID and
float32 PCM audio. The client should ignore audio from older turns
after an interruption.
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .engines import Engines
from .session import RealtimeSession, pack_audio_frame

logger = logging.getLogger("convo-ai.ws")
router = APIRouter()


@router.websocket("/ws/session")
async def voice_session(websocket: WebSocket):
    if not Engines.ready():
        await websocket.close(code=1013)  # try again later
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = RealtimeSession(session_id, Engines.stt, Engines.tts)
    loop = asyncio.get_running_loop()

    logger.info("session %s connected", session_id)
    session.start()

    async def pump_outbound():
        # Drain session.out_q (filled by worker threads) and forward to the client.
        while True:
            item = await loop.run_in_executor(None, session.out_q.get)
            kind = item[0]
            try:
                if kind == "event":
                    await websocket.send_text(json.dumps(item[1]))
                    if item[1].get("type") == "call_ended":
                        return
                elif kind == "audio":
                    _, turn_id, audio_chunk = item
                    await websocket.send_bytes(pack_audio_frame(turn_id, audio_chunk))
            except Exception:
                logger.exception(
                    "session %s: failed to forward outbound message", session_id)
                return

    pump_task = asyncio.create_task(pump_outbound())

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                session.feed_audio_frame(data)
            elif (text := message.get("text")) is not None:
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    continue
                msg_type = control.get("type")
                if msg_type == "end_call":
                    session.end_call()
                    break
                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info("session %s disconnected", session_id)
    except Exception:
        logger.exception("session %s: unexpected error", session_id)
    finally:
        session.stop()
        pump_task.cancel()
        try:
            await pump_task
        except (asyncio.CancelledError, Exception):
            pass
        logger.info("session %s cleaned up", session_id)
