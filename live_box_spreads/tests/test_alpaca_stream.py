import asyncio
import os

import pytest
import websockets

TEST_STREAM_URL = "wss://stream.data.alpaca.markets/v2/test"


@pytest.mark.skipif(
    os.getenv("RUN_ALPACA_STREAM_TEST") != "1",
    reason="Set RUN_ALPACA_STREAM_TEST=1 to enable the live websocket test",
)
def test_alpaca_test_stream_connects():
    async def _run():
        async with websockets.connect(TEST_STREAM_URL, ping_interval=10, ping_timeout=10) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            assert msg
            assert "error" not in str(msg).lower()
            print(f"alpaca test stream message: {msg}")

    asyncio.run(_run())
