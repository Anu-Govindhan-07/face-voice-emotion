from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List


class EventBroadcaster:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue[Dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id] = [q for q in self._subscribers[job_id] if q is not queue]

    async def publish(self, job_id: str, event: str, data: Dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(job_id, []))
        for queue in queues:
            await queue.put({"event": event, "data": data})

    async def stream(self, job_id: str) -> AsyncGenerator[str, None]:
        queue = await self.subscribe(job_id)
        try:
            while True:
                payload = await queue.get()
                event = payload["event"]
                data = payload["data"]
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        finally:
            await self.unsubscribe(job_id, queue)


broadcaster = EventBroadcaster()
