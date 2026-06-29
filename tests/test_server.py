import os
import tempfile
import unittest

from cuebee_speaker.config import AppConfig
from cuebee_speaker.factory import build_runtime
from cuebee_speaker.server import SpeakerTCPServer


class ServerDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.runtime = build_runtime(
            AppConfig(state_db_path=os.path.join(self.directory.name, "state.sqlite3"))
        )
        self.server = SpeakerTCPServer(self.runtime)

    async def asyncTearDown(self) -> None:
        await self.runtime.close()
        self.directory.cleanup()

    async def test_health_identifies_development_backend(self) -> None:
        response = await self.server.dispatch({"type": "health"})
        self.assertTrue(response["ok"])
        self.assertEqual(response["backend"], "deterministic-development-backend")

    async def test_autoscale_control_message(self) -> None:
        response = await self.server.dispatch(
            {
                "type": "autoscale",
                "audio_arrival_seconds_per_second": 40,
                "real_time_factor": 0.03,
                "backlog_seconds": 10,
                "queue_p95_ms": 150,
                "worker_utilization": 0.8,
                "current_replicas": 2,
                "observed_at": 100,
            }
        )
        self.assertEqual(response["decision"]["action"], "scale_up")


if __name__ == "__main__":
    unittest.main()

