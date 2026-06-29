import unittest

from cuebee_speaker.autoscaler import AudioBacklogAutoscaler, ScalingSample
from cuebee_speaker.config import AutoscalerConfig


class AutoscalerTests(unittest.TestCase):
    def test_backlog_and_queue_trigger_scale_up(self) -> None:
        scaler = AudioBacklogAutoscaler(AutoscalerConfig(cooldown_seconds=0))
        decision = scaler.recommend(
            ScalingSample(
                audio_arrival_seconds_per_second=40,
                real_time_factor=0.03,
                backlog_seconds=10,
                queue_p95_ms=150,
                worker_utilization=0.85,
                current_replicas=2,
                observed_at=100,
            )
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertGreater(decision.desired_replicas, 2)
        self.assertIn("backlog_high", decision.reasons)

    def test_scale_down_requires_sustained_low_windows(self) -> None:
        scaler = AudioBacklogAutoscaler(
            AutoscalerConfig(cooldown_seconds=0, scale_down_windows=3)
        )
        decisions = []
        for observed_at in (100, 110, 120):
            decisions.append(
                scaler.recommend(
                    ScalingSample(
                        audio_arrival_seconds_per_second=1,
                        real_time_factor=0.03,
                        backlog_seconds=0,
                        queue_p95_ms=1,
                        worker_utilization=0.1,
                        current_replicas=5,
                        observed_at=observed_at,
                    )
                )
            )
        self.assertEqual([item.action for item in decisions], ["hold", "hold", "scale_down"])
        self.assertEqual(decisions[-1].desired_replicas, 4)

    def test_cooldown_holds_repeated_scale_up(self) -> None:
        scaler = AudioBacklogAutoscaler(AutoscalerConfig(cooldown_seconds=30))
        first = scaler.recommend(ScalingSample(100, 0.03, 0, 10, 0.8, 2, 100))
        second = scaler.recommend(ScalingSample(120, 0.03, 0, 10, 0.8, 3, 110))
        self.assertEqual(first.action, "scale_up")
        self.assertEqual(second.action, "hold")
        self.assertIn("cooldown", second.reasons)


if __name__ == "__main__":
    unittest.main()

