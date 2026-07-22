"""SchedulerMetricsReporter.record_completed_output_len tests.

Fragility: bypasses SchedulerMetricsReporter.__init__ (which needs a real
Scheduler) via __new__, injecting only the attrs record_completed_output_len
reads (`stats`, `_output_len_ema_alpha`, `_output_len_ema_seen`).
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import SchedulerStats

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_reporter(alpha: float = 0.02) -> SchedulerMetricsReporter:
    reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
    reporter.stats = SchedulerStats()
    reporter._output_len_ema_alpha = alpha
    reporter._output_len_ema_seen = False
    return reporter


class TestRecordCompletedOutputLen(CustomTestCase):
    def test_first_sample_seeds_ema_exactly(self):
        """Guards against the EMA update blending the first sample against
        the 0.0 default (which would report a heavily-damped fraction of
        the first request's actual output length instead of its real
        length)."""
        reporter = _make_reporter()
        reporter.record_completed_output_len(200)
        self.assertEqual(reporter.stats.avg_output_len_ema, 200.0)

    def test_subsequent_samples_blend_with_alpha(self):
        reporter = _make_reporter(alpha=0.5)
        reporter.record_completed_output_len(100)
        reporter.record_completed_output_len(300)
        self.assertEqual(reporter.stats.avg_output_len_ema, 200.0)


if __name__ == "__main__":
    unittest.main()
