import unittest
from array import array

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(origin_input_ids, prefix_len, extend_end):
    req = Req("rid", "", array("q", origin_input_ids), SamplingParams())
    req.refresh_fill_ids()
    req.prefix_len = prefix_len
    req.extend_end = extend_end
    return req


class TestExtendInputIdsSlice(CustomTestCase):
    def test_prefix_extend_slice_matches_extend_tail(self):
        """prepare_for_extend builds each request's extend-round input ids as
        full_untruncated_fill_ids[prefix_len:extend_end]. With a long cached
        prefix (large prefix_len) and a short extend tail, this must still
        equal exactly the extend-round token ids, not the whole prefix."""
        origin_input_ids = list(range(1000, 1050))  # 50 tokens
        prefix_len, extend_end = 40, 45  # warm-cache hit: only 5 new tokens
        req = _make_req(origin_input_ids, prefix_len, extend_end)

        sliced = req.full_untruncated_fill_ids[req.prefix_len : req.extend_end]

        self.assertEqual(list(sliced), origin_input_ids[prefix_len:extend_end])
        self.assertEqual(len(sliced), req.extend_len)
        # Equivalent to the old two-slice form (get_fill_ids()[prefix_len:]),
        # just without copying the discarded prefix portion.
        self.assertEqual(list(sliced), list(req.get_fill_ids()[req.prefix_len :]))

    def test_full_prefix_extend_slice_is_empty(self):
        """A fully cached request (prefix_len == extend_end) extends zero tokens."""
        origin_input_ids = list(range(10))
        req = _make_req(origin_input_ids, prefix_len=10, extend_end=10)

        sliced = req.full_untruncated_fill_ids[req.prefix_len : req.extend_end]

        self.assertEqual(len(sliced), 0)
        self.assertEqual(req.extend_len, 0)


if __name__ == "__main__":
    unittest.main()
