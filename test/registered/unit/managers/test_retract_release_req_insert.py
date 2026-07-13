"""
Unit tests for release_req's is_insert gating on the retract_decode resume path.

retract_decode's per-request loop now calls release_req(..., is_insert=True) so a
retracted request's KV cache lands back in the radix tree (instead of being freed
outright) and can be resumed with a cache hit. is_insert=True must be suppressed
back to today's free-outright behavior under decode disaggregation (resume goes
through load_kv_cache's CPU copy, never tree_cache.match_prefix) and under HiSparse
(release_kv_cache would free a different, smaller range than hisparse_coordinator's
retract_req already accounted for). Every other release_req caller (retract_decode's
final force-abort, retract_all, priority preemption) must keep the pre-existing
default of is_insert=False.

Usage:
    python -m pytest test/registered/unit/managers/test_retract_release_req_insert.py -v
"""

from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=10, suite="base-c-test-cpu")

import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo, release_req
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_context


def _make_cache_with_pools(page_size=1):
    """Create a RadixCache with mock pools sufficient for release_req's KV release."""
    mock_allocator = MagicMock()
    mock_allocator.device = torch.device("cpu")
    # Large enough that evict_from_tree_cache's post-release eviction never
    # triggers -- eviction mechanics are covered elsewhere, not the target here.
    mock_allocator.available_size.return_value = 10**9

    max_seq_len = 64
    max_batch = 4
    req_to_token = torch.zeros(max_batch, max_seq_len, dtype=torch.int64)

    mock_pool = MagicMock()
    mock_pool.req_to_token = req_to_token
    mock_pool.write = lambda idx_tuple, values: req_to_token.__setitem__(
        idx_tuple, values
    )

    cache = RadixCache.create_simulated(
        mock_allocator=mock_allocator, page_size=page_size
    )
    cache.req_to_token_pool = mock_pool
    return cache, req_to_token


class MockReq:
    """Minimal mock Req covering what release_req + release_kv_cache touch."""

    def __init__(self, fill_ids, req_pool_idx=0):
        self.origin_input_ids = array("q", fill_ids)
        self.output_ids = array("q")
        self.extra_key = None
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = 0
        self.last_node = None
        self.priority = 0
        self.kv_committed_len = len(fill_ids)
        self.kv = ReqKvInfo(kv_allocated_len=len(fill_ids), swa_evicted_seqlen=0)
        self.reset_for_retract_calls = 0
        self.offload_kv_cache_calls = 0

    def finished(self):
        return False

    def effective_kv_committed_len(self):
        return self.kv_committed_len

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        self.offload_kv_cache_calls += 1

    def reset_for_retract(self):
        self.reset_for_retract_calls += 1


class TestRetractReleaseReqInsert(unittest.TestCase):
    def setUp(self):
        self._override = get_context().override_server_args(page_size=1)
        self._override.install()

    def tearDown(self):
        self._override.restore()

    _UNSET = object()

    def _release(
        self,
        req,
        cache,
        *,
        is_insert=_UNSET,
        disaggregation_mode="null",
        hisparse_coordinator=None,
    ):
        server_args = get_context().server_args
        server_args.override(source="test", disaggregation_mode=disaggregation_mode)
        kwargs = {} if is_insert is self._UNSET else {"is_insert": is_insert}
        release_req(
            req=req,
            remaing_req_count=0,
            server_args=server_args,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=cache,
            hisparse_coordinator=hisparse_coordinator,
            **kwargs,
        )

    def _make_req_with_kv(self, cache, req_to_token, fill_ids):
        req_to_token[0, : len(fill_ids)] = torch.tensor(
            [10 * t for t in fill_ids], dtype=torch.int64
        )
        return MockReq(fill_ids)

    def test_insert_true_preserves_prefix_for_resume(self):
        """Normal retract: is_insert=True must land the released KV in the tree
        so the resumed request's next prefill hits the cache instead of a cold
        re-prefill miss."""
        cache, req_to_token = _make_cache_with_pools()
        fill_ids = [1, 2, 3, 4, 5]
        req = self._make_req_with_kv(cache, req_to_token, fill_ids)

        self._release(req, cache, is_insert=True)

        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", fill_ids)))
        )
        self.assertEqual(len(result.device_indices), len(fill_ids))
        self.assertEqual(cache.evictable_size(), len(fill_ids))
        self.assertEqual(cache.root_node.lock_ref, 1)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(req.reset_for_retract_calls, 1)

    def test_omitted_is_insert_defaults_to_free(self):
        """Every other release_req caller (final force-abort, retract_all,
        priority preemption) calls release_req without an is_insert kwarg at
        all, relying on the pre-existing default of is_insert=False; a
        flipped default would silently start caching aborted/priority-
        preempted requests' prefixes too."""
        cache, req_to_token = _make_cache_with_pools()
        fill_ids = [1, 2, 3, 4, 5]
        req = self._make_req_with_kv(cache, req_to_token, fill_ids)

        self._release(req, cache)

        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", fill_ids)))
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(cache.evictable_size(), 0)

    def test_insert_true_disabled_under_decode_disaggregation(self):
        """Decode-disaggregation retraction resumes via load_kv_cache's CPU
        copy, not tree_cache.match_prefix -- inserting here would create a dead
        node while working against the retraction's GPU-memory-reclaim goal, so
        is_insert=True must be suppressed and offload_kv_cache must still run."""
        cache, req_to_token = _make_cache_with_pools()
        fill_ids = [1, 2, 3, 4, 5]
        req = self._make_req_with_kv(cache, req_to_token, fill_ids)

        self._release(req, cache, is_insert=True, disaggregation_mode="decode")

        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", fill_ids)))
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(req.offload_kv_cache_calls, 1)

    def test_insert_true_disabled_with_hisparse(self):
        """HiSparse's retract_req frees a kv_allocated_len-sized range sized to
        pair with release_kv_cache's is_insert=False free path; inserting
        instead would free a different (smaller) range and risks a double free
        into HiSparse's free list, so is_insert=True must be suppressed
        whenever a HiSparse coordinator is configured."""
        cache, req_to_token = _make_cache_with_pools()
        fill_ids = [1, 2, 3, 4, 5]
        req = self._make_req_with_kv(cache, req_to_token, fill_ids)
        hisparse_coordinator = MagicMock()

        self._release(
            req, cache, is_insert=True, hisparse_coordinator=hisparse_coordinator
        )

        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", fill_ids)))
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(cache.evictable_size(), 0)
        hisparse_coordinator.retract_req.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
