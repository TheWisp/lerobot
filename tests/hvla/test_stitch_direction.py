"""The stitch must not resume on an index that reverses the prefix.

Position-only matching left 42.7% of handoffs reversing against a chunk-interior
floor of 9.6%; preferring candidates that continue the prefix gives 15.4%.

Constructing these needs care. Being close to the continuation target p+v
implies being forward of p, so the two criteria usually agree and a naive test
passes with the filter removed. They only diverge when NO candidate is near the
target — then the closest one is free to point backwards. Every test below that
claims to pin the filter is built in that regime, and the whole file is
mutation-checked by replacing `forward or all_cands` with `all_cands`.
"""

import numpy as np

from lerobot.policies.hvla.s1_inference import choose_stitch_index

D = 2


def test_the_closest_index_is_rejected_when_it_reverses():
    """Discriminating case: nearest candidate is backwards, so distance alone fails.

    prefix ends at 1.0 moving +1, so the continuation target is 2.0.
      k=2 -> 0.5  distance 1.5, step -0.5  (backwards)
      k=3 -> 3.2  distance 1.2, step +2.2  (forwards)
      k=4 -> 0.9  distance 1.1, step -0.1  (backwards, and the closest of all)
    Position-only picks k=4 and reverses; the filter must pick k=3.
    """
    prefix = np.array([[0.0], [1.0]])
    chunk = np.array([[0.0], [1.0], [0.5], [3.2], [0.9]])
    k = choose_stitch_index(chunk, prefix, D, search=3)
    assert k == 3, f"picked k={k} (value {chunk[k][0]}) — expected the forward candidate at k=3"
    step = chunk[k][0] - prefix[D - 1][0]
    assert step > 0, "resumed on an index that moves backwards"


def test_multi_joint_closest_is_rejected_when_it_reverses():
    """Same regime with a real joint vector: direction is not per-joint.

    A backward candidate can never be nearer to p+v than ||v||, so every forward
    candidate has to be placed farther than that for distance alone to go wrong.
      k=2 -> (1.0, 0.9)  dist 1.49, step (0, -0.1)      backwards, nearest overall
      k=3 -> (3.5, 3.6)  dist 2.19, step (+2.5, +2.6)   forwards
      k=4 -> (0.5, 0.5)  dist 2.12, step (-0.5, -0.5)   backwards
    """
    prefix = np.array([[0.0, 0.0], [1.0, 1.0]])  # v = (+1, +1), target (2, 2)
    chunk = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.9],
            [3.5, 3.6],
            [0.5, 0.5],
        ]
    )
    k = choose_stitch_index(chunk, prefix, D, search=3)
    assert k == 3, f"picked k={k} — the only forward candidate is k=3"
    step = chunk[k] - prefix[D - 1]
    assert float(np.dot(step, np.array([1.0, 1.0]))) > 0, "resumed on a reversing index"


def test_among_forward_candidates_the_closest_wins():
    """The filter selects, it does not replace the distance cost."""
    prefix = np.array([[0.0], [1.0]])
    chunk = np.array([[0.0], [1.0], [1.2], [2.1], [5.0]])
    assert choose_stitch_index(chunk, prefix, D, search=3) == 3


def test_falls_back_to_position_when_nothing_continues():
    """No forward candidate: closest is better than forcing a direction."""
    prefix = np.array([[0.0], [1.0]])
    chunk = np.array([[0.0], [1.0], [0.8], [0.2], [-3.0]])
    assert choose_stitch_index(chunk, prefix, D, search=3) == 2


def test_disabled_search_is_unchanged():
    prefix = np.array([[0.0], [1.0]])
    chunk = np.array([[0.0], [1.0], [9.0], [2.0]])
    assert choose_stitch_index(chunk, prefix, D, search=0) == D


def test_stationary_prefix_does_not_crash():
    """A prefix with no motion gives no direction to prefer."""
    prefix = np.array([[1.0], [1.0]])
    chunk = np.array([[1.0], [1.0], [1.5], [2.0]])
    k = choose_stitch_index(chunk, prefix, D, search=2)
    assert D <= k < len(chunk)


def test_result_stays_within_the_chunk():
    prefix = np.array([[0.0], [1.0]])
    chunk = np.array([[0.0], [1.0], [1.5], [2.0]])
    k = choose_stitch_index(chunk, prefix, D, search=50)
    assert D <= k < len(chunk)
