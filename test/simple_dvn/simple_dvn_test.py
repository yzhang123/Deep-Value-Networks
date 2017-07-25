
import dvn.src.simple_dvn.simple_dvn as simple_dvn
import numpy as np

def test_oracle_score_works():
    gt_mask = np.zeros([1, 10, 10, 2])
    mask = np.zeros([1, 10, 10, 2])
    mask[..., 0] = 1.
    expected_score = [0.5]
    score = simple_dvn.oracle_score(mask, gt_mask)
    assert score == expected_score

    gt_mask = np.zeros([2, 10, 10, 2])
    mask = np.zeros([2, 10, 10, 2])
    mask[..., 0] = 1.
    expected_score = [0.5, 0.5]
    score = simple_dvn.oracle_score(mask, gt_mask)
    assert (score == expected_score).all



def test_oracle_score_zero():
    gt_mask = np.zeros([2, 10, 10, 2])
    gt_mask[..., 1] = 1.
    mask = np.zeros([2, 10, 10, 2])
    mask[..., 0] = 1.

    expected_score = [0., 0.]
    score = simple_dvn.oracle_score(mask, gt_mask)
    assert (score == expected_score).all

# def test_oracle_score_cpu():
#     gt_mask = np.zeros([1, 10, 10, 2])
#     mask = np.zeros([1, 10, 10, 2])
#     expected_score = [1]
#     score = simple_dvn.oracle_score(mask, gt_mask)
#     assert score == expected_score
#
#     gt_mask = np.zeros([2, 10, 10, 2])
#     mask = np.zeros([2, 10, 10, 2])
#     expected_score = [1, 1]
#     score = simple_dvn.oracle_score(mask, gt_mask)
#     assert (score == expected_score).all