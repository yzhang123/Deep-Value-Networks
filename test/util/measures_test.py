import numpy as np
from dvn.src.util.measures import oracle_score

class Test_Oracle_Score:

    def test_oracle_score(self):
        gt_mask = np.zeros([1, 10, 10, 2])
        mask = np.zeros([1, 10, 10, 2])
        mask[..., 0] = 1.
        expected_score = [0.5]
        score = oracle_score(mask, gt_mask)
        assert score == expected_score


        gt_mask = np.zeros([2, 10, 10, 2])
        mask = np.zeros([2, 10, 10, 2])
        mask[..., 0] = 1.
        expected_score = [0.5, 0.5]
        score = oracle_score(mask, gt_mask)
        assert (score == expected_score).all


    def test_oracle_score(self):
        gt_mask = np.zeros([1, 10, 10, 2])
        mask = np.zeros([1, 10, 10, 2])
        expected_score = [1]
        score = oracle_score(mask, gt_mask)
        assert score == expected_score


        gt_mask = np.zeros([2, 10, 10, 2])
        mask = np.zeros([2, 10, 10, 2])
        expected_score = [1, 1]
        score = oracle_score(mask, gt_mask)
        assert (score == expected_score).all