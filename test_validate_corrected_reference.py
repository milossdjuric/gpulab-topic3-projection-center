"""Regression test: validate_forward_search.py's --fix-ref path
must normalize MSE by the count of valid (on-detector, signal-bearing)
sample pairs, the same way the GPU kernels do -- not by a fixed count of N,
which was the reference's own behavior even after the out-of-bounds fix.
Without this, a --fix-ref comparison structurally fails the MSE
tolerance check even when the GPU is correct, since it's comparing two
different metric definitions.

This test hand-derives an exact expected value by choosing alpha=beta=0,
which collapses the reference's pixel-projection formula to something
computable by hand: y is always exactly the detector's vertical center
(no interpolation blending across rows), and x = center - tan(dtheta),
rx = center + tan(dtheta) exactly. Picking dtheta values with known tan()
results (0, 1, 2, 6, 3) makes every sample's pixel coordinates exact
integers, so there's no bilinear-interpolation rounding to account for
either -- every value in this test is exact, not approximate."""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from validate_forward_search import _corrected_get_linear_interpolate_MSE

SDD = 1.0
PIXEL_SIZE = 1.0
DETECTOR_WIDTH = 11   # valid x range: 0 <= x < 10
DETECTOR_HEIGHT = 3   # valid y range: 0 <= y < 2; y is always exactly 1 here
ALPHA = 0.0
BETA = 0.0
THETA_0 = 0.0

# dtheta values chosen so tan(dtheta) == 0, 1, 2, 6, 3 respectively, giving
# exact integer pixel coordinates x = 5 - tan(dtheta), rx = 5 + tan(dtheta):
#   dtheta=atan(0): x=5,  rx=5   -> in bounds, sino[1][5]=0.6  vs sino[1][5]=0.6  -> diff^2 = 0.00
#   dtheta=atan(1): x=4,  rx=6   -> in bounds, sino[1][4]=0.5  vs sino[1][6]=0.4  -> diff^2 = 0.01
#   dtheta=atan(2): x=3,  rx=7   -> in bounds, sino[1][3]=0.3  vs sino[1][7]=0.5  -> diff^2 = 0.04
#   dtheta=atan(6): x=-1, rx=11  -> BOTH out of bounds -> excluded
#   dtheta=atan(3): x=2,  rx=8   -> in bounds geometrically, but sino[1][2]=sino[1][8]=0.005,
#                                    both below MIN_SIGNAL=0.01 -> excluded (background-only pair)
NEAREST_THETA = np.array([
    math.atan(0.0),
    math.atan(1.0),
    math.atan(2.0),
    math.atan(6.0),
    math.atan(3.0),
], dtype=np.float64)
N = len(NEAREST_THETA)

SINO = np.zeros((DETECTOR_HEIGHT, DETECTOR_WIDTH), dtype=np.float64)
SINO[1] = [0.5, 0.5, 0.005, 0.3, 0.5, 0.6, 0.4, 0.5, 0.005, 0.5, 0.5]

# 3 valid, signal-bearing pairs out of 5 total samples.
EXPECTED_VALID_COUNT = 3
EXPECTED_SUM_SQUARED_DIFFS = 0.0 + 0.01 + 0.04  # = 0.05
EXPECTED_MSE = EXPECTED_SUM_SQUARED_DIFFS / EXPECTED_VALID_COUNT  # matches the GPU formula
OLD_BUGGY_MSE = EXPECTED_SUM_SQUARED_DIFFS / N  # what the un-fixed normalization gave


def test_valid_pair_normalization():
    pixel_mse = _corrected_get_linear_interpolate_MSE(
        N, SINO, SDD, DETECTOR_WIDTH, DETECTOR_HEIGHT, PIXEL_SIZE,
        ALPHA, BETA, THETA_0, NEAREST_THETA,
    )
    # Mirrors exactly what find_conebeam_COR_line_forward does with the
    # returned array: MSE = pixel_MSE.sum(axis=0) / N
    mse = pixel_mse.sum(axis=0) / N

    assert math.isclose(mse, EXPECTED_MSE, rel_tol=1e-9), (
        f"expected MSE normalized by valid_count={EXPECTED_VALID_COUNT} "
        f"({EXPECTED_MSE!r}), got {mse!r} "
        f"(old fixed-N normalization would give {OLD_BUGGY_MSE!r})"
    )
    assert not math.isclose(mse, OLD_BUGGY_MSE, rel_tol=1e-9), (
        "MSE matches the old fixed-N normalization -- the fix isn't being applied"
    )
    print(f"test_valid_pair_normalization: PASS  mse={mse!r} (old={OLD_BUGGY_MSE!r})")


def test_zero_valid_pairs_gives_inf():
    # Same geometry, but every sample pushed out of bounds (large tan values) --
    # a candidate that excludes every ray must score as the worst possible
    # combo (+inf), not 0.0 (which would look like a perfect match and
    # reintroduce the exact bug the exclusion fix exists to prevent).
    all_out_of_bounds_theta = np.array([math.atan(6.0), math.atan(7.0)], dtype=np.float64)
    n = len(all_out_of_bounds_theta)
    pixel_mse = _corrected_get_linear_interpolate_MSE(
        n, SINO, SDD, DETECTOR_WIDTH, DETECTOR_HEIGHT, PIXEL_SIZE,
        ALPHA, BETA, THETA_0, all_out_of_bounds_theta,
    )
    mse = pixel_mse.sum(axis=0) / n
    assert math.isinf(mse), f"expected +inf for zero valid pairs, got {mse!r}"
    print(f"test_zero_valid_pairs_gives_inf: PASS  mse={mse!r}")


if __name__ == "__main__":
    test_valid_pair_normalization()
    test_zero_valid_pairs_gives_inf()
