"""Regression test: validate_forward_search.py's compare() must recognize
the sign-flip symmetry in the search's geometry model. Negating both alpha
and beta simultaneously leaves xshift, center_x, MSE, and theta_0 exactly
unchanged (proved algebraically and confirmed numerically against the real
reference code), and only flips center_y's sign. On a dataset whose
sinogram is close to vertically symmetric (confirmed on
data/proj_shepplogan512.hdf5: correlation(sino, vertically-flipped sino) =
1.000000), (alpha, beta) and (-alpha, -beta) are both genuinely valid,
near-equally-scoring solutions, not a real mismatch -- reference found
alpha=+1deg beta=-10deg while opencl/cpu/cpp all found alpha=-1deg
beta=+10deg, with matching xshift/center_x and MSE differing by 0.0286%.

Without this fix, compare() reports a false FAIL on exactly this case."""
import math
import sys

sys.path.insert(0, ".")
from validate_forward_search import compare

XSHIFT_STEP_M = 0.001
ALPHA_STEP_RAD = math.radians(1.0)
BETA_STEP_RAD = math.radians(1.0)


def test_mirrored_pose_passes():
    gpu = {
        "xshift": 0.035, "alpha": math.radians(-1.0), "beta": math.radians(10.0),
        "MSE": 1.74708344e-04, "center_point": [0.22008997, 0.20769143],
    }
    ref = {
        "xshift": 0.035, "alpha": math.radians(1.0), "beta": math.radians(-10.0),
        "MSE": 1.74708344e-04, "center_point": [0.22008997, -0.20769143],
    }
    ok = compare(gpu, ref, xshift_step_m=XSHIFT_STEP_M,
                 alpha_step_rad=ALPHA_STEP_RAD, beta_step_rad=BETA_STEP_RAD)
    assert ok, "expected PASS: gpu and ref are exact mirror-symmetric poses"
    print("test_mirrored_pose_passes: PASS")


def test_genuine_mismatch_still_fails():
    # Sanity check: the mirror check must not become a blanket pass -- a
    # pose that is neither a direct nor a mirrored match must still fail.
    gpu = {
        "xshift": 0.035, "alpha": math.radians(-1.0), "beta": math.radians(10.0),
        "MSE": 1.74708344e-04, "center_point": [0.22008997, 0.20769143],
    }
    ref = {
        "xshift": 0.035, "alpha": math.radians(5.0), "beta": math.radians(-3.0),
        "MSE": 5.0e-03, "center_point": [0.1, -0.05],
    }
    ok = compare(gpu, ref, xshift_step_m=XSHIFT_STEP_M,
                 alpha_step_rad=ALPHA_STEP_RAD, beta_step_rad=BETA_STEP_RAD)
    assert not ok, "expected FAIL: gpu and ref are neither an exact nor mirrored match"
    print("test_genuine_mismatch_still_fails: PASS")


def test_direct_match_still_passes():
    # Sanity check: an ordinary, non-mirrored exact match must keep working.
    gpu = {
        "xshift": 0.035, "alpha": math.radians(-1.0), "beta": math.radians(10.0),
        "MSE": 1.74708344e-04, "center_point": [0.22008997, 0.20769143],
    }
    ref = dict(gpu)
    ref["center_point"] = list(gpu["center_point"])
    ok = compare(gpu, ref, xshift_step_m=XSHIFT_STEP_M,
                 alpha_step_rad=ALPHA_STEP_RAD, beta_step_rad=BETA_STEP_RAD)
    assert ok, "expected PASS: gpu and ref are an exact direct match"
    print("test_direct_match_still_passes: PASS")


if __name__ == "__main__":
    test_mirrored_pose_passes()
    test_genuine_mismatch_still_fails()
    test_direct_match_still_passes()
