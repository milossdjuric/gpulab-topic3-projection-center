#!/bin/bash
# Sequential run of every script/command in this project against
# data/proj_shepplogan128.hdf5. Each step prints its own header so output
# is easy to scan; nothing here uses `set -e`, since validate_forward_search.py
# is expected to exit nonzero on a real FAIL, not just a script error.

DATA="data/proj_shepplogan128.hdf5"
OUT="runs/all_128"
mkdir -p "$OUT"

# Everything below runs inside main() so its combined stdout/stderr can be
# piped through tee at the bottom: this writes the full run to $OUT/run.log
# while still printing live to the terminal.
main() {

echo "################################################################"
echo "# 1. projection-center devices"
echo "################################################################"
projection-center devices

echo
echo "################################################################"
echo "# 2. projection-center search --backend cpp (this repo's C++/OpenCL"
echo "#    binary, invoked through the CLI wrapper so the pose comes out as"
echo "#    JSON -- the raw forward_search binary's --output only writes"
echo "#    HDF5, and the raw forward_search_resample binary's --pose only"
echo "#    reads JSON, so the two raw binaries can't be chained by hand)"
echo "################################################################"
projection-center search --backend cpp --cpp-mode buffer --data "$DATA" \
  --output-pose "$OUT/cli_pose.json"

echo
echo "################################################################"
echo "# 3. projection-center resample --backend cpp"
echo "################################################################"
projection-center resample --backend cpp --cpp-mode buffer --data "$DATA" \
  --pose "$OUT/cli_pose.json" --output-data "$OUT/cli_resampled.hdf5"

echo
echo "################################################################"
echo "# 4. projection-center pipeline --backend cpp (search + resample in one"
echo "#    command, instead of steps 2/3's separate search/resample calls)"
echo "################################################################"
projection-center pipeline --backend cpp --cpp-mode buffer --data "$DATA" \
  --output-pose "$OUT/pipeline_pose.json" --output-data "$OUT/pipeline_resampled.hdf5"

echo
echo "################################################################"
echo "# 5. benchmark_backends.py -- reference (search+resample) + opencl/cpu/cpp (pipeline)"
echo "#    NOTE: the RAW reference is known to crash outright on this dataset's"
echo "#    default search range (UnboundLocalError -- the out-of-bounds bug hits"
echo "#    the very first sample of the very first, most extreme candidate, with"
echo "#    no prior iteration to silently inherit a stale value from). This is a"
echo "#    pre-existing characteristic of the untouched reference on this small"
echo "#    128px dataset, not something this run introduces -- benchmark_backends.py"
echo "#    catches it and continues with opencl/cpu/cpp below."
echo "################################################################"
python3 benchmark_backends.py --data "$DATA" --output-dir "$OUT/benchmark"

echo
echo "################################################################"
echo "# 6. validate_forward_search.py -- GPU pose vs. a fresh, bug-fixed reference run"
echo "#    (--fix-ref patches the reference in-process only, never touching the"
echo "#    file; this patched version does not have the crash above, since it"
echo "#    zero-initializes before the bounds check instead of leaving it unset)"
echo "################################################################"
python3 validate_forward_search.py \
  --gpu "$OUT/benchmark/opencl/pose.json" \
  --run-ref --fix-ref --data "$DATA"

echo
echo "################################################################"
echo "# 7. Compare resampled outputs across backends, full dataset"
echo "################################################################"
python3 -c "
import h5py, numpy as np

pairs = [('opencl', 'cpu'), ('opencl', 'cpp'), ('cpu', 'cpp')]
for a_name, b_name in pairs:
    a = h5py.File('$OUT/benchmark/' + a_name + '/resampled.hdf5')['Projection'][:].astype(np.float64)
    b = h5py.File('$OUT/benchmark/' + b_name + '/resampled.hdf5')['Projection'][:].astype(np.float64)
    d = np.abs(a - b)
    print(f'{a_name} vs {b_name}: max={d.max():.6e}  mean={d.mean():.6e}  median={np.median(d):.6e}')
"

echo
echo "################################################################"
echo "# 8. Screenshots: original + each backend's resampled projection 0"
echo "################################################################"
python3 -c "
import h5py, numpy as np
from PIL import Image

def save(path, out):
    p = h5py.File(path)['Projection'][0]
    n = (p - p.min()) / (p.max() - p.min())
    Image.fromarray((n * 255).astype(np.uint8)).save(out)

save('$DATA', '$OUT/projection_0_original.png')
for name in ('opencl', 'cpu', 'cpp'):
    save('$OUT/benchmark/' + name + '/resampled.hdf5', '$OUT/projection_0_' + name + '.png')
print('screenshots saved under $OUT/')
"

echo
echo "################################################################"
echo "# DONE. All outputs under $OUT/"
echo "################################################################"

}

main "$@" 2>&1 | tee "$OUT/run.log"
