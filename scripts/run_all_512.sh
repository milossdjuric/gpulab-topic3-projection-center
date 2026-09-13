#!/bin/bash
# Sequential run of every script/command in this project against
# data/proj_shepplogan512.hdf5.
#
# NOTE (fixed 2026-09-09): this dataset's own detector_width/detector_height
# scalars say 800/500, but the Projection array's real shape is
# (1000, 800, 500) -- swapped relative to those two scalars. The file on
# disk was never changed; instead, main.cpp/resample_main.cpp's loadHDF5()
# and validate_forward_search.py's _load_cb_para() now derive
# detector_width/detector_height from Projection's real shape (matching
# how projection-center's own hdf5_io.py already worked), so every step
# below now sees consistent dimensions regardless of what those two
# scalars say.
#
# The one thing this can't fix: step 4's raw, unmodified
# Topic_3_forwardsearching.py reference still trusts the file's stored
# scalars directly and can't be modified, so its search will still fail on
# this dataset -- benchmark_backends.py already catches that and continues
# with opencl/cpu/cpp below, same as it does on the 128px dataset for an
# unrelated reason (see step 4's own note there).
#
# Nothing here uses `set -e`, since validate_forward_search.py is expected
# to exit nonzero on a real FAIL, not just a script error.

DATA="data/proj_shepplogan512.hdf5"
OUT="runs/all_512"
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
echo "#    The reference search is expected to fail here (see the note at the"
echo "#    top of this file); benchmark_backends.py catches that and continues"
echo "#    with opencl/cpu/cpp. On a dataset this size (1000 x 800 x 500), those"
echo "#    three backends' own resample steps still take a while."
echo "################################################################"
python3 benchmark_backends.py --data "$DATA" --output-dir "$OUT/benchmark"

echo
echo "################################################################"
echo "# 6. validate_forward_search.py -- GPU pose vs. a fresh, bug-fixed reference run"
echo "#    (--fix-ref patches the reference in-process only, file untouched)"
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
