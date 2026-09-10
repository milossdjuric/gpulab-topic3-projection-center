project('forward_search', 'cpp', default_options: ['cpp_std=c++17', 'buildtype=release'])

opencl = dependency('OpenCL')
hdf5   = dependency('hdf5', language: 'cpp')
boost  = dependency('boost', modules: ['program_options'])
# Boost.JSON isn't header-only in this Boost version; link it explicitly
# (meson's boost dependency() doesn't recognize 'json' as a module name the
# way it does 'program_options') -- used only by resample_main.cpp to parse
# the pose JSON either forward-search implementation writes.
#
# Library name isn't portable: Linux/apt names it plain boost_json, but
# MSYS2/MinGW builds Boost with a "-mt" (multi-threaded) suffix on every
# component, same as boost_program_options-mt above -- so try that name
# first and fall back to the plain one, instead of hardcoding either.
cpp = meson.get_compiler('cpp')
boost_json_lib = cpp.find_library('boost_json-mt', required: false)
if not boost_json_lib.found()
  boost_json_lib = cpp.find_library('boost_json', required: true)
endif
boost_json = declare_dependency(dependencies: boost_json_lib)

executable('forward_search',
  ['src/main.cpp', 'src/forward_search.cpp'],
  dependencies: [opencl, hdf5, boost])

executable('forward_search_resample',
  ['src/resample_main.cpp', 'src/resample.cpp'],
  dependencies: [opencl, hdf5, boost, boost_json])

# Single-process search + resample: loads the HDF5 data once and passes the
# pose in memory instead of through the two binaries above and a JSON file
# in between. See src/pipeline_main.cpp's own comment for what this does
# and does not eliminate.
executable('forward_search_pipeline',
  ['src/pipeline_main.cpp', 'src/forward_search.cpp', 'src/resample.cpp'],
  dependencies: [opencl, hdf5, boost])
