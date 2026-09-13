project('topic_3_forward_search', 'cpp', default_options: ['cpp_std=c++17', 'buildtype=release'])

# Root-level wrapper so the whole repo (search, resample, and the combined
# pipeline) can be opened as a single KDevelop project from Topic_3/ itself,
# for the course's KDevelop packaging requirement. projection-center-cpp/meson.build
# stays independently buildable on its own (CI, tests, and RUNNING.md all
# build there directly) -- this file re-declares the same three targets
# against projection-center-cpp/src/ rather than subdir()-ing into it, since Meson
# only allows one project() call per build and projection-center-cpp/meson.build
# needs to keep its own.

opencl = dependency('OpenCL')
hdf5   = dependency('hdf5', language: 'cpp')
boost  = dependency('boost', modules: ['program_options'])

cpp = meson.get_compiler('cpp')
boost_json_lib = cpp.find_library('boost_json-mt', required: false)
if not boost_json_lib.found()
  boost_json_lib = cpp.find_library('boost_json', required: true)
endif
boost_json = declare_dependency(dependencies: boost_json_lib)

executable('forward_search',
  ['projection-center-cpp/src/main.cpp', 'projection-center-cpp/src/forward_search.cpp'],
  dependencies: [opencl, hdf5, boost])

executable('forward_search_resample',
  ['projection-center-cpp/src/resample_main.cpp', 'projection-center-cpp/src/resample.cpp'],
  dependencies: [opencl, hdf5, boost, boost_json])

executable('projection_center_pipeline_cpp',
  ['projection-center-cpp/src/pipeline_main.cpp', 'projection-center-cpp/src/forward_search.cpp', 'projection-center-cpp/src/resample.cpp'],
  dependencies: [opencl, hdf5, boost])
