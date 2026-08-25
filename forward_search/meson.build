project('forward_search', 'cpp', default_options: ['cpp_std=c++17', 'buildtype=release'])

opencl = dependency('OpenCL')
hdf5   = dependency('hdf5', language: 'cpp')
boost  = dependency('boost', modules: ['program_options'])
# Boost.JSON isn't header-only in this Boost version; link it explicitly
# (meson's boost dependency() doesn't recognize 'json' as a module name the
# way it does 'program_options') -- used only by resample_main.cpp to parse
# the pose JSON either forward-search implementation writes.
boost_json = declare_dependency(link_args: ['-lboost_json'])

executable('forward_search',
  ['main.cpp', 'forward_search.cpp'],
  dependencies: [opencl, hdf5, boost])

executable('forward_search_resample',
  ['resample_main.cpp', 'resample.cpp'],
  dependencies: [opencl, hdf5, boost, boost_json])
