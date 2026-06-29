project('forward_search', 'cpp', default_options: ['cpp_std=c++17'])

opencl = dependency('OpenCL')
hdf5   = dependency('hdf5', language: 'cpp')
boost  = dependency('boost', modules: ['program_options'])

executable('forward_search',
  ['main.cpp', 'forward_search.cpp'],
  dependencies: [opencl, hdf5, boost])
