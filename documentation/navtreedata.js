var NAVTREE =
[
  [ "Compute Library", "index.xhtml", [
    [ "Introduction", "index.xhtml", [
      [ "Contact / Support", "index.xhtml#S0_1_contact", null ],
      [ "Pre-built binaries", "index.xhtml#S0_2_prebuilt_binaries", null ],
      [ "File organisation", "index.xhtml#S1_file_organisation", null ],
      [ "Release versions and changelog", "index.xhtml#S2_versions_changelog", [
        [ "Release versions", "index.xhtml#S2_1_versions", null ],
        [ "Changelog", "index.xhtml#S2_2_changelog", null ]
      ] ],
      [ "How to build the library and the examples", "index.xhtml#S3_how_to_build", [
        [ "Build options", "index.xhtml#S3_1_build_options", null ],
        [ "Building for Linux", "index.xhtml#S3_2_linux", [
          [ "How to build the library ?", "index.xhtml#S3_2_1_library", null ],
          [ "How to manually build the examples ?", "index.xhtml#S3_2_2_examples", null ]
        ] ],
        [ "Building for Android", "index.xhtml#S3_3_android", [
          [ "How to build the library ?", "index.xhtml#S3_3_1_library", null ],
          [ "How to manually build the examples ?", "index.xhtml#S3_3_2_examples", null ]
        ] ],
        [ "Building for bare metal", "index.xhtml#S3_4_bare_metal", [
          [ "How to build the library ?", "index.xhtml#S3_4_1_library", null ],
          [ "How to manually build the examples ?", "index.xhtml#S3_4_2_examples", null ]
        ] ],
        [ "Building on a Windows host system", "index.xhtml#S3_5_windows_host", [
          [ "Bash on Ubuntu on Windows", "index.xhtml#S3_5_1_ubuntu_on_windows", null ],
          [ "Cygwin", "index.xhtml#S3_5_2_cygwin", null ]
        ] ],
        [ "The OpenCL stub library", "index.xhtml#S3_6_cl_stub_library", null ],
        [ "The Linux OpenGLES and EGL stub libraries", "index.xhtml#S3_7_gles_stub_library", null ]
      ] ]
    ] ],
    [ "Library architecture", "architecture.xhtml", [
      [ "Core vs Runtime libraries", "architecture.xhtml#S4_1", null ],
      [ "Windows, kernels, multi-threading and functions", "architecture.xhtml#S4_2_windows_kernels_mt_functions", [
        [ "Windows", "architecture.xhtml#S4_2_1_windows", null ],
        [ "Kernels", "architecture.xhtml#S4_2_2", null ],
        [ "Multi-threading", "architecture.xhtml#S4_2_3", null ],
        [ "Functions", "architecture.xhtml#S4_2_4", null ],
        [ "OpenCL Scheduler and kernel library", "architecture.xhtml#S4_4_1_cl_scheduler", null ],
        [ "OpenCL events and synchronization", "architecture.xhtml#S4_4_2_events_sync", null ],
        [ "OpenCL / NEON interoperability", "architecture.xhtml#S4_4_2_cl_neon", null ]
      ] ],
      [ "Algorithms", "architecture.xhtml#S4_5_algorithms", null ],
      [ "Images, padding, border modes and tensors", "architecture.xhtml#S4_6_images_tensors", [
        [ "Padding and border modes", "architecture.xhtml#S4_6_1_padding_and_border", [
          [ "Padding", "architecture.xhtml#padding", null ],
          [ "Valid regions", "architecture.xhtml#valid_region", null ]
        ] ],
        [ "Tensors", "architecture.xhtml#S4_6_2_tensors", null ],
        [ "Images and Tensors description conventions", "architecture.xhtml#S4_6_3_description_conventions", null ],
        [ "Working with Images and Tensors using iterators", "architecture.xhtml#S4_6_4_working_with_objects", null ]
      ] ],
      [ "MemoryManager", "architecture.xhtml#S4_7_memory_manager", [
        [ "MemoryGroup, MemoryPool and MemoryManager Components", "architecture.xhtml#S4_7_1_memory_manager_components", [
          [ "MemoryGroup", "architecture.xhtml#S4_7_1_1_memory_group", null ],
          [ "MemoryPool", "architecture.xhtml#S4_7_1_2_memory_pool", null ],
          [ "MemoryManager Components", "architecture.xhtml#S4_7_1_2_memory_manager_components", null ]
        ] ],
        [ "Working with the Memory Manager", "architecture.xhtml#S4_7_2_working_with_memory_manager", null ],
        [ "Function support", "architecture.xhtml#S4_7_3_memory_manager_function_support", null ]
      ] ],
      [ "OpenCL Tuner", "architecture.xhtml#S4_8_opencl_tuner", null ]
    ] ],
    [ "Validation and benchmarks tests", "tests.xhtml", [
      [ "Overview", "tests.xhtml#tests_overview", [
        [ "Directory structure", "tests.xhtml#tests_overview_structure", null ],
        [ "Fixtures", "tests.xhtml#tests_overview_fixtures", [
          [ "Fixture", "tests.xhtml#tests_overview_fixtures_fixture", null ],
          [ "Data fixture", "tests.xhtml#tests_overview_fixtures_data_fixture", null ]
        ] ],
        [ "Test cases", "tests.xhtml#tests_overview_test_cases", [
          [ "Test case", "tests.xhtml#tests_overview_test_cases_test_case", null ],
          [ "Fixture test case", "tests.xhtml#tests_overview_test_cases_fixture_fixture_test_case", null ],
          [ "Registering a fixture as test case", "tests.xhtml#tests_overview_test_cases_fixture_register_fixture_test_case", null ],
          [ "Data test case", "tests.xhtml#tests_overview_test_cases_data_test_case", null ],
          [ "Fixture data test case", "tests.xhtml#tests_overview_test_cases_fixture_data_test_case", null ],
          [ "Registering a fixture as data test case", "tests.xhtml#tests_overview_test_cases_register_fixture_data_test_case", null ]
        ] ]
      ] ],
      [ "Writing validation tests", "tests.xhtml#writing_tests", null ],
      [ "Running tests", "tests.xhtml#tests_running_tests", [
        [ "Benchmarking", "tests.xhtml#tests_running_tests_benchmarking", [
          [ "Filter tests", "tests.xhtml#tests_running_tests_benchmarking_filter", null ],
          [ "Runtime", "tests.xhtml#tests_running_tests_benchmarking_runtime", null ],
          [ "Output", "tests.xhtml#tests_running_tests_benchmarking_output", null ],
          [ "Mode", "tests.xhtml#tests_running_tests_benchmarking_mode", null ],
          [ "Instruments", "tests.xhtml#tests_running_tests_benchmarking_instruments", null ],
          [ "Examples", "tests.xhtml#tests_running_examples", null ]
        ] ]
      ] ]
    ] ],
    [ "Importing data from existing models", "data_import.xhtml", [
      [ "Extract data from pre-trained caffe model", "data_import.xhtml#caffe_data_extractor", [
        [ "How to use the script", "data_import.xhtml#caffe_how_to", null ],
        [ "What is the expected output from the script", "data_import.xhtml#caffe_result", null ]
      ] ],
      [ "Extract data from pre-trained tensorflow model", "data_import.xhtml#tensorflow_data_extractor", [
        [ "How to use the script", "data_import.xhtml#tensorflow_how_to", null ],
        [ "What is the expected output from the script", "data_import.xhtml#tensorflow_result", null ]
      ] ]
    ] ],
    [ "Namespaces", null, [
      [ "Namespace List", "namespaces.xhtml", "namespaces" ],
      [ "Namespace Members", "namespacemembers.xhtml", [
        [ "All", "namespacemembers.xhtml", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.xhtml", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.xhtml", null ],
        [ "Typedefs", "namespacemembers_type.xhtml", "namespacemembers_type" ],
        [ "Enumerations", "namespacemembers_enum.xhtml", null ],
        [ "Enumerator", "namespacemembers_eval.xhtml", null ]
      ] ]
    ] ],
    [ "Data Structures", "annotated.xhtml", [
      [ "Data Structures", "annotated.xhtml", "annotated_dup" ],
      [ "Data Structure Index", "classes.xhtml", null ],
      [ "Class Hierarchy", "hierarchy.xhtml", "hierarchy" ],
      [ "Data Fields", "functions.xhtml", [
        [ "All", "functions.xhtml", "functions_dup" ],
        [ "Functions", "functions_func.xhtml", "functions_func" ],
        [ "Variables", "functions_vars.xhtml", "functions_vars" ],
        [ "Typedefs", "functions_type.xhtml", null ],
        [ "Enumerations", "functions_enum.xhtml", null ],
        [ "Related Functions", "functions_rela.xhtml", null ]
      ] ]
    ] ],
    [ "Files", null, [
      [ "File List", "files.xhtml", "files" ],
      [ "Globals", "globals.xhtml", [
        [ "All", "globals.xhtml", "globals_dup" ],
        [ "Functions", "globals_func.xhtml", "globals_func" ],
        [ "Variables", "globals_vars.xhtml", null ],
        [ "Typedefs", "globals_type.xhtml", null ],
        [ "Macros", "globals_defs.xhtml", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_absolute_difference_8h.xhtml",
"_c_l_2_convolution_8cpp.xhtml#a38fe4b20a05bbaa1c844f3d7a19791ae",
"_c_l_2_pixel_wise_multiplication_8cpp.xhtml#ad3e50128989b47083e974e3567740655",
"_c_l_g_e_m_m_lowp_matrix_multiply_core_8h.xhtml",
"_c_l_unmap_8h.xhtml",
"_fixed_point_fixture_8h_source.xhtml",
"_graph_type_printer_8h.xhtml",
"_integral_image_8h_source.xhtml",
"_n_e_convolution_layer_8h_source.xhtml",
"_n_e_fixed_point_8h.xhtml#aed1eb26d2b6435ae9dba79558611b415",
"_n_e_im2_col_8h_source.xhtml",
"_n_e_o_n_2_depth_convert_layer_8cpp.xhtml#a3430329e55d600b885498ab229126ebe",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml#aa39faa5bc7636a7981279c09204fb4c7",
"_pixel_wise_multiplication_8h_source.xhtml",
"_type_printer_8h.xhtml#a2ab7370aab6e5698990c50c8871fa6fb",
"activation__layer_8cl.xhtml#ad3cc858846806e6b1d3694b9d0a2e6da",
"arm__compute_2core_2_types_8h.xhtml#ad8ed01ff3ff33333d8e19db4d2818bb6af14462d71aa842202c3e4b272c7ec924",
"benchmark_2_c_l_2_harris_corners_8cpp_source.xhtml",
"benchmark_2_n_e_o_n_2_g_e_m_m_8cpp.xhtml#a789c444c1307e85eec5f8b0d75fd5f7d",
"classarm__compute_1_1_activation_layer_info.xhtml#a63e05ce4946dd9807c005c1619fa337a",
"classarm__compute_1_1_c_l_copy_to_array_kernel.xhtml#a4b3ff5e7b109a563e4c98e45eb66ad7a",
"classarm__compute_1_1_c_l_g_e_m_m_lowp_matrix_multiply_kernel.xhtml",
"classarm__compute_1_1_c_l_kernel_library.xhtml#a293edc1bbcbddffb7228ac97917ed107",
"classarm__compute_1_1_c_l_pooling_layer_kernel.xhtml#a45eeb3e3b7cb65fe9c7eaaddbe994aa3",
"classarm__compute_1_1_c_l_tensor_allocator.xhtml#ae19b319fe4b1ddecacd35d45d9c4aa9b",
"classarm__compute_1_1_g_c_fully_connected_layer.xhtml#a68a5db52b1ed7f7ca1d619ac1263ab38",
"classarm__compute_1_1_h_o_g_info.xhtml#a0279e383beb758e477ce0673c5db8d57",
"classarm__compute_1_1_i_lut_allocator.xhtml#afc1c53ed4dcc1a723b9b9dcf67c578a1",
"classarm__compute_1_1_n_e_accumulate.xhtml",
"classarm__compute_1_1_n_e_depth_concatenate_layer_kernel.xhtml#ae3bbf74697684be2d67a4192f4451ef6",
"classarm__compute_1_1_n_e_g_e_m_m_interleave4x4.xhtml",
"classarm__compute_1_1_n_e_harris_score_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_n_e_pooling_layer_kernel.xhtml#a0e9871f265535f65226b65ae64ab2cfb",
"classarm__compute_1_1_n_e_winograd_layer_batched_g_e_m_m_kernel.xhtml#ae960c2433b8f9a63a5bc1ea61e20f705",
"classarm__compute_1_1_sub_tensor_info.xhtml#a7b6610e6100c67d39b712df6f4f80dcf",
"classarm__compute_1_1graph_1_1_i_tensor_object.xhtml#a99803b0c29444efee997abd9afd0c9ba",
"classarm__compute_1_1test_1_1_batch_normalization_layer_fixture.xhtml#a4adab6322a0276f34a7d656d49fc865c",
"classarm__compute_1_1test_1_1_tensor_cache.xhtml#a77d062d1fc4b239c215c4534deee8a2f",
"classarm__compute_1_1test_1_1datasets_1_1_optimized_depthwise_convolution_layer_dataset3x3.xhtml",
"classarm__compute_1_1test_1_1framework_1_1_instrument.xhtml#afc7df496e6026b91c3f48da6821c86a9",
"classarm__compute_1_1test_1_1networks_1_1_mobile_net_v1_network.xhtml",
"classarm__compute_1_1test_1_1validation_1_1_relative_tolerance.xhtml#afb9ded5f49336ae503bb9f2035ea902b",
"dir_9eca3c725728e372597fcd0d17c1ea0f.xhtml",
"fixed__point_8h.xhtml#adbd8b659ba077c1cceb5d560db65d4b9",
"graph__squeezenet__v1__1_8cpp.xhtml#a3c04138a5bfe5d72780bb7e82a18e627",
"index.xhtml#S3_7_gles_stub_library",
"neoncl__scale__median__gaussian_8cpp_source.xhtml",
"reference_2_pixel_wise_multiplication_8cpp.xhtml#af38a0302d7db0dcd7be6a2a444f2357f",
"structarm__compute_1_1_key_point.xhtml",
"structarm__compute_1_1test_1_1framework_1_1dataset_1_1_container_dataset_1_1iterator.xhtml#a7b5000e45c1386c4e56ef650f6b0ef5b",
"tests.xhtml#tests_running_tests_benchmarking_output",
"transpose_8cl.xhtml#af5dc9980a3e0aae2cba22e97b9458bc9",
"validation_2_c_l_2_normalization_layer_8cpp.xhtml#aa485ea6997e755f369f61ed9840fad92",
"validation_2_n_e_o_n_2_direct_convolution_layer_8cpp.xhtml#a254126d41d74b0728d68820637c24f13",
"validation_2reference_2_depth_concatenate_layer_8cpp.xhtml#ab9d829efe5d789bf902be0ecdc42ee7f"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';