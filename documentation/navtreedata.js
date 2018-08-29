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
        [ "The Linux OpenGLES and EGL stub libraries", "index.xhtml#S3_7_gles_stub_library", null ],
        [ "OpenCL DDK Requirements", "index.xhtml#S3_8_cl_requirements", [
          [ "Hard Requirements", "index.xhtml#S3_8_1_cl_hard_requirements", null ],
          [ "Performance improvements", "index.xhtml#S3_8_2_cl_performance_requirements", null ]
        ] ],
        [ "OpenCL Tuner", "index.xhtml#S3_9_cl_tuner", [
          [ "How to use it", "index.xhtml#S3_9_1_cl_tuner_how_to", null ]
        ] ]
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
        [ "Working with Images and Tensors using iterators", "architecture.xhtml#S4_6_4_working_with_objects", null ],
        [ "Sub-tensors", "architecture.xhtml#S4_6_5_sub_tensors", null ]
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
        [ "Benchmarking and validation suites", "tests.xhtml#tests_running_tests_benchmark_and_validation", [
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
        [ "Enumerations", "namespacemembers_enum.xhtml", null ]
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
"_c_l_2_channel_shuffle_8cpp.xhtml#ac2804db9b7a06b5b9749159349fb41ee",
"_c_l_2_u_n_i_t_2_tensor_allocator_8cpp.xhtml#a567a0fd1d570601eeb9a3ea311a8f82f",
"_c_l_device_backend_8h.xhtml",
"_c_l_scharr3x3_8h.xhtml",
"_dataset_modes_8h.xhtml",
"_g_c_g_e_m_m_matrix_accumulate_biases_kernel_8h.xhtml",
"_i_array_8h.xhtml#a276401fd5651e75431d3e1cc90601caa",
"_log_msg_decorators_8h.xhtml",
"_n_e_direct_convolution_detail_8h.xhtml#a96ebd967ad3ef1c2fcfaed80cab10567",
"_n_e_o_n_2_arithmetic_subtraction_8cpp.xhtml#a9457361de80f40bb3fdfefee5b7c3841",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml#aa670d48d66ab33363db0a492e2453e81",
"_phase_8h_source.xhtml",
"_type_reader_8h.xhtml#a1379e5f8a8b4640d7a52d901f0539ccd",
"and_8h.xhtml#ac071623f14587f7cc6c4aea757d4378b",
"arm__compute_2core_2_utils_8h.xhtml#afa7962c747457714d0944af80cb07058",
"benchmark_2_c_l_2_fast_corners_8cpp.xhtml#a03d6956873131b3bdca39600a58f905a",
"benchmark_2_g_l_e_s___c_o_m_p_u_t_e_2_fully_connected_layer_8cpp.xhtml#a300e663e27206e0b4c86cb86b5c79227",
"benchmark_2_n_e_o_n_2_laplacian_pyramid_8cpp.xhtml#a833c2620a79106b76c9e1edab3e7769c",
"classarm__compute_1_1_c_l_absolute_difference_kernel.xhtml#a493987e85723a8000eb26d1f00e2ad0e",
"classarm__compute_1_1_c_l_convolution_rectangle.xhtml",
"classarm__compute_1_1_c_l_fine_s_v_m_memory_region.xhtml#af540d8670cb2ccdec5a30fd2471c2e85",
"classarm__compute_1_1_c_l_h_o_g_detector_kernel.xhtml#ac8f1ca778b425c6408a93b672b041dd0",
"classarm__compute_1_1_c_l_median3x3_kernel.xhtml",
"classarm__compute_1_1_c_l_sobel3x3_kernel.xhtml#a493987e85723a8000eb26d1f00e2ad0e",
"classarm__compute_1_1_c_p_p_detection_window_non_maxima_suppression_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_g_c_g_e_m_m.xhtml",
"classarm__compute_1_1_g_e_m_m_reshape_info.xhtml#aee6f5a043173c4d51c11a54db8e0f519",
"classarm__compute_1_1_i_g_c_tensor.xhtml#aa49cf1d6ea350774509911cad4903ad6",
"classarm__compute_1_1_lut.xhtml#a6dc9c052a2732f899adccfaeee468364",
"classarm__compute_1_1_n_e_col2_im_kernel.xhtml#adad68aacb0ee78bab2f9079780368dd7",
"classarm__compute_1_1_n_e_edge_non_max_suppression_kernel.xhtml#a57a36344e261edfdfa97b19ac40b7c30",
"classarm__compute_1_1_n_e_g_e_m_m_matrix_vector_multiply_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_magnitude_phase_kernel.xhtml#aebf700d44017c1ff38ad0d3741aaac56",
"classarm__compute_1_1_n_e_separable_convolution_hor_kernel.xhtml#a423f9a45a52983b4de5e2b347f4369c7",
"classarm__compute_1_1_pixel_value.xhtml#a2abd12bde55e3887e34e030e8352e06d",
"classarm__compute_1_1_tensor_info.xhtml#a6e1c061accc47c7fa11d0f014d3795f9",
"classarm__compute_1_1graph_1_1_graph.xhtml#a6ebebd7e246eb58254aa2cff70fa7242",
"classarm__compute_1_1graph_1_1backends_1_1_c_l_sub_tensor_handle.xhtml#a3a3488108174600975461132e1e3cc03",
"classarm__compute_1_1graph__utils_1_1_validation_output_accessor.xhtml",
"classarm__compute_1_1test_1_1_padding_calculator.xhtml#a0e65ad13124ea2cb5e255b640464e35facdccee7243b006a86ce2e3f33795b5be",
"classarm__compute_1_1test_1_1framework_1_1_scheduler_timer.xhtml#ad94b6b2e29abc159c9624278938badb3",
"color__convert_8cl.xhtml#a47fa13a636a45eaa621bdbcbb206895f",
"functions_func_d.xhtml",
"helpers_8h.xhtml#a7e4940407322d6f0ccb8b6b86b856019",
"namespacearm__compute_1_1support.xhtml",
"reference_2_accumulate_8cpp.xhtml#a80107badc51d92c79886955f52c61a1b",
"sobel__filter_8cl.xhtml#ad40297671e8a51899c6114386492006b",
"structarm__compute_1_1detail_1_1brelu.xhtml#a866d1f31a29495f573ed758878f9028a",
"tests.xhtml#tests_overview",
"utils_2_type_printer_8h.xhtml#a8af36ae3a3613112c3a95e57f606359a",
"validation_2_c_l_2_direct_convolution_layer_8cpp.xhtml#ac7815e2d8e02049e895b982d8415ba2c",
"validation_2_c_l_2_reshape_layer_8cpp_source.xhtml",
"validation_2_n_e_o_n_2_convolution_layer_8cpp.xhtml#a2fdbbbe8bfaeed577986acf27b6f1389",
"validation_2_n_e_o_n_2_scale_8cpp.xhtml",
"validation_2reference_2_scale_8cpp.xhtml#a3ed6bfc6e3b5a03db71f1ab3f32b18fb"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';