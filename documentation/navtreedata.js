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
      ] ]
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
"_c_l_2_convolution_8cpp.xhtml#a97a9454a2282b20529391755a0b5c41a",
"_c_l_2_sobel_8cpp.xhtml#af0193468b2c7b02aa1d67b192ef8c220",
"_c_l_h_o_g_detector_kernel_8h_source.xhtml",
"_c_p_p_types_8h.xhtml#a5bc939238e1f14a4c39aaee8186a853da45a830a4c6240ac27c392266589a630c",
"_g_c_g_e_m_m_8h.xhtml",
"_i_array_8h.xhtml#a1ffe8af91a934c926d102938db8d3ce1",
"_median3x3_8h_source.xhtml",
"_n_e_direct_convolution_detail_8h.xhtml#af083008ad8a1e68fcc677c9186cc5f15",
"_n_e_fixed_point_8inl.xhtml#a9b9a80aab8862e344ef91591a0e6e199",
"_n_e_o_n_2_absolute_difference_8cpp.xhtml#ad9d2cfda8e6832a7a2b560e7b9af882d",
"_n_e_o_n_2_g_e_m_m_interleave4x4_8cpp_source.xhtml",
"_n_e_softmax_layer_kernel_8h.xhtml",
"_std_printer_8h.xhtml",
"_validation_8cpp.xhtml#a6c4e463dd04d06dc641fe48308e13334",
"arm__compute_2core_2_types_8h.xhtml#a673665b4587a2956fcbad5f0e9ba89d3ac9e68c0594494ec85805b225a7acb9c2",
"benchmark_2_c_l_2_depth_concatenate_layer_8cpp.xhtml#a785883955ec28a4da99cfb87d75dfa8e",
"benchmark_2_n_e_o_n_2_batch_normalization_layer_8cpp.xhtml#ac7369c169e6de526fcb6f68e4a959444",
"classarm__compute_1_1_access_window_auto_padding.xhtml#a55ae8ac3b55093b24fe30805828de164",
"classarm__compute_1_1_c_l_channel_extract_kernel.xhtml#af360eaecbadc6e66e1de099aa50c584d",
"classarm__compute_1_1_c_l_fill_border_kernel.xhtml#ac56c459beac7a6b904b6dece97b377e6",
"classarm__compute_1_1_c_l_histogram.xhtml#a2803ceaa34ef2077846e2eae3ae85d10",
"classarm__compute_1_1_c_l_normalization_layer_kernel.xhtml",
"classarm__compute_1_1_c_l_symbols.xhtml#a8f72204edecff30764f23d06c4e4ae20",
"classarm__compute_1_1_g_c_dropout_layer.xhtml",
"classarm__compute_1_1_h_o_g.xhtml#a548b8c6fda024da07113ff6cf6ec6af6",
"classarm__compute_1_1_i_lut_allocator.xhtml#a4ef157737b45c46f115e14b425512800",
"classarm__compute_1_1_n_e_activation_layer_kernel.xhtml",
"classarm__compute_1_1_n_e_depthwise_separable_convolution_layer.xhtml",
"classarm__compute_1_1_n_e_g_e_m_m_matrix_accumulate_biases_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_n_e_logits1_d_shift_exp_sum_kernel.xhtml#a541f8b17ae7011090ec7129487e61d04",
"classarm__compute_1_1_n_e_sobel5x5_hor_kernel.xhtml#a423f9a45a52983b4de5e2b347f4369c7",
"classarm__compute_1_1_semaphore.xhtml",
"classarm__compute_1_1graph_1_1_c_l_map.xhtml#a4c07c8a2c24bb60558618ae97bfd0ac0",
"classarm__compute_1_1test_1_1_accessor.xhtml#abdd3637f2bbde9d7d0cc0b7bbd8400bb",
"classarm__compute_1_1test_1_1_raw_lut_accessor.xhtml#a5fbc8143d612bfca981fa8639649fd60",
"classarm__compute_1_1test_1_1datasets_1_1_large_fully_connected_layer_dataset.xhtml#a67e061497f9e20d506a0016a13f76829",
"classarm__compute_1_1test_1_1framework_1_1_framework.xhtml#a5f16ced78873ee3a193490197fbf57a5",
"classarm__compute_1_1test_1_1networks_1_1_mobile_net_network.xhtml",
"classarm__compute_1_1test_1_1validation_1_1_softmax_validation_generic_fixture.xhtml",
"dir_f7024513cd67abef53e86ee9382ac5ce.xhtml",
"functions_d.xhtml",
"helpers_8h.xhtml#a6ded2cf071c127e518317e3c451af3ef",
"namespacemembers_e.xhtml",
"reference_2_dilate_8cpp_source.xhtml",
"structarm__compute_1_1_border_size.xhtml#adffbf97e7b8b64e7cf32f0254cddf3c4",
"structarm__compute_1_1test_1_1framework_1_1_test_result.xhtml#a67a0db04d321a74b7e7fcfd3f1a3f70b",
"tests_2_utils_8h.xhtml#ac7324cc960068b65c558b7d25dfe2914",
"utils_2_utils_8h_source.xhtml",
"validation_2_c_l_2_scale_8cpp.xhtml#aed4fea76ee6b2a8eede8dcacd2ac6c43",
"validation_2_n_e_o_n_2_g_e_m_m_lowp_8cpp.xhtml#a81a5e6a019eb49bf186adbae0baa7325",
"validation_2reference_2_normalization_layer_8cpp.xhtml#a451c44627ad6c06f72667812f2a9782d"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';