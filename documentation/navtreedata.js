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
          [ "Instruments", "tests.xhtml#tests_running_tests_benchmarking_instruments", null ]
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
        [ "Enumerations", "globals_enum.xhtml", null ],
        [ "Enumerator", "globals_eval.xhtml", null ],
        [ "Macros", "globals_defs.xhtml", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_absolute_difference_8h.xhtml",
"_c_l_2_depth_convert_layer_8cpp.xhtml#a8033cc53b2e8fd27f83544541930501c",
"_c_l_2_transpose_8cpp.xhtml#a76d9523853eb66c09dac14d748b6a831",
"_c_l_helpers_8h.xhtml#ae5f4dd9f672832f7dce02fc14487f544",
"_command_line_parser_8h.xhtml",
"_g_c_pixel_wise_multiplication_kernel_8h_source.xhtml",
"_i_c_l_simple3_d_kernel_8h_source.xhtml",
"_mobile_net_v1_network_8h_source.xhtml",
"_n_e_fixed_point_8h.xhtml#a252c74fae9a9238ef7c23d306e670706",
"_n_e_fixed_point_8inl.xhtml#ae236b9df08e568bb4b1dcc89896cff7e",
"_n_e_o_n_2_arithmetic_subtraction_8cpp.xhtml#aab1082ad89cbf2f61edc333ab00d524d",
"_n_e_o_n_2_non_linear_filter_8cpp.xhtml#adb5d3c20c5232c33ef76501d67210c88",
"_non_linear_filter_fixture_8h.xhtml",
"_threshold_8h_source.xhtml",
"a32__merge__float__8x6_8hpp.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a1ce9b523fd4f3b5bbcadcd796183455aadfcf28d0734569a6a693bc8194de62bf",
"benchmark_2_c_l_2_activation_layer_8cpp.xhtml",
"benchmark_2_n_e_o_n_2_g_e_m_m_lowp_8cpp.xhtml#ae32f1e3c34f40106570812eed538aa3a",
"classarm__compute_1_1_blob_memory_pool.xhtml#ae3f7b519f24156d81c6cb74af7803c06",
"classarm__compute_1_1_c_l_depthwise_convolution_layer3x3.xhtml",
"classarm__compute_1_1_c_l_g_e_m_m_matrix_vector_multiply_kernel.xhtml#a0921fc687913f0f60f8bce6e64acc193",
"classarm__compute_1_1_c_l_locally_connected_matrix_multiply_kernel.xhtml#a46326372b64fcdec57bf29e44270a6ea",
"classarm__compute_1_1_c_l_scale.xhtml#aa5f9a9b25bd6f529c29939f6269d59f5",
"classarm__compute_1_1_coordinates.xhtml#a8f72922fd4f55b0b740811ea27bc7e0f",
"classarm__compute_1_1_g_c_pooling_layer.xhtml",
"classarm__compute_1_1_i_distribution1_d.xhtml#ae3664b841732a09d7749953ca5b81373",
"classarm__compute_1_1_memory.xhtml#a5e46e80e8811b4deb258cd55124e55a8",
"classarm__compute_1_1_n_e_convolution_rectangle_kernel.xhtml#a437bfa972bdb089215368c344cce05d3",
"classarm__compute_1_1_n_e_fully_connected_layer_reshape_weights.xhtml#af20812ea72913d388a9a70f9d28e7c36",
"classarm__compute_1_1_n_e_histogram_kernel.xhtml#a4983480fbaa765aea4052cf5c28bf60e",
"classarm__compute_1_1_n_e_reduction_operation.xhtml",
"classarm__compute_1_1_pixel_value.xhtml#a6c754430610a785d5cb27e3b0ac396b8",
"classarm__compute_1_1_tensor_info.xhtml#af501bc12f51874e786b0f28016a99ff7",
"classarm__compute_1_1graph__utils_1_1_p_p_m_accessor.xhtml#a0e5d8741226206d2dcdfcef62e7eed0b",
"classarm__compute_1_1test_1_1_i_array_accessor.xhtml#ad1d196a1f89d9889461e6920b92d4db5",
"classarm__compute_1_1test_1_1datasets_1_1_le_net5_fully_connected_layer_dataset.xhtml#a3e6b1264353f809d02913c6d10a5530f",
"classarm__compute_1_1test_1_1framework_1_1_option.xhtml#a48a2672f362eeed9a3e93403f4d3de37",
"classarm__compute_1_1test_1_1validation_1_1_bitwise_xor_validation_fixture.xhtml",
"convolution7x7_8cl.xhtml#a26babb0c719990ecbdf3abc6de920875",
"fixed__point_8h.xhtml#a36f754c05b6fddf6df0d8d0a74f8159f",
"gaussian__pyramid_8cl.xhtml",
"hwc_8hpp.xhtml#a85a433bf6d7a52ebd2553300d628aa39",
"neon__cnn_8cpp.xhtml#a7616847a3120a787be556c0bb30f43b4",
"reference_2_non_linear_filter_8cpp.xhtml#af00274a4d66faf0892f122870f1d9fd9",
"structarm__compute_1_1_c_l_old_value.xhtml#a92420f1bb5bb799a59e081a540e4c7ed",
"structarm__compute_1_1test_1_1framework_1_1dataset_1_1_range_dataset_1_1iterator.xhtml#a7b5000e45c1386c4e56ef650f6b0ef5b",
"tests_2framework_2_macros_8h.xhtml#a926fbae618fcbb44111e683b8a1e2f99",
"validation_2_c_l_2_convolution_layer_8cpp.xhtml#a9278ff5e5c38f35be584da6bf706ab23",
"validation_2_n_e_o_n_2_fully_connected_layer_8cpp.xhtml#a36e6031840dd2ee13e3942435811aeea"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';