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
"_c_l_2_convert_fully_connected_weights_8cpp.xhtml#a2707daabc31d77ead2c865f2e6ff5033",
"_c_l_2_permute_8cpp.xhtml#ab3f4ab4a5c5018ffd9d99cf7b369a61a",
"_c_l_convolution_layer_8h_source.xhtml",
"_c_l_pixel_wise_multiplication_8h.xhtml",
"_depth_convert_layer_8h.xhtml#a742cdbdf3c1db76c9a189f1bcd745579",
"_g_c_g_e_m_m_matrix_multiply_kernel_8h.xhtml",
"_i_array_8h_source.xhtml",
"_mali_counter_8cpp.xhtml#a736088bbd41fd7407ac9fc811864db68",
"_n_e_direct_convolution_detail_8h.xhtml#abd70ac1509d341ca206da7d421a0b7a9",
"_n_e_h_o_g_descriptor_kernel_8h_source.xhtml",
"_n_e_o_n_2_convolution_8cpp.xhtml#a97f70c1390d9c076f79253311bc1c929",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml#a2e7e44aefaae0abe0312a137133917c4",
"_open_c_l_timer_8cpp.xhtml",
"_toolchain_support_8h.xhtml#a3c9ee94e1de3032f244cf041310868b4",
"and_8h.xhtml#a1f5ad185828729587ebd0356cd0ca275",
"arm__compute_2core_2_types_8h.xhtml#afdda916edc7502967bbec17ea3c06c02a290d4b81f4e2b47d86fd1b0170e9aab7",
"benchmark_2_c_l_2_equalize_histogram_8cpp_source.xhtml",
"benchmark_2_g_l_e_s___c_o_m_p_u_t_e_2_softmax_layer_8cpp_source.xhtml",
"channel__shuffle_8cl_source.xhtml",
"classarm__compute_1_1_c_l_build_options.xhtml#a58abf0792821f6f1d6271570aa27dd84",
"classarm__compute_1_1_c_l_derivative_kernel.xhtml#ac3a2c95536dfbe4ea71e3c1441cd2960",
"classarm__compute_1_1_c_l_gaussian_pyramid.xhtml#a8c57d617c3dd6f7c04c77231dc5928f0",
"classarm__compute_1_1_c_l_logits1_d_norm_kernel.xhtml#a19373a3162839643484511cec353d34b",
"classarm__compute_1_1_c_l_reshape_layer_kernel.xhtml#a074e10cfb217e657b9e81adeca2abc68",
"classarm__compute_1_1_c_l_winograd_convolution_layer.xhtml",
"classarm__compute_1_1_g_c_direct_convolution_layer_kernel.xhtml",
"classarm__compute_1_1_g_c_tensor_shift_kernel.xhtml#a8fd12b95bdde3f93db96bc9b1598db69",
"classarm__compute_1_1_i_g_c_kernel.xhtml#a6f56c8bb44166634652b877d6eb7a9a2",
"classarm__compute_1_1_kernel.xhtml#a83af2718294984c1053223a928164a7b",
"classarm__compute_1_1_n_e_channel_combine_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_direct_convolution_layer_output_stage_kernel.xhtml#ab02c33a0e6c13a21c8b8ceed03fd8b6d",
"classarm__compute_1_1_n_e_gaussian5x5_hor_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_min_max_kernel.xhtml#a14c484d0c1759b7163e3691ea1bb8bb5",
"classarm__compute_1_1_n_e_sobel5x5_hor_kernel.xhtml#ab3fe9be760d9e2faa8cb36872889fc98",
"classarm__compute_1_1_program.xhtml#a1615f70129362da8c0c0cd82ef6c1b44",
"classarm__compute_1_1_window.xhtml#a5e8e5ab06329702df09ab7eb07e9b84d",
"classarm__compute_1_1graph_1_1_i_node_visitor.xhtml#ad1265b49ac3e299a4025cd29987476d3",
"classarm__compute_1_1graph_1_1frontend_1_1_activation_layer.xhtml#ae538c5265a9f26cac7d41eeca26a60a0",
"classarm__compute_1_1test_1_1_c_l_accessor.xhtml#abdd3637f2bbde9d7d0cc0b7bbd8400bb",
"classarm__compute_1_1test_1_1framework_1_1_framework.xhtml#a47fbbce86aea0ba4a9d539c069f840c9",
"classarm__compute_1_1test_1_1framework_1_1dataset_1_1_singleton_dataset.xhtml#a18b86dafc894ede9c2b71030a731a106",
"dir_445738e6fed662d0b65e690e3f1e1de0.xhtml",
"fixed__point_8h.xhtml#abf77757e07132c678f7146c860e80fe1",
"graph_8h_source.xhtml",
"max_8h.xhtml#a5305cd57eda646455fa9e5638d828db9",
"normalization__layer_8cl.xhtml#a7db22b7ddf8f433e02ede3bd9c56d9d9",
"reference_2_reduction_operation_8cpp_source.xhtml",
"structarm__compute_1_1_i_o_format_info.xhtml#a36c28b28da4e04d698d6b598fb1eaca6a90589c47f06eb971d548591f23c285af",
"structarm__compute_1_1test_1_1framework_1_1dataset_1_1_cartesian_product_dataset_1_1iterator.xhtml#af1b1c7856a59f34c7d3570f946a2ff00",
"tests_2validation_2_fixed_point_8h.xhtml#add6426cbf2e057a195846d4ba09a50bea02ff1fff1812f84c89547fcd6c176150",
"validation_2_c_l_2_depthwise_convolution_layer_8cpp.xhtml#af5ad79218ce618faeba270aa32dd6583",
"validation_2_c_l_2_scale_8cpp.xhtml#ae72c3e42124a897611899895e37d9450",
"validation_2_n_e_o_n_2_g_e_m_m_8cpp.xhtml#ac2c6b9d9c2494e45cb83313f09519dc4",
"validation_2reference_2_g_e_m_m_8cpp.xhtml#acc8055ed1ae62ec87a4b389047c1464a"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';