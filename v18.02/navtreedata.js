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
"arm__compute_2core_2_types_8h.xhtml#adf2ced65e536375a1c96425d9fced858",
"benchmark_2_c_l_2_l2_normalize_layer_8cpp.xhtml",
"benchmark_2_n_e_o_n_2_g_e_m_m_8cpp.xhtml#ad4c3a97cdb229530c472949b2cca3ad2",
"classarm__compute_1_1_activation_layer_info.xhtml#ac06d9557031b0febb70869efd793cb38",
"classarm__compute_1_1_c_l_copy_to_array_kernel.xhtml#a7fa1d109f0c6d72521e336fd3955cee9",
"classarm__compute_1_1_c_l_g_e_m_m_lowp_matrix_multiply_kernel.xhtml#a493987e85723a8000eb26d1f00e2ad0e",
"classarm__compute_1_1_c_l_kernel_library.xhtml#a389a543ae040ca6bf6b57fd6215f154b",
"classarm__compute_1_1_c_l_pooling_layer_kernel.xhtml#a493987e85723a8000eb26d1f00e2ad0e",
"classarm__compute_1_1_c_l_threshold.xhtml",
"classarm__compute_1_1_g_c_g_e_m_m.xhtml#a8184f9bf2e8f4fdc16cfe7812e229d95",
"classarm__compute_1_1_h_o_g_info.xhtml#a6a21c3fbc2803bbf3d975bec8977307f",
"classarm__compute_1_1_i_memory_manager.xhtml#a3a47f406dc0d1b046ad26c3097f15293",
"classarm__compute_1_1_n_e_accumulate_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_depth_convert_layer.xhtml#a82e5c73f95671393c6e54da2e1496c4b",
"classarm__compute_1_1_n_e_g_e_m_m_interleave4x4_kernel.xhtml#a83a344e60eb7db895953a942abf16628",
"classarm__compute_1_1_n_e_harris_score_kernel.xhtml#ae3278fce9a66ca7603efcb367e6b3816",
"classarm__compute_1_1_n_e_pooling_layer_kernel.xhtml#a423f9a45a52983b4de5e2b347f4369c7",
"classarm__compute_1_1_n_e_winograd_layer_transform_input_kernel.xhtml#a4370ae5fda7bd455a171fc8ed4d3f283",
"classarm__compute_1_1_sub_tensor_info.xhtml#a907f837b924945ad1981c8fe8eca61e4",
"classarm__compute_1_1graph_1_1_i_tensor_object.xhtml#aeef25dd1e341012fd05197bea6ab4e41",
"classarm__compute_1_1test_1_1_c_l_accessor.xhtml#a0189be0971a32a0171c51b90a1ab1020",
"classarm__compute_1_1test_1_1benchmark_1_1_depth_concatenate_layer_fixture.xhtml#a13a43e6d814de94978c515cb084873b1",
"classarm__compute_1_1test_1_1datasets_1_1_pooling_layer_dataset.xhtml#a8069c6eabcf150084f21353256a829f4",
"classarm__compute_1_1test_1_1framework_1_1_instruments_stats.xhtml#a5378e6e486090c47ac55da43a654e93d",
"classarm__compute_1_1test_1_1networks_1_1_mobile_net_v1_network.xhtml#a3b778cda9ac3fad08e7217edbcb942e0",
"classarm__compute_1_1test_1_1validation_1_1_reshape_layer_validation_fixture.xhtml",
"dir_b30607d43e07fadf88f4c95a88a6c88e.xhtml",
"fixed__point_8h.xhtml#addf730860de5f7752f0c85a385088e96",
"graph__vgg19_8cpp_source.xhtml",
"intrinsics_8h_source.xhtml",
"non__linear__filter5x5_8cl.xhtml#a41b21a372a22ccea4fc7794e6de9b0cc",
"reference_2_remap_8cpp.xhtml#ae76991329781f552c3a2d66a9dd5af79",
"structarm__compute_1_1_key_point.xhtml#af6d3062751bd565decb1a2cd3b63bdb2",
"structarm__compute_1_1test_1_1framework_1_1dataset_1_1_initializer_list_dataset_1_1iterator.xhtml#af85de6015a44d7ff25198be6265ea56b",
"tests_2_types_8h.xhtml#afa20b6a7f4383003babd690f026f22dca241dd841abade20fcb27b8a9f494e1eb",
"unionmali__userspace_1_1uk__header.xhtml",
"validation_2_c_l_2_pooling_layer_8cpp.xhtml#a131007f56dc7d0a83672ce1dadf44713",
"validation_2_n_e_o_n_2_floor_8cpp.xhtml#a6cc35d5c6f16df2aa67e16246b4ee0e3",
"validation_2reference_2_depthwise_separable_convolution_layer_8cpp.xhtml#a0c7524d5bce923f96be77f49e1da3913"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';