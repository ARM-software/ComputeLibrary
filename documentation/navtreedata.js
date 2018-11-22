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
    [ "List of functions", "functions_list.xhtml", [
      [ "NEON functions", "functions_list.xhtml#S5_1", null ],
      [ "OpenCL functions", "functions_list.xhtml#S5_2", null ],
      [ "GLES Compute functions", "functions_list.xhtml#S5_3", null ]
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
    [ "Adding new operators", "add_operator.xhtml", [
      [ "Introduction", "add_operator.xhtml#S4_1_introduction", null ],
      [ "Supporting new operators", "add_operator.xhtml#S4_1_supporting_new_operators", [
        [ "Adding new data types", "add_operator.xhtml#S4_1_1_add_datatypes", null ],
        [ "Add a kernel", "add_operator.xhtml#S4_1_2_add_kernel", null ],
        [ "Add a function", "add_operator.xhtml#S4_1_3_add_function", null ],
        [ "Add validation artifacts", "add_operator.xhtml#S4_1_4_add_validation", [
          [ "Add the reference implementation and the tests", "add_operator.xhtml#S4_1_4_1_add_reference", null ],
          [ "Add dataset", "add_operator.xhtml#S4_1_4_2_add_dataset", null ],
          [ "Add a fixture and a data test case", "add_operator.xhtml#S4_1_4_3_add_fixture", null ]
        ] ]
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
"_c_l_2_batch_to_space_layer_8cpp.xhtml#acc070dfe82e8e6f33802197244d1c194",
"_c_l_2_l_s_t_m_layer_8cpp.xhtml#a15221160dbe648335c35231c529f65d7",
"_c_l_2_winograd_8cpp.xhtml#ade3cccfc740829a3b6ed570a3c4efeec",
"_c_l_copy_kernel_8cpp_source.xhtml",
"_c_l_gaussian5x5_8cpp_source.xhtml",
"_c_l_normalization_layer_kernel_8h_source.xhtml",
"_c_l_types_8h.xhtml#a3a440b3893fa10608d4428958be1c52e",
"_color_convert_helper_8h_source.xhtml",
"_fast_corners_8h.xhtml",
"_g_c_normalize_planar_y_u_v_layer_kernel_8cpp.xhtml",
"_helpers_8inl.xhtml#ab7b3af731907e85fcaf72555c446176b",
"_initializer_list_dataset_8h_source.xhtml",
"_n_e_bitwise_xor_8h_source.xhtml",
"_n_e_direct_convolution_layer_output_stage_kernel_8h.xhtml",
"_n_e_l_k_tracker_kernel_8h.xhtml",
"_n_e_o_n_2_deconvolution_layer_8cpp.xhtml#a3ba5002c3abe00c29ed121456da6bef8",
"_n_e_o_n_2_threshold_8cpp.xhtml",
"_n_e_softmax_layer_kernel_8cpp.xhtml#ab11431f1a64a618e5ed1d37634d0e0fe",
"_open_g_l_e_s_8cpp.xhtml#a0f290b0be95432b64e73ead723a947b7",
"_reorg_layer_8h_source.xhtml",
"_toolchain_support_8h.xhtml",
"_window_8inl_source.xhtml",
"arithmetic__op__quantized_8cl_source.xhtml",
"arm__compute_2graph_2_type_printer_8h.xhtml#abcd88ed51472e534decef274fb32bcaa",
"benchmark_2_c_l_2_floor_8cpp.xhtml#a4a14e383a632057e99845c74a72a6454",
"benchmark_2_g_l_e_s___c_o_m_p_u_t_e_2_scale_8cpp_source.xhtml",
"benchmark_2_n_e_o_n_2_pooling_layer_8cpp.xhtml#a3c3bb6166f756cef06811f326a7d9c46",
"classarm__compute_1_1_c_l_activation_layer_kernel.xhtml#a25c6b2015d9ec0c05a1f561458b445f0",
"classarm__compute_1_1_c_l_convert_fully_connected_weights_kernel.xhtml#afccf9b2ebcad0e6b3049b138fb6b800b",
"classarm__compute_1_1_c_l_fast_corners_kernel.xhtml#ac18eae467f4bbad88cbf69c5b835926b",
"classarm__compute_1_1_c_l_gaussian_pyramid_orb.xhtml#a9f1c2312374125fd95ee145a4f07515c",
"classarm__compute_1_1_c_l_logits1_d_norm_kernel.xhtml#a65a94de36a0b70e490bcdee287ff6c4d",
"classarm__compute_1_1_c_l_quantization_layer_kernel.xhtml#a33657445d6cf5905c0b74e566f769069",
"classarm__compute_1_1_c_l_symbols.xhtml#a3085f45d872eda6b1adf8557fd4915b9",
"classarm__compute_1_1_c_p_p_permute.xhtml#a93c836ab36443b23753d99495761daf7",
"classarm__compute_1_1_g_c_fully_connected_layer_reshape_weights.xhtml",
"classarm__compute_1_1_g_e_m_m_info.xhtml#a11d8f855e323a8396fe6944edcef4238",
"classarm__compute_1_1_i_g_c_kernel.xhtml#ad5ba9d34a3a855bf1dd2e36316ff550a",
"classarm__compute_1_1_i_tensor_info.xhtml#a9273842d8e5dc1a3c7fab727176fd7fe",
"classarm__compute_1_1_n_e_bitwise_xor_kernel.xhtml#a837b139cf977a6c4530e3d574fcceef2",
"classarm__compute_1_1_n_e_depthwise_weights_reshape_kernel.xhtml#a29a958bbfd9d88bebd984b833603c15c",
"classarm__compute_1_1_n_e_g_e_m_m_lowp_matrix_multiply_core.xhtml#a7d07d7fef064043cb810851831be5868",
"classarm__compute_1_1_n_e_integral_image_kernel.xhtml#a83a344e60eb7db895953a942abf16628",
"classarm__compute_1_1_n_e_quantization_layer_kernel.xhtml#a56c4b1f3df503c5f6308c318e305d7f8",
"classarm__compute_1_1_n_e_weights_reshape_kernel.xhtml#a7337b121d4f8ca7978ed617297e6397b",
"classarm__compute_1_1_semaphore.xhtml#aa3b21853f890838c88d047d6c2786917",
"classarm__compute_1_1gles_1_1_n_d_range.xhtml#a259cb5a711406a8c3e5d937eb9350cca",
"classarm__compute_1_1graph_1_1_i_node_visitor.xhtml",
"classarm__compute_1_1graph_1_1backends_1_1_c_l_tensor_handle.xhtml#a6e509c2a177b0b29e9e2369535094dee",
"classarm__compute_1_1graph__utils_1_1_random_accessor.xhtml#a220e3aa92e1b83c41e0df00505894a5a",
"classarm__compute_1_1test_1_1_lut_accessor.xhtml#ac8490a0e13403aa46250b736a3a9b1cc",
"classarm__compute_1_1test_1_1framework_1_1_profiler.xhtml#a8c528baf37154d347366083f0f816846",
"classarm__gemm_1_1_buffer.xhtml#a427705bf76071f12aa29992bd5cffa73",
"dir_4d03f28cfd35f8f734a3b0a2f1168d27.xhtml",
"gemm__fp32_8cpp.xhtml#a409b3529a2cd017beac811d67006a7da",
"helpers__asymm_8h.xhtml#a4cc3ff3a2eeb5f5e9d6743e08f632928",
"mul_8h.xhtml#a4f2e93cec76891a55fde0d4ead3f7728",
"optical__flow__pyramid__lk_8cl.xhtml#ae0b2360d4b8e961bf2709b0663fd9f2a",
"reference_2_table_lookup_8cpp_source.xhtml",
"store_8h.xhtml#ad417a154c619b3568b9eb97fc475a20c",
"structarm__compute_1_1graph_1_1_execution_task.xhtml",
"tensor__transform_8cpp.xhtml",
"tests_2validation_2_helpers_8cpp.xhtml#ad03c1c39d75226aecd5acd8e3959b02a",
"utils_2_utils_8h.xhtml#af214346f90d640ac468dd90fa2a275cc",
"validation_2_c_l_2_g_e_m_m_8cpp.xhtml#a8c12b8c19f4e7f45756d4a4dcb4c3156",
"validation_2_c_l_2_softmax_layer_8cpp.xhtml#a9c8bb6887ae1f3c848e04ec4c71324ff",
"validation_2_n_e_o_n_2_depthwise_convolution_layer_8cpp.xhtml#a77380f19fffc5978baf99cf01c239736",
"validation_2_n_e_o_n_2_softmax_layer_8cpp.xhtml#aa2cd1d5c891efd016370b418b5cd0c4c",
"warp__helpers__quantized_8h.xhtml#ac05d99f194a7cf429a7ecbadd1ffb018"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';