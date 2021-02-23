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
          [ "How to manually build the examples ?", "index.xhtml#S3_2_2_examples", null ],
          [ "Build for SVE or SVE2", "index.xhtml#S3_2_3_sve", null ]
        ] ],
        [ "Building for Android", "index.xhtml#S3_3_android", [
          [ "How to build the library ?", "index.xhtml#S3_3_1_library", null ],
          [ "How to manually build the examples ?", "index.xhtml#S3_3_2_examples", null ]
        ] ],
        [ "Building for macOS", "index.xhtml#S3_4_macos", null ],
        [ "Building for bare metal", "index.xhtml#S3_5_bare_metal", [
          [ "How to build the library ?", "index.xhtml#S3_5_1_library", null ],
          [ "How to manually build the examples ?", "index.xhtml#S3_5_2_examples", null ]
        ] ],
        [ "Building on a Windows host system", "index.xhtml#S3_6_windows_host", [
          [ "Bash on Ubuntu on Windows", "index.xhtml#S3_6_1_ubuntu_on_windows", null ],
          [ "Cygwin", "index.xhtml#S3_6_2_cygwin", null ]
        ] ],
        [ "OpenCL DDK Requirements", "index.xhtml#S3_7_cl_requirements", [
          [ "Hard Requirements", "index.xhtml#S3_7_1_cl_hard_requirements", null ],
          [ "Performance improvements", "index.xhtml#S3_7_2_cl_performance_requirements", null ]
        ] ],
        [ "OpenCL Tuner", "index.xhtml#S3_8_cl_tuner", [
          [ "How to use it", "index.xhtml#S3_8_1_cl_tuner_how_to", null ]
        ] ]
      ] ]
    ] ],
    [ "Library architecture", "architecture.xhtml", [
      [ "Core vs Runtime libraries", "architecture.xhtml#S4_1_1", null ],
      [ "Data-type and Data-layout support", "architecture.xhtml#S4_1_2", null ],
      [ "Fast-math support", "architecture.xhtml#S4_1_3", null ],
      [ "Thread-safety", "architecture.xhtml#S4_1_4", null ],
      [ "Windows, kernels, multi-threading and functions", "architecture.xhtml#S4_2_windows_kernels_mt_functions", [
        [ "Windows", "architecture.xhtml#S4_2_1_windows", null ],
        [ "Kernels", "architecture.xhtml#S4_2_2", null ],
        [ "Multi-threading", "architecture.xhtml#S4_2_3", null ],
        [ "Functions", "architecture.xhtml#S4_2_4", null ],
        [ "OpenCL Scheduler and kernel library", "architecture.xhtml#S4_4_1_cl_scheduler", null ],
        [ "OpenCL events and synchronization", "architecture.xhtml#S4_4_2_events_sync", null ],
        [ "OpenCL / Neon interoperability", "architecture.xhtml#S4_4_2_cl_neon", null ]
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
      [ "Import Memory Interface", "architecture.xhtml#S4_8_import_memory", null ],
      [ "OpenCL Tuner", "architecture.xhtml#S4_9_opencl_tuner", null ],
      [ "Weights Manager", "architecture.xhtml#S4_10_weights_manager", [
        [ "Working with the Weights Manager", "architecture.xhtml#S4_10_1_working_with_weights_manager", null ]
      ] ],
      [ "Experimental Features", "architecture.xhtml#S5_0_experimental", [
        [ "Run-time Context", "architecture.xhtml#S5_1_run_time_context", null ]
      ] ]
    ] ],
    [ "Validation and benchmarks tests", "tests.xhtml", [
      [ "Overview", "tests.xhtml#tests_overview", [
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
      ] ],
      [ "Extract data from pre-trained frozen tensorflow model", "data_import.xhtml#tf_frozen_model_extractor", [
        [ "How to use the script", "data_import.xhtml#tensorflow_frozen_how_to", null ],
        [ "What is the expected output from the script", "data_import.xhtml#tensorflow_frozen_result", null ]
      ] ],
      [ "Validating examples", "data_import.xhtml#validate_examples", null ]
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
    [ "Contribution guidelines", "contribution_guidelines.xhtml", [
      [ "Coding standards and guidelines", "contribution_guidelines.xhtml#S5_1_coding_standards", [
        [ "Rules", "contribution_guidelines.xhtml#S5_1_1_rules", null ],
        [ "How to check the rules", "contribution_guidelines.xhtml#S5_1_2_how_to_check_the_rules", null ],
        [ "Library size: best practices and guidelines", "contribution_guidelines.xhtml#S5_1_3_library_size_guidelines", [
          [ "Template suggestions", "contribution_guidelines.xhtml#S5_1_3_1_template_suggestions", null ]
        ] ],
        [ "Secure coding practices", "contribution_guidelines.xhtml#S5_1_4_secure_coding_practices", [
          [ "General Coding Practices", "contribution_guidelines.xhtml#S5_1_4_1_general_coding_practices", null ],
          [ "Secure Coding Best Practices", "contribution_guidelines.xhtml#S5_1_4_2_secure_coding_best_practices", null ]
        ] ],
        [ "Guidelines for stable API/ABI", "contribution_guidelines.xhtml#S5_1_5_guidelines_for_stable_api_abi", [
          [ "Guidelines for API", "contribution_guidelines.xhtml#S5_1_5_1_guidelines_for_api", null ],
          [ "Guidelines for ABI", "contribution_guidelines.xhtml#S5_1_5_2_guidelines_for_abi", null ],
          [ "API deprecation process", "contribution_guidelines.xhtml#S5_1_5_3_api_deprecation_process", null ]
        ] ]
      ] ],
      [ "How to submit a patch", "contribution_guidelines.xhtml#S5_2_how_to_submit_a_patch", null ],
      [ "Patch acceptance and code review", "contribution_guidelines.xhtml#S5_3_code_review", null ]
    ] ],
    [ "List of functions", "functions_list.xhtml", [
      [ "Neon functions", "functions_list.xhtml#S6_1", null ],
      [ "OpenCL functions", "functions_list.xhtml#S6_2", null ],
      [ "GLES Compute functions", "functions_list.xhtml#S6_3", null ],
      [ "CPP functions", "functions_list.xhtml#S6_4", null ]
    ] ],
    [ "Errata", "errata.xhtml", [
      [ "Errata", "errata.xhtml#S7_1_errata", null ]
    ] ],
    [ "Gemm Tuner", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml", null ],
    [ "Deprecated List", "deprecated.xhtml", null ],
    [ "Namespaces", null, [
      [ "Namespace List", "namespaces.xhtml", "namespaces" ],
      [ "Namespace Members", "namespacemembers.xhtml", [
        [ "All", "namespacemembers.xhtml", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.xhtml", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.xhtml", "namespacemembers_vars" ],
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
"_abs_layer_8cpp.xhtml",
"_c_l_2_arg_min_max_8cpp.xhtml#a47d01269d8515680e7a30ec8ab19bc31",
"_c_l_2_cast_8cpp.xhtml#ad268d70596992c77669e6bf32f8bacc3",
"_c_l_2_depth_convert_layer_8cpp.xhtml#a656df9dd7c5ab11fed1ea622b1d30880",
"_c_l_2_flatten_8cpp.xhtml#aa2c7ada491af83e6b0263e057cbfd8d0",
"_c_l_2_l_s_t_m_layer_quantized_8cpp.xhtml",
"_c_l_2_pooling_layer_8cpp.xhtml#a9a01fbd21389371299bff64956fb24e2",
"_c_l_2_stack_layer_8cpp.xhtml#a457dd390a99b5422ad608759be6f2c4e",
"_c_l_array_accessor_8h_source.xhtml",
"_c_l_direct_convolution_layer_8h.xhtml",
"_c_l_g_e_m_m_reshape_r_h_s_matrix_kernel_8cpp.xhtml",
"_c_l_non_linear_filter_kernel_8cpp.xhtml",
"_c_l_split_8h_source.xhtml",
"_c_p_p_non_maximum_suppression_8h.xhtml",
"_common_8h.xhtml#abf73f846abd6dfb414f6b51f8286ba09a1149b156e352294778e3596869f2f799",
"_cpu_sub_kernel_8cpp.xhtml#aead5dda1dac70c94402452aacee5d1a8",
"_eltwise_layer_node_8cpp_source.xhtml",
"_g_c_batch_normalization_layer_kernel_8cpp.xhtml",
"_g_e_m_m_lowp_8h.xhtml#afc20df3bafb985ea3722b47b48dbb135",
"_g_p_u_target_8h.xhtml#a735ac6c2a02e320969625308810444f3aa78cc0fd1cab24af0fad71dc4c256f8e",
"_i_cpu_kernel_8h.xhtml#a5e357b8c8241b9204a9d5e502565c20d",
"_l2_normalize_layer_node_8cpp_source.xhtml",
"_n_e_asymm_8inl.xhtml",
"_n_e_dequantization_layer_8h_source.xhtml",
"_n_e_g_e_m_m_interleave4x4_kernel_8cpp.xhtml",
"_n_e_max_unpooling_layer_kernel_8cpp.xhtml",
"_n_e_o_n_2_cast_8cpp.xhtml#a77863b95e33e3562cbeb409ec5cdfeb5",
"_n_e_o_n_2_depth_convert_layer_8cpp.xhtml#a3ed17ec92926b05f647ae5dc75f18397",
"_n_e_o_n_2_fuse_batch_normalization_8cpp.xhtml#a512eb649fdb115f2dee5df9f1d156b16",
"_n_e_o_n_2_magnitude_8cpp.xhtml#a3ed28dd356e85eef81a7d7128b8e7390",
"_n_e_o_n_2_reduction_operation_8cpp.xhtml#ac2de55a9ccd7c4cd070892f20faa6aec",
"_n_e_o_n_2kernels_2scale_2impl_2_s_v_e_2integer_8cpp.xhtml",
"_n_e_sub_tensor_handle_8cpp_source.xhtml",
"_open_g_l_e_s_8cpp.xhtml#a334ce7125cfb54a811b748f23f228417",
"_range_dataset_8h.xhtml",
"_sobel_8h.xhtml",
"_validation_8h.xhtml",
"a64__s8q__nhwc__avg__generic__depthfirst_8hpp.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a15a05537a472ee742404821851529327a8d6b5cada83510220f59e00ce86d4d92",
"arm__compute_2core_2_validate_8h.xhtml#a13722b17f287d58f2a24f039dc2b4fc1",
"arm__compute_2runtime_2_c_l_2_i_c_l_operator_8h_source.xhtml",
"benchmark__examples_2_run_example_8cpp_source.xhtml",
"clang__tidy__rules_8py_source.xhtml",
"classarm__compute_1_1_c_l_arithmetic_division.xhtml#aac5c48f59131c5da160b59e875e5705f",
"classarm__compute_1_1_c_l_compile_context.xhtml#a1d4cf09994ef5a094eed69cd37ca26ba",
"classarm__compute_1_1_c_l_depthwise_convolution_layer3x3_n_c_h_w_kernel.xhtml#a2914f42b72a38522e963d64bb84eab5d",
"classarm__compute_1_1_c_l_f_f_t_scale_kernel.xhtml#a2848da7e825ae54ef2258bb76938a39b",
"classarm__compute_1_1_c_l_g_e_m_m_matrix_multiply_kernel.xhtml#a170f236fd8751c4e1675873b496f7cf8",
"classarm__compute_1_1_c_l_histogram.xhtml#acb6c95349bcbf0a7f25e7e7e1e8f4b0e",
"classarm__compute_1_1_c_l_lut_allocator.xhtml#ac7e247b94ceae35db2a172961ab83df9",
"classarm__compute_1_1_c_l_pixel_wise_multiplication.xhtml#a99b95a175203c56994a1bbe8cc1b228f",
"classarm__compute_1_1_c_l_scale_kernel.xhtml#a896bee994fc25c819f7faac1578182cf",
"classarm__compute_1_1_c_l_sub_tensor.xhtml#a3f3d38e602db0a6af8ac5e932d798617",
"classarm__compute_1_1_c_p_p_corner_candidates_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_g_c_buffer_memory_region.xhtml#a567fad71aedc47306dbf7b0c020257d8",
"classarm__compute_1_1_g_c_normalize_planar_y_u_v_layer_kernel.xhtml#a954898cff15f9ad68595daa507f8742c",
"classarm__compute_1_1_i_c_l_g_e_m_m_lowp_reduction_kernel.xhtml#a171dc54d0a3009a00ad971feaf705c78",
"classarm__compute_1_1_i_memory.xhtml#a5f1239a904c50f8013ba163960605bce",
"classarm__compute_1_1_iterator.xhtml",
"classarm__compute_1_1_n_e_batch_normalization_layer.xhtml#a8f98f383f8998195408b570534483536",
"classarm__compute_1_1_n_e_convert_quantized_signedness_kernel.xhtml#acf2aa85acb943c3ee50b565fff090488",
"classarm__compute_1_1_n_e_direct_convolution_layer.xhtml#afd02ea33ac8b34975a45ca672a939967",
"classarm__compute_1_1_n_e_fill_border_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_g_e_m_m_matrix_multiply_kernel.xhtml",
"classarm__compute_1_1_n_e_instance_normalization_layer.xhtml",
"classarm__compute_1_1_n_e_min_max_location.xhtml#ab996eedb0d719f5faa091726e69e73c3",
"classarm__compute_1_1_n_e_reduction_operation.xhtml#a86a20d6d15876935dd8999ef0c26d216",
"classarm__compute_1_1_n_e_space_to_depth_layer_kernel.xhtml#aa1d37a46a3d36d6ffaa41bf8c0244883",
"classarm__compute_1_1_pixel_value.xhtml#a951fd574457b3c1d973bab27852497bf",
"classarm__compute_1_1_tensor_allocator.xhtml#a943854ce6f3aafac2714340e0915de75",
"classarm__compute_1_1cpu_1_1_cpu_softmax_generic.xhtml#a684a54d1fb1634a348a585c6b5e76df0",
"classarm__compute_1_1graph_1_1_channel_shuffle_layer_node.xhtml#a65d13dc93e2df5e8ab725263cf9f4ac5",
"classarm__compute_1_1graph_1_1_i_node.xhtml",
"classarm__compute_1_1graph_1_1_split_layer_node.xhtml#a3f18a7449b9d7fc9e5fec212b8e61710",
"classarm__compute_1_1graph_1_1frontend_1_1_convolution_layer.xhtml#af896fc13b0eaaafe855f0fbcd65a07de",
"classarm__compute_1_1logging_1_1_printer.xhtml#aaa62e7eb81a5dcf350ab8b677ddd8f60",
"classarm__compute_1_1test_1_1_accessor.xhtml#ae15e08764c8bb788c6b275244e0f3205",
"classarm__compute_1_1test_1_1_simple_tensor_accessor.xhtml#a39537b09ccc3ce3d17922f4ef49a123f",
"classarm__compute_1_1test_1_1framework_1_1_wall_clock.xhtml#a5d5d9ebd12cd3a7d268bc98a3fc7bf97",
"classarm__compute_1_1utils_1_1mmap__io_1_1_m_mapped_file.xhtml#a3616a9b4f59739152c904d57117eb3a5",
"classarm__gemm_1_1_std_transforms_fixed.xhtml#acffac2ffe7ded8d0e1f14387ece45d1d",
"core_2_c_l_2_c_l_types_8h_source.xhtml",
"dir_3aa2f38a5cb88602a607f0fea1df9588.xhtml",
"elementwise__operation__quantized_8cl.xhtml",
"gemm__helpers_8h.xhtml#a02cb70709fbb9650a1a639c7abe638fa",
"gemm__helpers_8h.xhtml#af2a551fa9fc7f5f9e53063170ca579b0",
"graph__inception__resnet__v1_8cpp.xhtml#ae89536b490f3ad0e9070bd2dd0a6df5d",
"include__functions__kernels_8py.xhtml#a087978f32d7bd9f7ed6d899c80812d88",
"load__store__utility_8h.xhtml#a82259edb818cf0cf2bb6163f5fbeaa24",
"namespacearm__compute_1_1test_1_1convolution__3d.xhtml",
"neon__gemm__qasymm8_8cpp.xhtml#a58ee4b8dd599d0ff978435d7120812ce",
"quantized_8h.xhtml#af149abce9fb73b4e642934e2a9342a66",
"reference_2_fuse_batch_normalization_8cpp.xhtml#aeca7e17be27219cf39a476d9e63f45f1",
"reference_2_range_8cpp.xhtml#ad7059348bb7b955422375b8759747abb",
"runtime_2_c_l_2_c_l_types_8h.xhtml#a6f7d2fe6b41e36d408375ad6bf9a612aa98032d1fe61a8745281dbd3269082825",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a5d4fd32c9d4c17d73119ce48b68b1e2a",
"src_2core_2_utils_8cpp.xhtml#a3e4a3ad379526db61e1ebf469b455f83",
"structarm__compute_1_1_c_l_device_options.xhtml#ae19afa6eae26622a0068daa092b25f6a",
"structarm__compute_1_1_valid_region.xhtml#a1fcd64682b37ed3c2098d0094ce788d8",
"structarm__compute_1_1mlgo_1_1parser_1_1_token.xhtml#a6e2d2148929ac137f2e2ef4091a4dc69",
"structarm__conv_1_1pooling_1_1_pooling_implementation.xhtml#a9bc0beccc29f52877b72a95e4f0d4f1c",
"tensor__info_8h.xhtml#a70a4fcb965a670f56fd8cec4be60caad",
"tests_2validation_2_g_l_e_s___c_o_m_p_u_t_e_2_fully_connected_layer_8cpp.xhtml#a50a80428b618e7eec7f17e81df71ca84",
"utils_2_type_printer_8h.xhtml#a3cf43a216912b361eaffc3c71452a31e",
"warp__helpers_8h_source.xhtml"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';