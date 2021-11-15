var NAVTREE =
[
  [ "Compute Library", "index.xhtml", [
    [ "Introduction", "index.xhtml", null ],
    [ "Introduction", "introduction.xhtml", [
      [ "Contact / Support", "introduction.xhtml#S0_1_contact", null ],
      [ "Pre-built binaries", "introduction.xhtml#S0_2_prebuilt_binaries", null ],
      [ "File organisation", "introduction.xhtml#S0_3_file_organisation", null ]
    ] ],
    [ "How to Build and Run Examples", "how_to_build.xhtml", [
      [ "Build options", "how_to_build.xhtml#S1_1_build_options", null ],
      [ "Building for Linux", "how_to_build.xhtml#S1_2_linux", [
        [ "How to build the library ?", "how_to_build.xhtml#S1_2_1_library", null ],
        [ "How to manually build the examples ?", "how_to_build.xhtml#S1_2_2_examples", null ],
        [ "Build for SVE or SVE2", "how_to_build.xhtml#S1_2_3_sve", null ]
      ] ],
      [ "Building for Android", "how_to_build.xhtml#S1_3_android", [
        [ "How to build the library ?", "how_to_build.xhtml#S1_3_1_library", null ],
        [ "How to manually build the examples ?", "how_to_build.xhtml#S1_3_2_examples", null ]
      ] ],
      [ "Building for macOS", "how_to_build.xhtml#S1_4_macos", null ],
      [ "Building for bare metal", "how_to_build.xhtml#S1_5_bare_metal", [
        [ "How to build the library ?", "how_to_build.xhtml#S1_5_1_library", null ],
        [ "How to manually build the examples ?", "how_to_build.xhtml#S1_5_2_examples", null ]
      ] ],
      [ "Building on a Windows host system", "how_to_build.xhtml#S1_6_windows_host", [
        [ "Bash on Ubuntu on Windows", "how_to_build.xhtml#S1_6_1_ubuntu_on_windows", null ],
        [ "Cygwin", "how_to_build.xhtml#S1_6_2_cygwin", null ]
      ] ],
      [ "OpenCL DDK Requirements", "how_to_build.xhtml#S1_7_cl_requirements", [
        [ "Hard Requirements", "how_to_build.xhtml#S1_7_1_cl_hard_requirements", null ],
        [ "Performance improvements", "how_to_build.xhtml#S1_7_2_cl_performance_requirements", null ]
      ] ]
    ] ],
    [ "Library Architecture", "architecture.xhtml", [
      [ "Core vs Runtime libraries", "architecture.xhtml#architecture_core_vs_runtime", null ],
      [ "Fast-math support", "architecture.xhtml#architecture_fast_math", null ],
      [ "Thread-safety", "architecture.xhtml#architecture_thread_safety", null ],
      [ "Algorithms", "architecture.xhtml#architecture__algorithms", null ],
      [ "Images, padding, border modes and tensors", "architecture.xhtml#architecture_images_tensors", [
        [ "Padding and border modes", "architecture.xhtml#architecture_images_tensors_padding_and_border", [
          [ "Padding", "architecture.xhtml#architecture_images_tensors_padding", null ],
          [ "Valid regions", "architecture.xhtml#architecture_images_tensors_valid_region", null ]
        ] ],
        [ "Tensors", "architecture.xhtml#architecture_images_tensors_tensors", null ],
        [ "Images and Tensors description conventions", "architecture.xhtml#architecture_images_tensors_description_conventions", null ],
        [ "Working with Images and Tensors using iterators", "architecture.xhtml#architecture_images_tensors_working_with_objects", null ],
        [ "Sub-tensors", "architecture.xhtml#architecture_images_tensors_sub_tensors", null ]
      ] ],
      [ "MemoryManager", "architecture.xhtml#architecture_memory_manager", [
        [ "MemoryGroup, MemoryPool and MemoryManager Components", "architecture.xhtml#architecture_memory_manager_component", [
          [ "MemoryGroup", "architecture.xhtml#architecture_memory_manager_component_memory_group", null ],
          [ "MemoryPool", "architecture.xhtml#architecture_memory_manager_component_memory_pool", null ],
          [ "MemoryManager Components", "architecture.xhtml#architecture_memory_manager_component_memory_manager_components", null ]
        ] ],
        [ "Working with the Memory Manager", "architecture.xhtml#architecture_memory_manager_working_with_memory_manager", null ],
        [ "Function support", "architecture.xhtml#architecture_memory_manager_function_support", null ]
      ] ],
      [ "Import Memory Interface", "architecture.xhtml#architecture_import_memory", null ],
      [ "OpenCL Tuner", "architecture.xhtml#architecture_opencl_tuner", null ],
      [ "OpenCL Queue Priorities", "architecture.xhtml#architecture_cl_queue_priorities", null ],
      [ "Weights Manager", "architecture.xhtml#architecture_weights_manager", [
        [ "Working with the Weights Manager", "architecture.xhtml#architecture_weights_manager_working_with_weights_manager", null ]
      ] ],
      [ "Programming Model", "architecture.xhtml#programming_model", [
        [ "Functions", "architecture.xhtml#programming_model_functions", null ],
        [ "OpenCL Scheduler", "architecture.xhtml#programming_model_scheduler", null ],
        [ "OpenCL events and synchronization", "architecture.xhtml#programming_model__events_sync", null ],
        [ "OpenCL / Arm® Neon™ interoperability", "architecture.xhtml#programming_model_cl_neon", null ]
      ] ],
      [ "Experimental Features", "architecture.xhtml#architecture_experimental", [
        [ "Run-time Context", "architecture.xhtml#architecture_experimental_run_time_context", null ],
        [ "CLVK", "architecture.xhtml#architecture_experimental_clvk", null ]
      ] ],
      [ "Experimental Application Programming Interface", "architecture.xhtml#architecture_experimental_api", [
        [ "Overview", "architecture.xhtml#architecture_experimental_api_overview", null ],
        [ "Fundamental objects", "architecture.xhtml#architecture_experimental_api_objects", [
          [ "AclContext or Context", "architecture.xhtml#architecture_experimental_api_objects_context", [
            [ "AclTarget", "architecture.xhtml#architecture_experimental_api_object_context_target", null ],
            [ "AclExecutionMode", "architecture.xhtml#architecture_experimental_api_object_context_execution_mode", null ],
            [ "AclTargetCapabilities", "architecture.xhtml#architecture_experimental_api_object_context_capabilities", null ],
            [ "Allocator", "architecture.xhtml#architecture_experimental_api_object_context_allocator", null ]
          ] ],
          [ "AclTensor or Tensor", "architecture.xhtml#architecture_experimental_api_objects_tensor", null ],
          [ "AclQueue or Queue", "architecture.xhtml#architecture_experimental_api_objects_queue", null ]
        ] ],
        [ "Internal", "architecture.xhtml#architecture_experimental_api_internal", [
          [ "Operators vs Kernels", "architecture.xhtml#architecture_experimental_api_internal_operator_vs_kernels", null ]
        ] ],
        [ "Build fat binary", "architecture.xhtml#architecture_experimental_build_fat_binary", null ],
        [ "Per-operator build", "architecture.xhtml#architecture_experimental_per_operator_build", null ],
        [ "Build high priority operators", "architecture.xhtml#architecture_experimental_build_high_priority_operators", null ]
      ] ]
    ] ],
    [ "Data Type Support", "data_type_support.xhtml", [
      [ "Supported Data Types", "data_type_support.xhtml#data_type_support_supported_data_type", null ]
    ] ],
    [ "Data Layout Support", "data_layout_support.xhtml", [
      [ "Supported Data Layouts", "data_layout_support.xhtml#data_layout_support_supported_data_layout", null ]
    ] ],
    [ "Supported Operators", "operators_list.xhtml", [
      [ "Supported Operators", "operators_list.xhtml#S9_1_operators_list", null ]
    ] ],
    [ "Validation and Benchmarks", "tests.xhtml", [
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
    [ "Advanced", "advanced.xhtml", [
      [ "OpenCL Tuner", "advanced.xhtml#S1_8_cl_tuner", [
        [ "How to use it", "advanced.xhtml#S1_8_1_cl_tuner_how_to", null ]
      ] ]
    ] ],
    [ "Release Versions and Changelog", "versions_changelogs.xhtml", [
      [ "Release versions", "versions_changelogs.xhtml#S2_1_versions", null ],
      [ "Changelog", "versions_changelogs.xhtml#S2_2_changelog", null ]
    ] ],
    [ "Errata", "errata.xhtml", [
      [ "Errata", "errata.xhtml#S7_1_errata", null ]
    ] ],
    [ "Contribution Guidelines", "contribution_guidelines.xhtml", [
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
    [ "How to Add a New Operator", "adding_operator.xhtml", [
      [ "Adding new operators", "adding_operator.xhtml#S4_0_introduction", null ],
      [ "Introduction", "adding_operator.xhtml#S4_1_introduction", null ],
      [ "Supporting new operators", "adding_operator.xhtml#S4_1_supporting_new_operators", [
        [ "Adding new data types", "adding_operator.xhtml#S4_1_1_add_datatypes", null ],
        [ "Add a kernel", "adding_operator.xhtml#S4_1_2_add_kernel", null ],
        [ "Add a function", "adding_operator.xhtml#S4_1_3_add_function", null ],
        [ "Add validation artifacts", "adding_operator.xhtml#S4_1_4_add_validation", [
          [ "Add the reference implementation and the tests", "adding_operator.xhtml#S4_1_4_1_add_reference", null ],
          [ "Add dataset", "adding_operator.xhtml#S4_1_4_2_add_dataset", null ],
          [ "Add a fixture and a data test case", "adding_operator.xhtml#S4_1_4_3_add_fixture", null ]
        ] ]
      ] ]
    ] ],
    [ "Implementation Topics", "implementation_topic.xhtml", [
      [ "Windows", "implementation_topic.xhtml#implementation_topic_windows", null ],
      [ "Kernels", "implementation_topic.xhtml#implementation_topic_kernels", null ],
      [ "Multi-threading", "implementation_topic.xhtml#implementation_topic_multithreading", null ],
      [ "OpenCL kernel library", "implementation_topic.xhtml#implementation_topic_cl_scheduler", null ]
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
        [ "Enumerations", "globals_enum.xhtml", null ],
        [ "Enumerator", "globals_eval.xhtml", null ],
        [ "Macros", "globals_defs.xhtml", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"8b__mla_8cpp.xhtml",
"_acl_types_8h.xhtml#ac583b03d5acef3d22d0597b214b166bfae1da0b93c2a6ab07f455c07e4716cdf8",
"_c_l_2_arithmetic_addition_8cpp.xhtml#a6faaa6a770361529ed83471997cea631",
"_c_l_2_cast_8cpp.xhtml#afdc2c25fed3f6acd0b8794f281abf69e",
"_c_l_2_depthwise_convolution_layer_8cpp.xhtml#a13c6c2cbcccd87a61b589ee6e825e9cc",
"_c_l_2_fuse_batch_normalization_8cpp.xhtml#ab51cef146a30872194d612b0d16ab959",
"_c_l_2_normalization_layer_8cpp.xhtml#a4b1005817e3e850db979775f8c757030",
"_c_l_2_reorg_layer_8cpp.xhtml#afb0b52b4f9be21c5363b9843ba2b496e",
"_c_l_arg_min_max_layer_kernel_8h.xhtml",
"_c_l_g_e_m_m_heuristics_handle_8cpp.xhtml",
"_c_l_space_to_depth_layer_kernel_8h.xhtml",
"_cartesian_product_dataset_8h.xhtml",
"_cl_p_relu_8cpp.xhtml#a1ff1311212ea2de86763205e52649bff",
"_cpu_add_kernel_8cpp.xhtml",
"_cpu_gemm_direct_conv2d_8h_source.xhtml",
"_data_layer_visitor_8cpp_source.xhtml",
"_error_8h.xhtml#a5e3e8db314706c6f3d822bafa3d8d761",
"_g_e_m_m_matrix_multiply_native_8cpp.xhtml",
"_heuristic_tree_8h.xhtml#a1169278c54b2e2f0c3e6656b902da079acd6a9bd2a175104eed40f0d33a8b4020",
"_instruments_8h.xhtml#aac6b258eb313f96447f4f0e273431005a873865e4a94b7ec656c000502a719b90",
"_n_e_asymm_8h.xhtml#a70ecc99f2f6b646579b58463dc026671",
"_n_e_f_f_t_convolution_layer_8h.xhtml",
"_n_e_o_n_2_arithmetic_addition_8cpp.xhtml#a94537bbf49be5400891d56640b065254",
"_n_e_o_n_2_crop_resize_8cpp.xhtml#aaea8e95ff3943b3e9a1139a2bee38ab0",
"_n_e_o_n_2_elementwise_max_8cpp.xhtml#afcf78e82d4aa6d9686fb5b315a9233ca",
"_n_e_o_n_2_mean_std_dev_normalization_layer_8cpp.xhtml#a315ab8ee116b2cafc206479b69d09edb",
"_n_e_o_n_2_select_8cpp.xhtml#aa701b1f3cd76f80a5d6e9b463d556557",
"_n_e_reshape_layer_8h_source.xhtml",
"_open_c_l_8cpp.xhtml#ae2ab434699c41d87099b48a774db0b9d",
"_s_v_e_asymm_8inl_source.xhtml",
"_sub_tensor_8cpp.xhtml",
"a32__interleave6__block1__fp32__fp32_8hpp_source.xhtml",
"a64__merge__s32__4x4_8hpp_source.xhtml",
"add_2sve_2qasymm8__signed_8cpp.xhtml",
"arm__compute_2core_2_types_8h.xhtml#ab4e88c89b3b7ea1735996cc4def22d58a165f06116e7b8d9b2481dfc805db4619",
"arm__compute_2core_2utils_2logging_2_macros_8h_source.xhtml",
"arm__conv_2depthwise_2kernels_2sve__fp32__nhwc__3x3__s1__output2x2__mla__depthfirst_2generic_8cpp_source.xhtml",
"arm__gemm_2kernels_2sve__interleaved__u8u32__dot__8x3_v_l_2generic_8cpp_source.xhtml",
"clang-tidy_8h.xhtml#a71a01ba321c47e2723f21234bb57fb99",
"classarm__compute_1_1_c_l_activation_layer.xhtml",
"classarm__compute_1_1_c_l_compute_mean_variance.xhtml#a42dfa8d59da6fc6c832b72a48c545ba6",
"classarm__compute_1_1_c_l_exp_layer.xhtml#a074e10cfb217e657b9e81adeca2abc68",
"classarm__compute_1_1_c_l_instance_normalization_layer_kernel.xhtml#ae0b0bb7bc259ef0c85795e7518a81cc0",
"classarm__compute_1_1_c_l_pooling_layer.xhtml#af2bded4afffd2ab1110b9e45686ffd75",
"classarm__compute_1_1_c_l_sin_layer.xhtml#a533548787f021909ec0955e1730ce96c",
"classarm__compute_1_1_c_l_tuning_params.xhtml#a911324ec4a6233263cccbea220d2cdb8",
"classarm__compute_1_1_i_array.xhtml#a4b55468220c2029e7ee1594e34b2b21f",
"classarm__compute_1_1_i_simple_lifetime_manager.xhtml#a2f3b9c36674e2c4097552d9ffb30fbf2",
"classarm__compute_1_1_n_e_arithmetic_subtraction.xhtml#a6d6a3299e3615a5944c0e5c3e4c9b9e7",
"classarm__compute_1_1_n_e_depth_convert_layer.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_n_e_fuse_batch_normalization_kernel.xhtml#a245ec81a11c260c325ec481c41b15d96",
"classarm__compute_1_1_n_e_permute.xhtml#a4d665fc3e2c474c186b4e54f834c0c44",
"classarm__compute_1_1_n_e_space_to_batch_layer_kernel.xhtml#a2afc95185d007d3f3e8b4166cf103d07",
"classarm__compute_1_1_r_o_i_pooling_layer_info.xhtml#a70fd561a558e9f433088b64587a9cb62",
"classarm__compute_1_1_tensor_shape.xhtml#a99e09337e5b6ef762cd1f2d0bd10c346",
"classarm__compute_1_1cpu_1_1_cpu_pool2d.xhtml#a4ca9809c1ea30b9573581164d69b8a0d",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_elementwise_kernel.xhtml#a883429dd6cf828bfdd64b255afc458da",
"classarm__compute_1_1experimental_1_1_operator_tensor.xhtml#a47d74e4e51f9b1a636c4831bd747a97c",
"classarm__compute_1_1graph_1_1_flatten_layer_node.xhtml#adb48b5745c55605a2d4ec6f665bb7f3f",
"classarm__compute_1_1graph_1_1_normalization_layer_node.xhtml#adb48b5745c55605a2d4ec6f665bb7f3f",
"classarm__compute_1_1graph_1_1backends_1_1_fused_convolution_batch_normalization_function.xhtml",
"classarm__compute_1_1graph__utils_1_1_dummy_accessor.xhtml#a4a48cd865f33e4a907338c42d652d80f",
"classarm__compute_1_1opencl_1_1_cl_gemm_conv2d.xhtml#a1b57c95f96237eb42570587efbcbb654",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_im2_col_kernel.xhtml#a98cef52489d010f16f5d9761bbc9424f",
"classarm__compute_1_1test_1_1_context_scheduler_user.xhtml#ab260edfb937660d0ab9accd539cd9e0e",
"classarm__compute_1_1test_1_1framework_1_1_p_m_u.xhtml#ab30d96d1a047f366dd309049f139fa35",
"classarm__compute_1_1utils_1_1_i_image_data_feeder.xhtml#ae18857a90dabdfda0bf26ee448f46652",
"classarm__gemm_1_1_gemm_hybrid.xhtml#a7cde60064c88a363415fc0639d09cc3e",
"common_2batchnormalization__layer_8cl.xhtml",
"cvt_8h.xhtml#a34507275dfeb74e0b3dbf7591658b2cf",
"dir_82c95e891f4f6cb143d59bacb5289f7d.xhtml",
"elementwise__operation__quantized_8cl.xhtml#a5b0d9908c0af31eaa7a31d0b5cf8e56d",
"gemm__helpers_8h.xhtml#a01d4e59034b6b588d3f1c8c4f4788345",
"gemm__helpers_8h.xhtml#ae5e76ab03fa9ac5c4fb85d3b043173e3",
"gpu_2cl_2operators_2_cl_permute_8cpp_source.xhtml",
"helpers__asymm_8h.xhtml#ac5d336b40941ee7c63a750b3dc92b030",
"load_8h.xhtml#a34d952ba7147dd78ef05879bfead78f5",
"namespacearm__compute_1_1helpers_1_1fft.xhtml",
"nchw_2scale_8cl.xhtml",
"pmax_8h.xhtml#a77e7bf27d6c698eeb78c6e59c7736ff3",
"reference_2_d_f_t_8cpp.xhtml#afc02bc2a9b06db328e63c6bfbbdb23a8",
"reference_2_reorg_layer_8cpp.xhtml",
"scalar_2add_8h.xhtml#ac4358b7409261f013124bd6068aeee79",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a49bf041b15188b2ad497ddf991cab6a2",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#ad548b33c9a80420d227fa7df12f467a7",
"src_2runtime_2_c_l_2mlgo_2_utils_8cpp.xhtml#ade4abb3e900403674ebb640596c2117e",
"structarm__compute_1_1_direct_convolution_layer_output_stage_kernel_info.xhtml#ab233758aca2751c6e71a2f79baf7b92a",
"structarm__compute_1_1cpuinfo_1_1_cpu_isa_info.xhtml#afbdf12e6f5662a3bece45ef6153b6053",
"structarm__compute_1_1mlgo_1_1_heuristic_tree_1_1_branch_node.xhtml#a65d13dc93e2df5e8ab725263cf9f4ac5",
"structarm__compute_1_1wrapper_1_1traits_1_1promote_3_01float_01_4.xhtml",
"structarm__gemm_1_1_gemm_implementation_3_01_top_00_01_tret_00_01_nothing_01_4.xhtml#a3999cbfecd0c7dc11594c48a4a033a32",
"sve__interleaved__bf16fp32__mmla__8x3_v_l_8hpp_source.xhtml",
"tests_2framework_2_macros_8h.xhtml#a25cd4b5b342c79a328499d04939fa5c5",
"tests_2validation_2_n_e_o_n_2_reshape_layer_8cpp.xhtml#ada07f36b299d444952b3d0dcdcb21a47",
"tile__helpers_8h.xhtml#a832fce8769605f438ec0812c2d7c4ed7",
"utils_2_type_printer_8h.xhtml#aae534105c7ea67999ccbb34a0ed567cd"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';