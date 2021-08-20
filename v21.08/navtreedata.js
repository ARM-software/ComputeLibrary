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
"_acl_types_8h.xhtml#ad90cee8eef5037938d5c0c7a678f3866",
"_c_l_2_arithmetic_addition_8cpp.xhtml#a9fcca2851c7c8c023bcc3266976c6a5c",
"_c_l_2_cast_8cpp_source.xhtml",
"_c_l_2_depthwise_convolution_layer_8cpp.xhtml#a8c8e41b008166f75f84b8abef4622e6c",
"_c_l_2_g_e_m_m_lowp_8cpp.xhtml#a3f60fb67a9460272473604cc8f936e39",
"_c_l_2_p_relu_layer_8cpp.xhtml#a6ba2767e0315bf55a585eeb0f61ac496",
"_c_l_2_select_8cpp.xhtml#aebc10ebaa1aa56ca405a5bc8b7e11026",
"_c_l_arg_min_max_layer_kernel_8cpp_source.xhtml",
"_c_l_g_e_m_m_default_type_valhall_8h_source.xhtml",
"_c_l_space_to_depth_layer_8cpp_source.xhtml",
"_c_p_p_upsample_kernel_8cpp_source.xhtml",
"_cl_p_relu_8cpp.xhtml",
"_cpu_col2_im_kernel_8cpp.xhtml",
"_cpu_gemm_lowp_quantize_down_int32_to_int16_scale_by_fixed_point_kernel_8h.xhtml",
"_default_l_w_s_heuristics_8cpp_source.xhtml",
"_exceptions_8h.xhtml#aca1fd1d8935433e6ba2e3918214e07f9",
"_g_e_m_m_matrix_multiply_reshaped_8cpp.xhtml#a0be61ccdaac6fb2f875c3baebbbff845",
"_i_cpu_kernel_8h.xhtml#a5e357b8c8241b9204a9d5e502565c20d",
"_logical_8h_source.xhtml",
"_n_e_channel_shuffle_layer_8h_source.xhtml",
"_n_e_generate_proposals_layer_kernel_8h_source.xhtml",
"_n_e_o_n_2_cast_8cpp.xhtml#a1100482c249e44b8f4c453fba560555c",
"_n_e_o_n_2_depth_convert_layer_8cpp.xhtml#aa2d7a9f123e98cad9f150dde4e095c13",
"_n_e_o_n_2_g_e_m_m_lowp_8cpp.xhtml#afd854c13e81aec26443edc864432e61f",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml#ae3c85b05691e133bf8fbacd0417e1a4b",
"_n_e_o_n_2_tile_8cpp.xhtml#a630c291504e0242d4ce76ef1b7c884b6",
"_n_e_symm_8h.xhtml#aa9985607399e329fcb371a92c84a6acd",
"_pretty_printer_8cpp_source.xhtml",
"_shape_calculator_8h.xhtml#ac4d688e137d670d209b647ec37592a92",
"_utility_8h.xhtml#a1e384f81bb641de61df2800a432c51fe",
"a64__fp32__nhwc__max__2x2__s1__output2x2__depthfirst_8hpp_source.xhtml",
"a64__transpose__interleave__16__1x4_8hpp_source.xhtml",
"architecture.xhtml#programming_model_cl_neon",
"arm__compute_2core_2_utils_8h.xhtml#ab14153fb809c18823af3c9c8bc4286cb",
"arm__compute_2graph_2_types_8h.xhtml#acac9cbaeea226ed297804c012dc12b16ae1b0b91c003f871664a4e289977ef02e",
"arm__conv_2pooling_2kernels_2sve__fp16__nhwc__avg__generic__depthfirst_2generic_8cpp_source.xhtml",
"bsl_8h.xhtml#ab2a026522ed9848e50efc0d0c1c983d2",
"classacl_1_1_tensor.xhtml#a993f232f4c79265dde0292b14d827759",
"classarm__compute_1_1_c_l_batch_normalization_layer_kernel.xhtml#ad2ee56da9dc1ec4a93968576e564eedc",
"classarm__compute_1_1_c_l_deconvolution_layer_upsample.xhtml#a7f5d6054403fbfe5102c46f50987cf50",
"classarm__compute_1_1_c_l_fine_s_v_m_memory_region.xhtml#a322777fe48f83f5eeb8083499a578aaa",
"classarm__compute_1_1_c_l_logical_or.xhtml#a25b288c9dc766a7fcf2ff27ec1733c3e",
"classarm__compute_1_1_c_l_range_kernel.xhtml#a1f1b7ad0df178d4e0656ad22db60a691",
"classarm__compute_1_1_c_l_stack_layer_kernel.xhtml#a8980cefb7252a70a4d05dbc9ecc446d5",
"classarm__compute_1_1_c_p_p_top_k_v_kernel.xhtml",
"classarm__compute_1_1_i_c_l_tensor.xhtml#a7b94593f2b06f00d380ec3a3e6abccb2",
"classarm__compute_1_1_i_transform_weights.xhtml",
"classarm__compute_1_1_n_e_bitwise_xor_kernel.xhtml#a837b139cf977a6c4530e3d574fcceef2",
"classarm__compute_1_1_n_e_elementwise_min.xhtml#ae69e37d2f4ec355766f2fe252ff8ce8d",
"classarm__compute_1_1_n_e_l2_normalize_layer.xhtml#ac2f579ddecdff2d368add08094105173",
"classarm__compute_1_1_n_e_r_o_i_pooling_layer.xhtml#a72ac8281fc4849dc7256265502aae741",
"classarm__compute_1_1_n_e_unstack.xhtml#a515c2c1520db8f1cdbfdc2bc6e66b8b3",
"classarm__compute_1_1_sub_tensor_info.xhtml#a16d275e5270f2c36a04764c863d852b6",
"classarm__compute_1_1cpu_1_1_cpu_aux_tensor_handler.xhtml#ad0334363467dcf7fc3375ed2ffcb72f3",
"classarm__compute_1_1cpu_1_1_i_cpu_winograd_conv2d_transform_weights_kernel.xhtml#ae4a40fc825c6741cd99c171544a6eda3",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_mul_kernel.xhtml#a883429dd6cf828bfdd64b255afc458da",
"classarm__compute_1_1graph_1_1_depth_to_space_layer_node.xhtml",
"classarm__compute_1_1graph_1_1_i_node_visitor.xhtml#a3ab3d1a5fedf335e979356b1236af8a9",
"classarm__compute_1_1graph_1_1_tensor.xhtml#aa3dc08c2e62f20a8fbfbcc46c6b4cb68",
"classarm__compute_1_1graph_1_1frontend_1_1_prior_box_layer.xhtml",
"classarm__compute_1_1opencl_1_1_cl_complex_mul.xhtml",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_gemm_lowp_offset_contribution_output_stage_kernel.xhtml#a02f6710210cba23985513487eed1391d",
"classarm__compute_1_1test_1_1_assets_library.xhtml#ab419bdd4d1b71e56517cbd99428e3740",
"classarm__compute_1_1test_1_1framework_1_1_instrument.xhtml#a4d3582d354eb3e9028f627624ad7e126",
"classarm__compute_1_1utils_1_1_common_graph_options.xhtml#a23eb83ee3164eeade04dfbefed43beda",
"classarm__conv_1_1depthwise_1_1_depthwise_depthfirst_quantized.xhtml#aced4dd5c55c1e2d369a51cec6cd5d92e",
"classgemm__tuner_1_1_common_gemm_example_options.xhtml",
"cpu_2kernels_2scale_2neon_2fp16_8cpp_source.xhtml",
"dir_638da73e9a5ccd710c765a5874cc9f96.xhtml",
"dup__n_8h.xhtml#a5cab0757ae9983d6f81120e1eeb0d495",
"gemm__helpers_8h.xhtml#a0ea97fdaed3fa6414afbb5022b07b9e4",
"gemm__helpers_8h_source.xhtml",
"graph__googlenet_8cpp.xhtml#a3c04138a5bfe5d72780bb7e82a18e627",
"helpers__asymm_8h_source.xhtml",
"load__store__utility_8h.xhtml#a7249d30c01773ebe7a7c15899550967f",
"namespacearm__compute_1_1test_1_1convolution__3d_1_1detail.xhtml",
"neon_2elementwise__list_8h.xhtml#a29b215a8ed4583d1d4ac5d1e226bf9ef",
"pooling__u8_8cpp.xhtml",
"reference_2_floor_8cpp_source.xhtml",
"reference_2_transpose_8cpp.xhtml#acf9a40691cf4bd2c9af93bc806358a24",
"src_2common_2_types_8h.xhtml#ae98a46f4ea1a43ca48acaf15d2eb7113",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#aa2ec2f7bb8c1566d6c5034595807d8bb",
"src_2runtime_2_c_l_2mlgo_2_m_l_g_o_heuristics_8cpp.xhtml",
"structarm__compute_1_1_c_l_tuning_info.xhtml#a93490954be8daf7e7326c5375e122259",
"structarm__compute_1_1cpu_1_1_asm_gemm_info.xhtml#a0cbb6ae220a0fc33e557441771cf75a6",
"structarm__compute_1_1mlgo_1_1parser_1_1_token.xhtml#a5c33da5929c9f2f9ef8d006d2350e6ec",
"structarm__conv_1_1depthwise_1_1_depthwise_implementation.xhtml#ae2c7913d3939773504989e6086b883c5",
"structarm__gemm_1_1_gemm_implementation.xhtml#a4a30e706a3d061aacca1b73da4a8c73e",
"sve__hybrid__u8u32__dot__6x4_v_l_2a64fx_8cpp.xhtml",
"tests_2_utils_8h.xhtml#af8e1c85e80e1784f08ca535a18feacf2",
"tests_2validation_2_n_e_o_n_2_reshape_layer_8cpp.xhtml#a1c9abced5979b89a9b7940e5ba20fc1f",
"tile__helpers_8h.xhtml#a83971fa3e89259a6fc57310c2a85d22d",
"utils_2_utils_8cpp.xhtml#acdab4f47904c52221c85cc80f408f183"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';