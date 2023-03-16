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
        [ "Build for SVE or SVE2", "how_to_build.xhtml#S1_2_3_sve", null ],
        [ "Build for SME2", "how_to_build.xhtml#S1_2_4_sme", null ]
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
      [ "Building on a Windows host system (cross-compile)", "how_to_build.xhtml#S1_6_windows_host", [
        [ "Bash on Ubuntu on Windows (cross-compile)", "how_to_build.xhtml#S1_6_1_ubuntu_on_windows", null ],
        [ "Cygwin (cross-compile)", "how_to_build.xhtml#S1_6_2_cygwin", null ],
        [ "Windows on ARM (native build)", "how_to_build.xhtml#S1_6_3_WoA", null ]
      ] ],
      [ "OpenCL DDK Requirements", "how_to_build.xhtml#S1_7_cl_requirements", [
        [ "Hard Requirements", "how_to_build.xhtml#S1_7_1_cl_hard_requirements", null ],
        [ "Performance improvements", "how_to_build.xhtml#S1_7_2_cl_performance_requirements", null ]
      ] ],
      [ "Experimental Bazel and CMake builds", "how_to_build.xhtml#S1_8_experimental_builds", [
        [ "Bazel build", "how_to_build.xhtml#S1_8_1_bazel_build", [
          [ "File structure", "how_to_build.xhtml#S1_8_1_1_file_structure", null ],
          [ "Build options", "how_to_build.xhtml#S1_8_1_2_build_options", null ],
          [ "Example builds", "how_to_build.xhtml#S1_8_1_3_example_builds", null ]
        ] ],
        [ "CMake build", "how_to_build.xhtml#S1_8_2_cmake_build", [
          [ "File structure", "how_to_build.xhtml#S1_8_2_1_file_structure", null ],
          [ "Build options", "how_to_build.xhtml#S1_8_2_2_build_options", null ],
          [ "Example builds", "how_to_build.xhtml#S1_8_2_3_example_builds", null ]
        ] ]
      ] ]
    ] ],
    [ "Library Architecture", "architecture.xhtml", [
      [ "Core vs Runtime libraries", "architecture.xhtml#architecture_core_vs_runtime", null ],
      [ "Fast-math support", "architecture.xhtml#architecture_fast_math", null ],
      [ "BF16 acceleration", "architecture.xhtml#bf16_acceleration", null ],
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
        [ "Build multi-ISA binary", "architecture.xhtml#architecture_experimental_build_multi_isa", null ],
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
      ] ],
      [ "Concerns", "advanced.xhtml#Security", [
        [ "process running under the same uid could read another process memory", "advanced.xhtml#A", null ],
        [ "users could alter Compute Library related files", "advanced.xhtml#Malicious", null ],
        [ "concerns", "advanced.xhtml#Various", null ]
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
      [ "Inclusive language guideline", "contribution_guidelines.xhtml#S5_0_inc_lang", null ],
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
"_c_l_2_activation_layer_8cpp.xhtml#ad5b12b22c816f5f7724d386528bd3292",
"_c_l_2_cast_8cpp.xhtml#aba74be599f90af2f11d8d5d6fa9b72ea",
"_c_l_2_depth_convert_layer_8cpp.xhtml#a2805518f4013df2891e55b1cc32ce19a",
"_c_l_2_fill_8cpp.xhtml#ab882d1d77b88ac230e903bd68d04123a",
"_c_l_2_l_s_t_m_layer_quantized_8cpp.xhtml#a8e8c77fbfef699c0d6a7bd040e3b4d02",
"_c_l_2_r_n_n_layer_8cpp.xhtml#ae9a599036cbe43ea19a36c05257d5f87",
"_c_l_2_u_n_i_t_2_dynamic_tensor_8cpp.xhtml#a965151ed88fb7dac5a5efa2048820353",
"_c_l_f_f_t_digit_reverse_kernel_8cpp_source.xhtml",
"_c_l_range_8cpp_source.xhtml",
"_c_p_p_detection_post_process_layer_8cpp_source.xhtml",
"_cl_direct_conv_default_config_valhall_8h.xhtml",
"_cl_sub_8h_source.xhtml",
"_convert_fully_connected_weights_8h.xhtml",
"_cpu_gemm_interleave4x4_kernel_8cpp_source.xhtml",
"_d_f_t_8h.xhtml#ad12f9453958fc91c32e2ba138bbe85c2",
"_elementwise_square_diff_8cpp.xhtml#a770f93006059e614d72ce3868caabc8e",
"_fused_convolution_batch_normalization_node_8cpp.xhtml",
"_g_p_u_target_8h.xhtml#a735ac6c2a02e320969625308810444f3a8e081b1cdec7bc064a67f31560ef7fe0",
"_i_c_l_array_8h.xhtml#a631bac7c033a0d341c631870f9755217",
"_instruments_8h.xhtml#aac6b258eb313f96447f4f0e273431005a503aa8ac13b547591098b2ce77c28e99",
"_mutex_8h.xhtml#acded863dbfdd730829d4188d67eefcf0",
"_n_e_direct_convolution_detail_8h.xhtml#a8412c50a5db2c94cc74cc899096c3556",
"_n_e_mean_std_dev_normalization_kernel_8h_source.xhtml",
"_n_e_o_n_2_convert_fully_connected_weights_8cpp.xhtml#a512eb649fdb115f2dee5df9f1d156b16",
"_n_e_o_n_2_depthwise_convolution_layer_8cpp.xhtml#ac2330f9e9a464b7165a9c9c97c1f6ddf",
"_n_e_o_n_2_im2_col_8cpp.xhtml#a51fc7811115a811b78a1795634b22774",
"_n_e_o_n_2_prior_box_layer_8cpp.xhtml#a681969f1363fe7b3c3fa093ad3e70e17",
"_n_e_o_n_2_unstack_8cpp.xhtml#a20a17a8e59e65e544949213b6095d0a6",
"_node_fusion_mutator_8cpp.xhtml#a0b1901e17bde1eedfd8fb289dffe66d6",
"_post_op_utils_8h.xhtml#ad576060443e0e20b7c4a637d80aa94b6",
"_scheduler_factory_8h.xhtml",
"_sub_tensor_8h.xhtml",
"_workload_8h_source.xhtml",
"a64__interleaved__bf16fp32__dot__8x12_2x1_8cpp.xhtml",
"activation__layer__quant_8cl.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a1ce9b523fd4f3b5bbcadcd796183455aa57cec4137b614c87cb4e24a3d003a3e0",
"arm__compute_2core_2_utils_8h.xhtml#ae0204b49ec236bbaedb1cf96def276d4",
"arm__compute_2graph_2_types_8h.xhtml#acac9cbaeea226ed297804c012dc12b16ad87240455a60f188b358a370fe1a83a5",
"arm__conv_2pooling_2kernels_2a64__fp16__nhwc__avg__generic__depthfirst_2generic_8cpp_source.xhtml",
"arm__gemm_2kernels_2sve__ffhybrid__fp16__mla__6x4_v_l_2generic_8cpp_source.xhtml",
"ceq_8h.xhtml#aa399fa958ad81ea96aa9bc5300fbe94e",
"classarm__compute_1_1_access_window_transpose.xhtml#a7ad9fe991410dc3550f6c4fd6e87a708",
"classarm__compute_1_1_c_l_buffer_allocator.xhtml#a4d12a487a34bbcc481147eac34c7a3a1",
"classarm__compute_1_1_c_l_depth_to_space_layer_kernel.xhtml#a77c8af47fa2996f51bd5bdb61c527bc1",
"classarm__compute_1_1_c_l_fuse_batch_normalization_kernel.xhtml#a49821364328600f20fa538d64468709f",
"classarm__compute_1_1_c_l_mean_std_dev_normalization_layer.xhtml#a4a48674437b05deed54457c396161609",
"classarm__compute_1_1_c_l_reorg_layer_kernel.xhtml#a0245e9481fcbb4f5aa21ee19505f9ca3",
"classarm__compute_1_1_c_l_symbols.xhtml#a70f0c6b172261cc89e5d90239a8b70d0",
"classarm__compute_1_1_compute_anchors_info.xhtml#a70fd561a558e9f433088b64587a9cb62",
"classarm__compute_1_1_i_device.xhtml#a6a9a87290ceb85023335d5b87b4d6013",
"classarm__compute_1_1_iterator.xhtml",
"classarm__compute_1_1_n_e_cast.xhtml",
"classarm__compute_1_1_n_e_elementwise_power.xhtml#ae91cb21c9e461b11ee603866f46b85f3",
"classarm__compute_1_1_n_e_l2_normalize_layer_kernel.xhtml#a52cd2d465ac76389e383fda9997be37b",
"classarm__compute_1_1_n_e_range.xhtml#a88d81d0bfd6d1be443a59073f340a36f",
"classarm__compute_1_1_o_m_p_scheduler.xhtml#ae64eebaa07f4d2da6cc2ba538c3cb095",
"classarm__compute_1_1_sub_tensor_info.xhtml#a78bf77b2d9b959259f77a32b9a412184",
"classarm__compute_1_1cl__gemm_1_1_c_l_g_e_m_m_default_type_valhall.xhtml#a2146038e0a4fd533d583b67f9d38887e",
"classarm__compute_1_1cpu_1_1_cpu_winograd_conv2d_transform_input_kernel.xhtml#aa23ef6cab4bc10b53683165aac8dd314",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_gemm_matrix_multiply_kernel.xhtml#a883429dd6cf828bfdd64b255afc458da",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_cl_component_cast_settings.xhtml",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_cl_template_reshape.xhtml#acd5efd9e0abf91fc8eb0afdbd5cebff2",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_gpu_workload_sketch_1_1_implementation.xhtml#a53a99846286646c8e36726c50407b5c3",
"classarm__compute_1_1graph_1_1_dequantization_layer_node.xhtml#adb48b5745c55605a2d4ec6f665bb7f3f",
"classarm__compute_1_1graph_1_1_i_node_visitor.xhtml",
"classarm__compute_1_1graph_1_1_task_executor.xhtml",
"classarm__compute_1_1graph_1_1frontend_1_1_normalization_layer.xhtml#a225fb162b43cbef27d5a1dbd7a51c60d",
"classarm__compute_1_1mlgo_1_1_m_l_g_o_heuristics.xhtml#aef5085c60d40498a4e7127b5ae60eb42",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_direct_conv3d_kernel.xhtml",
"classarm__compute_1_1opencl_1_1kernels_1_1gemm_1_1_cl_gemm_default_config_native_midgard.xhtml#ad262a5c7f1f9eb5f84747c3fbc7466ff",
"classarm__compute_1_1test_1_1_simple_tensor_accessor.xhtml#aedcfdd4c3b92fe0d63b5463c7ad1d21e",
"classarm__compute_1_1test_1_1framework_1_1dataset_1_1_cartesian_product_dataset.xhtml#a0c62c15c8ed609e7e5e9518cf5f5c712",
"classarm__compute_1_1utils_1_1memory_1_1deep__unique__ptr.xhtml#ae58d8465d2b804da4fb9e750e3947a38",
"classarm__conv_1_1depthwise_1_1sme2__u8q__planar__5x5__s2__4rows__dot__za.xhtml#ac5c864b95e48d8920e527b49ec12361f",
"classarm__gemm_1_1_quantize_wrapper.xhtml#a8bf38de8dffa2ee89dc39ca7a5113a2f",
"cpu_2kernels_2activation_2generic_2sve_2fp32_8cpp.xhtml#ab01d09b8c1aea6b452dfebb7db046577",
"cpu_2kernels_2gemm__matrix__add_2list_8h.xhtml#a77e69bea899bb8caf2f855db36ba2307",
"crop__helper_8h.xhtml#ad3be09105863ab9faa320e7c0c253071",
"dir_46222a3755173d58e1d0ee7356bfffde.xhtml",
"dir_bac1e0ccb33c995a317bb4a028e4fabd.xhtml",
"dynamic__fusion_2gpu_2cl_2_cast_8cpp.xhtml#a705ee8c62e54819077618cf0244ff6d9",
"elementwise__binary_2generic_2sve2_2qasymm8__signed_8cpp_source.xhtml",
"functions_func.xhtml",
"gemm__helpers_8h.xhtml#a766e3dbd09f1bf9511dafcc19eb96dbe",
"generate__build__files_8py.xhtml#a8187411843a6284ffb964ef3fb9fcab3",
"graph__mobilenet_8cpp.xhtml",
"how_to_build.xhtml#S1_6_3_WoA",
"l2normlayer_2generic_2neon_2impl_8h.xhtml#a2242af116946637c1754030f5ac27fca",
"namespacearm__compute_1_1cl__dwc.xhtml",
"namespacemembers_vars_t.xhtml",
"pmin_8h.xhtml#a3f1bc2edd11ddc6647c50e9f52ef64e5",
"reference_2_copy_8cpp.xhtml#adea115d1cd3a84fa4620dad6b42744cf",
"reference_2_q_l_s_t_m_layer_normalization_8cpp.xhtml",
"rowsum__indirect__u8_8cpp.xhtml",
"sme2__interleave4_v_l__block2__fp32__bf16_8hpp.xhtml",
"softmax_2generic_2sve_2impl_8h.xhtml#a55285cff4817fd07fc12c9a63ac6da0c",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a5e5bc33c96b433a6aed9e17ac1698f9b",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#aee0e4aeffd58aa999fa138681b3daaa4",
"src_2runtime_2_c_l_2mlgo_2_utils_8h.xhtml#adb816b2ff36f9afae817527258bab9ea",
"structarm__compute_1_1_f_f_t1_d_info.xhtml",
"structarm__compute_1_1cpu_1_1_asm_gemm_info.xhtml#aba258627772f6f76f644911b027cbab5",
"structarm__compute_1_1experimental_1_1_post_op_eltwise_p_relu.xhtml#abe3f4fb4f063604571dc8906dab9c520",
"structarm__compute_1_1test_1_1framework_1_1_measurement.xhtml",
"structarm__conv_1_1depthwise_1_1_tensor_spec.xhtml#aa6545bf4fbb65fb7740e5ca3ee13b48e",
"structarm__gemm_1_1_kernel_description.xhtml#a0c9f19a7b68ad0dad836ee153d3a230a",
"sve__interleaved__s8s32__mmla__8x3_v_l_8hpp.xhtml",
"tests_2framework_2_macros_8h.xhtml#a8b3c06c9e7676202a34f711b1a7625fc",
"tests_2validation_2_n_e_o_n_2fft_8cpp_source.xhtml",
"tile__helpers_8h.xhtml#a7c1e6b07f6fef9e0bf9049ce6cd91de1",
"utils_2_type_printer_8h.xhtml#a6411da4e87e64e8859f8b725645ee9aa",
"validation_2_n_e_o_n_2_scale_8cpp.xhtml#ae78a2afc8cb469d8629dc419237e5c68"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';