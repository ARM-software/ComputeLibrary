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
      [ "Building on a Windows host system (cross-compile)", "how_to_build.xhtml#S1_6_windows_host", [
        [ "Bash on Ubuntu on Windows (cross-compile)", "how_to_build.xhtml#S1_6_1_ubuntu_on_windows", null ],
        [ "Cygwin (cross-compile)", "how_to_build.xhtml#S1_6_2_cygwin", null ],
        [ "Windows on ARM (native build)", "how_to_build.xhtml#S1_6_3_WoA", null ]
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
    [ "Dynamic Fusion Example: Conv2d + Elementwise Addition (OpenCL target)", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml", [
      [ "Describe the workload to run using OperatorGraph", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml#describe_workload_using_operator_graph", [
        [ "Add the first operator (root operator) Conv2d", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml#add_conv2d", null ],
        [ "Add the second operator Elementwise Add", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml#add_elementwise_add", null ]
      ] ],
      [ "Build ClWorkload", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml#build_clworkload", null ],
      [ "Run the fused operator workload with ClCompositeOperator", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml#run_fused_op_with_clcompositeoperator", [
        [ "Validate ClWorkload and Configure ClCompositeOperator", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml#configure_and_validate_clcompositeoperator", null ],
        [ "Run ClCompositeOperator", "example_dynamic_fusion_cl_conv2d_elementwise_add.xhtml#run_clcompositeoperator", null ]
      ] ]
    ] ],
    [ "Gemm Tuner", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml", null ],
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
    ] ],
    [ "Examples", "examples.xhtml", "examples" ]
  ] ]
];

var NAVTREEINDEX =
[
"8b__mla_8cpp.xhtml",
"_acl_types_8h.xhtml#ac583b03d5acef3d22d0597b214b166bfae1da0b93c2a6ab07f455c07e4716cdf8",
"_c_l_2_arithmetic_addition_8cpp.xhtml#a183e95e56ab3eaf97f3c7b3c54f06e16",
"_c_l_2_cast_8cpp.xhtml#aef7e11e7ca183d43f6c610c1cbb3edc3",
"_c_l_2_depth_to_space_layer_8cpp.xhtml#a3e4970f770f23aaa759017adf107da3e",
"_c_l_2_fuse_batch_normalization_8cpp.xhtml#a512eb649fdb115f2dee5df9f1d156b16",
"_c_l_2_mean_std_dev_normalization_layer_8cpp.xhtml#a0dc2519b952340c82956b937bfa5ed2e",
"_c_l_2_reduce_mean_8cpp_source.xhtml",
"_c_l_2_winograd_8cpp.xhtml#a92ea6ac2e4c17a8b8cb61325165e6d52",
"_c_l_g_e_m_m_auto_heuristics_8h.xhtml",
"_c_l_select_kernel_8h.xhtml",
"_c_p_p_types_8h.xhtml#aa41d7415a5386798147cccae2333d5d4a011d2705a24f3d97d766b9e9be3ebd35",
"_cl_gemm_lowp_matrix_multiply_native_kernel_8h.xhtml",
"_cl_workload_8cpp.xhtml#a0ba70c7ef99250a96e87378c163f2b0e",
"_cpu_dequantize_8cpp.xhtml",
"_cpu_mul_8cpp_source.xhtml",
"_elementwise_division_8cpp.xhtml#a7f18a00365b042db0e2f5f986ef5b5df",
"_flatten_layer_node_8cpp_source.xhtml",
"_g_e_m_m_matrix_multiply_reshaped_8cpp_source.xhtml",
"_i_cl_gemm_kernel_config_8h.xhtml",
"_kernel_types_8h.xhtml",
"_n_e_batch_normalization_layer_8h.xhtml",
"_n_e_fill_8cpp_source.xhtml",
"_n_e_o_n_2_arithmetic_addition_8cpp.xhtml#a7bc5c1ee768aa91aa62ce0ee9a958256",
"_n_e_o_n_2_copy_8cpp.xhtml#aa6be742b6456804dfb0d62030f90511b",
"_n_e_o_n_2_direct_convolution_layer_8cpp.xhtml#adc93519886bc5f865309acc70dd15236",
"_n_e_o_n_2_log_softmax_layer_8cpp.xhtml#a35c3eb596cd1e48831bb4e8bf8f22780",
"_n_e_o_n_2_reduce_mean_8cpp.xhtml#aaa36e9d0738a246005f10d0823e4f1a1",
"_n_e_r_o_i_pooling_layer_8cpp.xhtml",
"_offset_memory_pool_8cpp.xhtml",
"_quantization_info_8h.xhtml#ab8938bcb3ec0f5f8d93285eb3a28b701",
"_singleton_dataset_8h.xhtml#a295c6355f21fa6cf19ab8ef26ed2edd0",
"_validate_helpers_8h.xhtml#a891884386a167090b520930b164870f9",
"a64__hybrid__bf16fp32__dot__6x16_8hpp_source.xhtml",
"a64__u8__nhwc__max__generic__depthfirst_8hpp_source.xhtml",
"architecture.xhtml#architecture_experimental_clvk",
"arm__compute_2core_2_types_8h.xhtml#ad818ba0ecd4a87d8f1bb0d5b17f07830a1150a8d7752b01d30d91fe18fe9d8a54",
"arm__compute_2graph_2_type_printer_8h.xhtml",
"arm__conv_2depthwise_2kernels_2sve__fp32__packed__to__nhwc__5x5__s1__with__multiplier__output2x4__mla__depthfirst_2generic_8cpp.xhtml",
"arm__gemm_2kernels_2sve__hybrid__u8qa__dot__4x4_v_l_2generic_8cpp.xhtml",
"cgt_8h.xhtml#a96e5521a7f298420945cce5c15f41e08",
"classarm__compute_1_1_activation_layer_info.xhtml#af56abff12f887fddfa02e0bc18a318a1",
"classarm__compute_1_1_c_l_cast.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_c_l_dequantization_layer.xhtml#aa047ad145604827aa3f55253664ed61b",
"classarm__compute_1_1_c_l_g_e_m_m_deconvolution_layer.xhtml#a4f2603e6535b0504427d6296b9687428",
"classarm__compute_1_1_c_l_normalization_layer_kernel.xhtml#aaf8c501a2a845b96a3232927841cf9c7",
"classarm__compute_1_1_c_l_round_layer.xhtml#a7d132663d25399356998bd14af378d87",
"classarm__compute_1_1_c_l_tensor.xhtml#a2edd900d6f8eb9a995be55adfffbede5",
"classarm__compute_1_1_dimensions.xhtml#a08834d57877df4172a35bccc6719ab3a",
"classarm__compute_1_1_i_memory_manageable.xhtml#added32d63012f1fe88309043bdf65059",
"classarm__compute_1_1_l_s_t_m_params.xhtml#acce0a047e80de4da37b9add7acef765c",
"classarm__compute_1_1_n_e_concatenate_layer.xhtml#a323553c0bffc8dda63242a297512d151",
"classarm__compute_1_1_n_e_f_f_t_digit_reverse_kernel.xhtml#a9217160306107b2e11bfc20d8cea0a60",
"classarm__compute_1_1_n_e_max_unpooling_layer.xhtml",
"classarm__compute_1_1_n_e_reorg_layer_kernel.xhtml#aeaccf2356bf5ae491b7da58e3e6d0628",
"classarm__compute_1_1_pixel_value.xhtml#a66c4c1f8b1962d71162d7ac0b3ef65bc",
"classarm__compute_1_1_tensor_allocator.xhtml#ace0f71b15cbb5b468070f5a2fd4110ad",
"classarm__compute_1_1cpu_1_1_cpu_dequantize.xhtml#a7bd91d4a82cea79b87cd481e774614e1",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_convert_quantized_signedness_kernel.xhtml#a1a1da0fdc0f2ca33b0af98e8dd137c45",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_weights_reshape_kernel.xhtml#a883429dd6cf828bfdd64b255afc458da",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_i_tensor_desc_pack.xhtml#a0fbc1e6c05c5ba1bdb859da1af2115e6",
"classarm__compute_1_1graph_1_1_edge.xhtml#af20a11225e8c9ae2029e12f2ff05d95b",
"classarm__compute_1_1graph_1_1_in_place_operation_mutator.xhtml",
"classarm__compute_1_1graph_1_1backends_1_1_c_l_sub_tensor_handle.xhtml",
"classarm__compute_1_1graph_1_1frontend_1_1_slice_layer.xhtml",
"classarm__compute_1_1opencl_1_1_cl_dequantize.xhtml#a684a54d1fb1634a348a585c6b5e76df0",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_gemm_lowp_quantize_down_int32_scale_by_fixed_point_kernel.xhtml#ab13e88f1c4ff4db69033167a4d5da62c",
"classarm__compute_1_1test_1_1_assets_library.xhtml#ad6182f07b3eda32931598aa4f2bfc11a",
"classarm__compute_1_1test_1_1framework_1_1_instrument.xhtml#ac92db65cbe174915234b0d99b3ad4be7",
"classarm__compute_1_1utils_1_1_common_graph_options.xhtml#aa1fb173121a67e58388e74cf4f9f42f4",
"classarm__conv_1_1depthwise_1_1_depthwise_depthfirst_common.xhtml#ae75d6e84c912cdd1a26f79db4b8d7aaa",
"classarm__gemm_1_1_gemm_hybrid_quantized.xhtml#abf5a58f6feffeae31f48b750cba9303d",
"convert__fc__weights_8cl.xhtml",
"cpu_2kernels_2directconv2d_2nhwc_2neon_2fp32_8cpp_source.xhtml",
"cpu_2kernels_2roialign_2list_8h_source.xhtml",
"dir_03ea4a839f7d26eae2986fba1072d9b9.xhtml",
"dir_90bd923040ab47cd4e174fc5f8b9013a.xhtml",
"dwc__native__quantized__nhwc_8cl_source.xhtml",
"elementwise__operation__quantized_8cl.xhtml#ab05bd244cb8823c69d6dce6a23a758fd",
"functions_vars_x.xhtml",
"gemm__helpers_8h.xhtml#aba356632fc55f4bc64148a233e3c121e",
"globals_defs_a.xhtml",
"helpers__asymm_8h.xhtml#a3dc07539dda26f6cd2dcdc08a222292d",
"intrinsics_2add_8h.xhtml#a6f3b7a799d294a3dc8caa490f1073e2b",
"meanstddevnorm_2generic_2neon_2impl_8cpp.xhtml",
"namespacearm__conv.xhtml",
"nhwc_2direct__convolution_8cl.xhtml#adf4d58143bcf53e86afb9c16c66be1d3",
"range_2generic_2neon_2impl_8cpp.xhtml#ac574b455eb0708cabb76f3a6fbf6d01f",
"reference_2_gather_8cpp_source.xhtml",
"repeat_8h.xhtml#a567d26b128e2a92d05e039d1bb2c6c44",
"setlane_8h.xhtml#a0160f519b1fe681e083a2b43975e340e",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a3ccc3d81492204171c3da256ef223e64",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#ac4de77b1ce3370c09844370f9d72e977",
"src_2gpu_2cl_2operators_2_c_l_floor_8h_source.xhtml",
"structarm__compute_1_1_c_l_g_e_m_m_kernel_selection_params.xhtml#adc51cf0dbc18ebc0a4f9da57c22693ca",
"structarm__compute_1_1_valid_region.xhtml#a1de12d43d932fc3b1e05ff15e85212bf",
"structarm__compute_1_1experimental_1_1_memory_info.xhtml#a3afda29223eeda35afd21a6c5e121ba2",
"structarm__compute_1_1experimental_1_1dynamic__fusion_1_1_conv2d_content.xhtml#a4eaae335d8c8c54bdc7ccd8e71bc7c62",
"structarm__compute_1_1mlgo_1_1_g_e_m_m_config_reshaped_only_r_h_s.xhtml#a41618231c7d7b990c42fe6846b4dde27",
"structarm__compute_1_1utils_1_1_tensor_params.xhtml#ad1793c1a8dd3db1a8c4e2d76eadf0036",
"structarm__gemm_1_1_kernel_description.xhtml#a4249851da86a46c4a0d6fa21f6e49afd",
"sve__merge__fp16__3_v_lx8_8hpp_source.xhtml",
"tests_2framework_2_macros_8h.xhtml#aad0bc347eda11a18ce71dda6b5a3170c",
"tests_2validation_2_n_e_o_n_2fft_8cpp.xhtml#a6602a6f8c4c718ef681f3f32f388b407",
"tile__helpers_8h.xhtml#a8b2582ef60296962b7c8007a676efbfe",
"utils_2_type_printer_8h.xhtml#aae534105c7ea67999ccbb34a0ed567cd"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';