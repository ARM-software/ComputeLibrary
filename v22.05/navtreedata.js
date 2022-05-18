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
"_c_l_2_arithmetic_addition_8cpp.xhtml#a6faaa6a770361529ed83471997cea631",
"_c_l_2_cast_8cpp.xhtml#afd68cdfe92763f60825c3c355b9af9ce",
"_c_l_2_depthwise_convolution_layer_8cpp.xhtml#a0dabc3ee4c06c701e99214462233a594",
"_c_l_2_fuse_batch_normalization_8cpp.xhtml#aad02f34f2471c5d477e44f9d79a2f83b",
"_c_l_2_normalization_layer_8cpp.xhtml#a09929aa316ea25237bfff2f344f05634",
"_c_l_2_reduction_operation_8cpp.xhtml#a9ad6319a33abd7333640c9112c520814",
"_c_l_2_winograd_8cpp.xhtml#aef554aa97e83764c6914288c18309a00",
"_c_l_g_e_m_m_convolution_layer_8h_source.xhtml",
"_c_l_space_to_batch_layer_8cpp_source.xhtml",
"_c_p_p_types_8h.xhtml#aa41d7415a5386798147cccae2333d5d4aecf3e2493afebcfefb0b9b86bd50ef0d",
"_cl_gemm_lowp_quantize_down_int32_scale_kernel_8h_source.xhtml",
"_color_convert_helper_8h.xhtml#a9f946d571e7fe2fed0aee64adcfc9e3f",
"_cpu_elementwise_8h.xhtml#adb6ece39851b42c09fe28c2ab0b27adf",
"_cpu_quantize_kernel_8cpp.xhtml",
"_elementwise_operations_8cpp.xhtml#a89bf6564436779db314a3c36b7d1fa88",
"_function_helpers_8h.xhtml#a69dd1fc17c7a15f4125873be182c8c76",
"_g_p_u_target_8h.xhtml#a735ac6c2a02e320969625308810444f3a57a3b40730a5af7ea3e13eb4bba56d82",
"_i_pool_manager_8h_source.xhtml",
"_m_l_g_o_parser_8cpp.xhtml#a21adc9dc5e39e405fe91b4d81ec33a3a",
"_n_e_channel_shuffle_layer_kernel_8h.xhtml",
"_n_e_gather_8h_source.xhtml",
"_n_e_o_n_2_bitwise_not_8cpp.xhtml#aaa1de70e01038ba111df3ae69b6d81d9",
"_n_e_o_n_2_depth_concatenate_layer_8cpp.xhtml#a884bd3d0dcec77be72e8a3d30b69fb82",
"_n_e_o_n_2_fill_border_8cpp_source.xhtml",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml",
"_n_e_o_n_2_space_to_batch_layer_8cpp.xhtml#aab5524a31f22d59089d4387bb2d0c3c1",
"_n_e_slice_8cpp_source.xhtml",
"_open_c_l_timer_8h.xhtml#ac7a4d964c8ac4712a767d7e72be73a8e",
"_round_layer_8cpp.xhtml#ae7b9eaebbc5f863aec87551728eba105",
"_strided_slice_layer_node_8cpp.xhtml",
"_window_iterator_8cpp.xhtml#a8619d01d562aa6c1f6e47dee70ca18f3",
"a64__interleave8__block8__s8__s8_8hpp_source.xhtml",
"activation__float__helpers_8h.xhtml#afb5afc4d2c7dde25c6d9725572255f6d",
"arm__compute_2core_2_types_8h.xhtml#a23d9f0c01c9e120dfb828ee922b7a8ae",
"arm__compute_2core_2_validate_8h.xhtml#aa8b68ddd24c352a4cdd4fb1eea263429",
"arm__compute_2runtime_2_c_l_2functions_2_c_l_g_e_m_m_8h_source.xhtml",
"arm__gemm_2kernels_2a64__hybrid__fp32bf16fp32__mmla__6x16_2generic_8cpp_source.xhtml",
"cast_8cl.xhtml#a5b0d9908c0af31eaa7a31d0b5cf8e56d",
"classarm__compute_1_1_access_window_rectangle.xhtml#a4c22b1f2583d9a660f7bb13e0b07fa1f",
"classarm__compute_1_1_c_l_bitwise_not.xhtml#aa047ad145604827aa3f55253664ed61b",
"classarm__compute_1_1_c_l_depth_convert_layer.xhtml#a1db278485dac6440e16a8d224c3c1372",
"classarm__compute_1_1_c_l_fully_connected_layer.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_c_l_mean_std_dev_normalization_kernel.xhtml#a4a48674437b05deed54457c396161609",
"classarm__compute_1_1_c_l_reduction_operation_kernel.xhtml#aa037c3742d874eab9ba733ee60ef145d",
"classarm__compute_1_1_c_l_symbols.xhtml#a31fd3504c695582b52ee2426dc71c1d7",
"classarm__compute_1_1_c_p_u_info.xhtml#ae3a44e5d711e5707a9d974d085bcccdd",
"classarm__compute_1_1_i_context.xhtml#af601a7b7ed3dc8865ba22584c394da6f",
"classarm__compute_1_1_iterator.xhtml#a599f5025b7e6b8bfead740a88e56d5bc",
"classarm__compute_1_1_n_e_channel_shuffle_layer_kernel.xhtml",
"classarm__compute_1_1_n_e_elementwise_unary_layer.xhtml#a15aaf27b046c87962fd4a4871bdad39e",
"classarm__compute_1_1_n_e_l_s_t_m_layer.xhtml#aa899feaf94d69eb04afb0cd412869548",
"classarm__compute_1_1_n_e_range_kernel.xhtml#ae17557c16785ace25e05ecf95c08ed33",
"classarm__compute_1_1_offset_memory_pool.xhtml",
"classarm__compute_1_1_sub_tensor_info.xhtml#ac74736e3863207232a23b7181c1d0f44",
"classarm__compute_1_1cpu_1_1_cpu_convert_fully_connected_weights.xhtml",
"classarm__compute_1_1cpu_1_1kernel_1_1_cpu_gemm_assembly_wrapper_kernel.xhtml#aefdb54cefd49dfa1bb7726dd38d18598",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_max_unpooling_layer_kernel.xhtml#a883429dd6cf828bfdd64b255afc458da",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_dependency_graph.xhtml#a3ae00f571426a9c55804b0591646b396",
"classarm__compute_1_1graph_1_1_detection_output_layer_node.xhtml#a3f18a7449b9d7fc9e5fec212b8e61710",
"classarm__compute_1_1graph_1_1_i_node_visitor.xhtml#a0a1deb26be8ff3f1282f6cbb84a134db",
"classarm__compute_1_1graph_1_1_tensor.xhtml",
"classarm__compute_1_1graph_1_1frontend_1_1_normalize_planar_y_u_v_layer.xhtml",
"classarm__compute_1_1mlgo_1_1parser_1_1_token_stream.xhtml",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_fill_kernel.xhtml#a85f336590ad7806fb8c8b7455afc867f",
"classarm__compute_1_1test_1_1_array_accessor.xhtml#a32e914dbe65c0591501e44790957bb0f",
"classarm__compute_1_1test_1_1framework_1_1_framework.xhtml#a5a2d19934af27623634c5ab3c47a9446",
"classarm__compute_1_1test_1_1framework_1_1dataset_1_1_zip_dataset.xhtml#af9593d4a5ff4274efaf429cb4f9e57cc",
"classarm__conv_1_1depthwise_1_1_depthfirst_multiplier_strategy.xhtml#a1acb91503a7760505b240fc04bbab5ad",
"classarm__gemm_1_1_gemm_hybrid.xhtml#a82ec71716669b6765670e0ed242f5a47",
"common_2batchnormalization__layer_8cl_source.xhtml",
"cpu_2kernels_2crop_2list_8h.xhtml#afce91e14811fc7ab82c764efb12c0443",
"cpu_2kernels_2range_2generic_2neon_2fp32_8cpp.xhtml#a1d3f077656e993e4d09054fb5addad39",
"depthwiseconv2d_2generic_2neon_2impl_8cpp.xhtml#ae3c2bd486bdab2aab84d1e0fc1afec53",
"dir_85562dfbf17db5d8a5ce180a29f04330.xhtml",
"dwc__native__quantized__nhwc_8cl.xhtml#a484eede56f4a958163015d2beddb372e",
"elementwise__binary_2generic_2sve_2impl_8cpp.xhtml#ac9ab88f190528dc081524370ed1dfdc8",
"functions_e.xhtml",
"gemm__helpers_8h.xhtml#a7633831eb41cdf133dddb7743580f53d",
"generic__quantized__dot__product_8hpp.xhtml#aaf545a8af159f092c65d88af297ba84b",
"graph__squeezenet__v1__1_8cpp.xhtml",
"include__functions__kernels_8py.xhtml#a6893570fc47fd11203da7f397a0b7f2d",
"load__store__utility_8h.xhtml#adaf7b05a85b39c28a9d8c6bb3f1e1781",
"namespacearm__compute_1_1test_1_1convolution__3d.xhtml",
"neg_8h_source.xhtml",
"pow_8h.xhtml",
"reference_2_fuse_batch_normalization_8cpp_source.xhtml",
"reference_2_unstack_8cpp.xhtml#ab3cc70aa61afb1c147878c2c9e7646ed",
"scale_2sve_2qasymm8__signed_8cpp_source.xhtml",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a27349cf6810998445207a85ac272bcfc",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#aaa93f37edb33e102ecb9dae5eafaaf43",
"src_2core_2common_2_registrars_8h.xhtml#acfa632794dcd2aaa48882f9804f191b7",
"structacl_1_1_tensor_pack_1_1_pack_pair.xhtml#a56104a5fbf07dd0724ba94122df99001",
"structarm__compute_1_1_pooling_layer_info.xhtml#a38fe393e11c69a5d459a75e2858e9f17",
"structarm__compute_1_1detail_1_1linear.xhtml#a08d488da4b10e6ee4cdbbb9781baaa7e",
"structarm__compute_1_1experimental_1_1dynamic__fusion_1_1_cl_kernel_graph.xhtml#adda92e981d244d14d1266f9746bccc6d",
"structarm__compute_1_1graph_1_1descriptors_1_1_unary_eltwise_layer_descriptor.xhtml#a36c3d2d309d852df2e5cb7df0e2332cf",
"structarm__compute_1_1utils_1_1_common_params.xhtml#aa9a98a0656a58548b5582d40fac47b06",
"structarm__gemm_1_1_gemm_implementation_3_01_top_00_01_tret_00_01_nothing_01_4.xhtml#aa3af9748daae0117fe4066ae1b3354a0",
"sve__interleaved__s8s32__mmla__8x3_v_l_8hpp.xhtml",
"tests_2framework_2_macros_8h.xhtml#a5948998e4f7badeca767900ba91334ac",
"tests_2validation_2_n_e_o_n_2_reshape_layer_8cpp.xhtml#aabf65afb5f889e8b0645a2c1329c554d",
"tile__helpers_8h.xhtml#a7876b97412192ce0d951d148ca20deeb",
"utils_2_type_printer_8h.xhtml#a9caac285297030f50a972095e767aff9"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';