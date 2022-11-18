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
"_c_l_2_arithmetic_addition_8cpp.xhtml#a03600bb4c3b2c8e31e9ffd80d11f42a2",
"_c_l_2_cast_8cpp.xhtml#aeeb532afa00b43fcbfbc6c4f54bcc847",
"_c_l_2_depth_to_space_layer_8cpp.xhtml#a2cc453fd537f8e15d1126095444ff2c3",
"_c_l_2_floor_8cpp_source.xhtml",
"_c_l_2_logical_8cpp.xhtml#aed9725c086c468c15463004a7d03e575",
"_c_l_2_reduce_mean_8cpp.xhtml#a512eb649fdb115f2dee5df9f1d156b16",
"_c_l_2_winograd_8cpp.xhtml#a2bd84a2e27e9d1d931b54acdc1643bda",
"_c_l_fuse_batch_normalization_kernel_8h.xhtml",
"_c_l_runtime_context_8h.xhtml",
"_c_p_p_top_k_v_8cpp.xhtml",
"_cl_gemm_default_config_reshaped_rhs_only_valhall_8h_source.xhtml",
"_cl_transpose_kernel_8cpp.xhtml",
"_cpu_concatenate_width_kernel_8h.xhtml",
"_cpu_isa_info_8cpp.xhtml#aa94fa838ec994f56d56d252d54294454",
"_dequantization_layer_8h_source.xhtml",
"_execution_helpers_8cpp.xhtml#aa39e0698b2de495c235129b12fa39b08",
"_g_e_m_m_matrix_multiply_reshaped_8cpp.xhtml#a004ea5787390564adcaffa81b2a60565",
"_graph_8h_source.xhtml",
"_i_tensor_info_8h_source.xhtml",
"_m_l_g_o_parser_8h_source.xhtml",
"_n_e_copy_8cpp.xhtml",
"_n_e_l_s_t_m_layer_8cpp.xhtml",
"_n_e_o_n_2_cast_8cpp.xhtml#a525a5a53051ab7e5c60347a1dd244344",
"_n_e_o_n_2_depth_convert_layer_8cpp.xhtml#a96fa90ce9c6e2cda998684257d4d384b",
"_n_e_o_n_2_g_e_m_m_8cpp.xhtml#a775519522412acfeebf8a81dbc93cb93",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml#a8cc20a2bd7617c78ce1456bbf2e8b443",
"_n_e_o_n_2_stack_layer_8cpp.xhtml#a1219d2e4e290d33e099239f53fb0c40d",
"_n_e_split_8h.xhtml",
"_operator_graph_8h.xhtml#a8aad35ee7c2bfa06456fade719732d21",
"_s_v_e_asymm_8h.xhtml",
"_string_support_8h.xhtml#aebc1ea57c14482b68dac0c0dab766126a329acff975f3cc434528bec43bd5b6aa",
"_winograd_8h.xhtml#adacc73fb5c03e7a1273c0c81c8f8dad5",
"a64__interleave8__block4__u8__u8__summing_8hpp.xhtml",
"activation__float__helpers_8h.xhtml#ad9f618240c5338e554649d52a2588c33",
"arm__compute_2core_2_types_8h.xhtml#a1f8212fef242f87e34893c8db13fc16e",
"arm__compute_2core_2_validate_8h.xhtml#a069d7a20d1a7d62042f23039b356113b",
"arm__compute_2graph_2_types_8h.xhtml#acac9cbaeea226ed297804c012dc12b16af2ee14b628c0a45e1682de8f33983dc1",
"arm__conv_2pooling_2kernels_2sve__fp32__nhwc__max__generic__depthfirst_2generic_8cpp.xhtml",
"arm__gemm__compute__iface_8hpp.xhtml",
"clang-tidy_8h_source.xhtml",
"classarm__compute_1_1_c_l_arg_min_max_layer.xhtml#a4e98e9a0fd242396ab0f512983668489",
"classarm__compute_1_1_c_l_concatenate_layer.xhtml#a8e1b10bb7765c3cfd3871f0c317e75db",
"classarm__compute_1_1_c_l_f_f_t1_d.xhtml#a576116db2142d895eb5b54093554c9e5",
"classarm__compute_1_1_c_l_kernel_library.xhtml#a5a6865d04c8d3348860f3cb8100fdb97",
"classarm__compute_1_1_c_l_prior_box_layer.xhtml",
"classarm__compute_1_1_c_l_slice.xhtml#ae883a7cb96f6111b0e8bf3a64842c438",
"classarm__compute_1_1_c_l_winograd_convolution_layer.xhtml#a548924ae4a250d1010c286069c55d921",
"classarm__compute_1_1_i_array.xhtml#a9deaea40b2409ee68d354bdb9d3f8443",
"classarm__compute_1_1_i_simple_lifetime_manager.xhtml#a011196c118f00760877e8113b6256148",
"classarm__compute_1_1_n_e_arithmetic_addition.xhtml#af056b2278d99024dee71f6327ce25c83",
"classarm__compute_1_1_n_e_depth_convert_layer.xhtml",
"classarm__compute_1_1_n_e_fuse_batch_normalization.xhtml#aaa015eb4547a5cb0a9cb6adf3142dc66",
"classarm__compute_1_1_n_e_permute.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_n_e_space_to_depth_layer.xhtml",
"classarm__compute_1_1_runtime_context.xhtml#a2aecd3530be7fc6db21549eb7ee221d5",
"classarm__compute_1_1_tensor_shape.xhtml#acb74edf42335de0dca0da5158b704c4b",
"classarm__compute_1_1cpu_1_1_cpu_gemm_direct_conv2d.xhtml#a74ae0d6e96f38fecd38471431786b870",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_floor_kernel.xhtml#a7bd91d4a82cea79b87cd481e774614e1",
"classarm__compute_1_1experimental_1_1_operator_tensor.xhtml#aeef0fcab81ca9e1ee6904e25e8b0d8fb",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_dependency_graph_1_1_serial_id_allocator.xhtml#aebc5dc6e56fe725378002f1454d09978",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_operator.xhtml#a4749b98f551a7ec56649d34bd4de0da0",
"classarm__compute_1_1graph_1_1_flatten_layer_node.xhtml#adb48b5745c55605a2d4ec6f665bb7f3f",
"classarm__compute_1_1graph_1_1_node_execution_method_mutator.xhtml",
"classarm__compute_1_1graph_1_1backends_1_1_c_l_tensor_handle.xhtml#a0940aabdc8069229baa191d992e43140",
"classarm__compute_1_1graph_1_1frontend_1_1_strided_slice_layer.xhtml",
"classarm__compute_1_1opencl_1_1_cl_exp.xhtml#aea102dead92e3eedb5947de4358bc233",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_gemm_matrix_multiply_reshaped_kernel.xhtml#a4d77ae6c5a0d88056c1f0453500204c2",
"classarm__compute_1_1test_1_1_c_l_accessor.xhtml#aba5871b3e4a65d057ec1c28fce8b00ba",
"classarm__compute_1_1test_1_1framework_1_1_j_s_o_n_printer.xhtml#a3109b7e2d037aafe5422e92c7388d89f",
"classarm__compute_1_1utils_1_1_common_graph_validate_options.xhtml#a8047a16c04029dd91c525b1572282448",
"classarm__conv_1_1depthwise_1_1_depthwise_depthfirst_generic_with_multiplier.xhtml#ad5039576663a8b9a8ed58e927c040b20",
"classarm__gemm_1_1_gemm_interleaved.xhtml",
"core_2_c_l_2_c_l_helpers_8cpp.xhtml#a45b165e0796ef5e262fc5f05f03719c1",
"cpu_2kernels_2elementwise__binary_2generic_2sve_2fp32_8cpp.xhtml#a2e21843bb90caa3548333f95fcd41db1",
"cpu_2kernels_2scale_2sve_2list_8h.xhtml#ab8225bb56462d5dc53bcd91a7ab7e427",
"dir_15acb9606a67b692f1c5b11ae176d45d.xhtml",
"dir_9960658ec14b35d25ef38a57a625b1df.xhtml",
"elementwise__binary_2generic_2neon_2impl_8h.xhtml#a0460ff15e5977217b8fae5cf6abed0b1",
"elementwise__operation__quantized_8cl.xhtml#afc658c67277583af4eaca9a340d4128c",
"functions_vars_w.xhtml",
"gemm__helpers_8h.xhtml#ab90c95d1f3bfe0eabd7ce45a805b44fb",
"globals_defs.xhtml",
"helpers__asymm_8h.xhtml#a3d8a3968a069eb8abbc28d31c20f6b8c",
"intrinsics_2add_8h.xhtml#a5facaa0c056dc73dfa6a49cde91240c4",
"md_examples_gemm_tuner__r_e_a_d_m_e.xhtml",
"namespacearm__compute_1_1weights__transformations.xhtml",
"nhwc_2direct__convolution_8cl.xhtml#a654a9a18953d08fc64b2526a039bee7b",
"range_2generic_2neon_2impl_8cpp.xhtml",
"reference_2_gather_8cpp.xhtml",
"repeat_8h.xhtml#a364708cdd26aa3540ddbf93741283798",
"select_2generic_2neon_2integer_8cpp.xhtml#aa9e21306be38f9b98e4ef4f830c2799b",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a31f6a7d430f01eb0015a574019a7a682",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#ab5975aa797b02c8e21551e4a3425c499",
"src_2core_2utils_2logging_2_helpers_8cpp.xhtml",
"structarm__compute_1_1_blob_info.xhtml#a11702cc272b3bf3c15074162cf3b7a05",
"structarm__compute_1_1_pooling_layer_info.xhtml#a16875a1122accc1277a22e0ddbb8e94d",
"structarm__compute_1_1detail_1_1linear.xhtml#a59101c0482fb40b207bcaf599771d223",
"structarm__compute_1_1experimental_1_1dynamic__fusion_1_1_cl_kernel_fusion_group.xhtml#a55774ab9526a3034bf7bf01187a09689",
"structarm__compute_1_1graph_1_1_execution_workload.xhtml#aa8479921c38182de22a1a1a4d24481e6",
"structarm__compute_1_1test_1_1framework_1_1dataset_1_1_singleton_dataset_1_1iterator.xhtml#acb3b5c956de1c36eb15a8fc1fa40a20b",
"structarm__gemm_1_1_activation.xhtml#a1d1cfd8ffb84e947f82999c682b666a7",
"sve__fp32__nhwc__3x3__s1__output4x4__mla__depthfirst_8hpp.xhtml",
"tensor__transform_8h.xhtml#a9e3d9626e902ce6f9555d3580e13011e",
"tests_2validation_2_helpers_8cpp.xhtml",
"tile__helpers_8h.xhtml#a221f1b823338e5036bff6b0b0541e83e",
"utils_2_type_printer_8h.xhtml#a18d43401e8f84ed020429b41ec4e99f5",
"validation_2_c_l_2_scale_8cpp.xhtml#ad0d65fc433c44fe4100b9533c5e4a770"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';