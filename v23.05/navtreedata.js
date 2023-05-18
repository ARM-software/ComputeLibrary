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
      [ "Building on a Windows® host system (cross-compile)", "how_to_build.xhtml#S1_6_windows_host", [
        [ "Bash on Ubuntu on Windows® (cross-compile)", "how_to_build.xhtml#S1_6_1_ubuntu_on_windows", null ],
        [ "Cygwin (cross-compile)", "how_to_build.xhtml#S1_6_2_cygwin", null ],
        [ "Windows® on Arm™ (native build)", "how_to_build.xhtml#S1_6_3_WoA", null ]
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
      ] ],
      [ "Building with support for fixed format kernels", "how_to_build.xhtml#S1_8_fixed_format", [
        [ "What are fixed format kernels?", "how_to_build.xhtml#S1_8_1_intro_to_fixed_format_kernels", null ],
        [ "Building with fixed format kernels", "how_to_build.xhtml#S1_8_2_building_fixed_format", null ]
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
        [ "Typedefs", "functions_type.xhtml", "functions_type" ],
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
"_acl_types_8h.xhtml#ac20c0c3a354c89f830a326f6b916a6b9",
"_c_l_2_arg_min_max_8cpp.xhtml#a2e42583c6ea5741312d3184cfe0e0f0f",
"_c_l_2_cast_8cpp.xhtml#ac7944755dc10872060bd9046a28be272",
"_c_l_2_depth_convert_layer_8cpp.xhtml#a690ae73cb61d20552a1fa63e4fcd6626",
"_c_l_2_fill_8cpp.xhtml#af6c649617ab823f51e329840320def77",
"_c_l_2_l_s_t_m_layer_quantized_8cpp.xhtml#a924cb272ca88beabaf2f262731d226fd",
"_c_l_2_quantization_layer_8cpp.xhtml#aaf95124de0430fcafd6212d26bb890b5",
"_c_l_2_transpose_8cpp.xhtml#a7cd972e9a808a3d80e82be51b1d3c687",
"_c_l_f_f_t1_d_8cpp.xhtml",
"_c_l_r_o_i_align_layer_kernel_8cpp.xhtml",
"_c_p_p_box_with_non_maxima_suppression_limit_8cpp.xhtml#a4feaaa70771629f4b5dcf3b219c8b647",
"_cl_direct_conv3d_8cpp_source.xhtml",
"_cl_pool3d_kernel_8cpp.xhtml",
"_concatenate_layer_8cpp_source.xhtml",
"_cpu_gemm_assembly_dispatch_8cpp.xhtml#a42247993f81d6a8f73d2e0cb805c663b",
"_cpu_tensor_8cpp.xhtml",
"_elementwise_log_8cpp.xhtml#a76652a64bcd1a2d903f5ee31f299355c",
"_fixture_8h_source.xhtml",
"_g_e_m_m_matrix_multiply_reshaped_8cpp.xhtml#ab053b493888137886d24d6ba9f9a2ccc",
"_gpu_workload_context_8h.xhtml#aa334134c2d2596d8de13df6f9cb67d1da88183b946cc5f0e8c96b2e66e1c74a7e",
"_i_stream_operators_8h.xhtml#a8aa83abfaa698052fbf74953829a607e",
"_m_mapped_file_8cpp_source.xhtml",
"_n_e_bitwise_not_8h_source.xhtml",
"_n_e_function_factory_8cpp_source.xhtml",
"_n_e_o_n_2_arithmetic_subtraction_8cpp.xhtml#aa3f24910340c319e6ff1ce23a9213150",
"_n_e_o_n_2_deconvolution_layer_8cpp.xhtml#a15615dccb81b27e84867632665b3190b",
"_n_e_o_n_2_elementwise_max_8cpp_source.xhtml",
"_n_e_o_n_2_mat_mul_8cpp.xhtml#a4309ecc6c8d0109613a5d6b04a7482a2",
"_n_e_o_n_2_reduce_mean_8cpp.xhtml#a61a2fd1e747e9ee16cefcbd2f87d9ad1",
"_n_e_r_o_i_align_layer_kernel_8h.xhtml",
"_object_8h.xhtml#a2c0ee4eb5bed32d6fb8358d113995aa4a722ad2d05ecf4868b00c5484b82fd808",
"_quantization_info_8h.xhtml#aea6dad9b7cd5d5028a50e2aceafa7d1d",
"_shape_calculator_8h.xhtml#a97d2a3c7b284e3a976b3bf1dbea10af9",
"_tensor_shape_8cpp.xhtml#aebfc2f8ea56e865959855650b84d9237",
"a64__ffinterleaved__fp32__mla__8x12_8hpp_source.xhtml",
"a64__s8__nhwc__max__2x2__s1__output2x2__depthfirst_8hpp.xhtml#ae36a8f494ef3276c9f802ebb47bcee96",
"add_2generic_2neon_2impl_8cpp_source.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a23ab0e5c6b5d13e084628686c4f282d5a2a4c9b03dd6ecd782cdc2174edcaf58d",
"arm__compute_2core_2_validate_8h.xhtml#a4b97ba5512e8deb4a428515bc61c0c7a",
"arm__compute_2graph_2_utils_8h.xhtml#a75f8e464c6b01b0a4771de38669ee9a1",
"arm__conv_2pooling_2kernels_2a64__u8__nhwc__max__2x2__s1__output2x2__depthfirst_2generic_8cpp.xhtml",
"arm__gemm_2kernels_2sve__hybrid__fp32__mla__8x1_v_l_2generic_8cpp.xhtml",
"cge_8h_source.xhtml",
"classarm__compute_1_1_activation_layer_info.xhtml#ac06d9557031b0febb70869efd793cb38",
"classarm__compute_1_1_c_l_cast.xhtml#ab1c78306496883e3167356989bcc77f6",
"classarm__compute_1_1_c_l_dequantization_layer.xhtml",
"classarm__compute_1_1_c_l_g_e_m_m_convolution_layer.xhtml#ae3962c6fb667ff2b1663fe4497ceaa32",
"classarm__compute_1_1_c_l_neg_layer.xhtml",
"classarm__compute_1_1_c_l_reshape_layer.xhtml#ac7d776a643c4f66e97415de1c00669f0",
"classarm__compute_1_1_c_l_symbols.xhtml#aa9752599bfd3a28d9c34cfb59c8feaf3",
"classarm__compute_1_1_detection_output_layer_info.xhtml#a158d49c7c1df3c6c6589b47d3de56cf0",
"classarm__compute_1_1_i_kernel.xhtml#a341b60d15a5e12a5b8f3825194dd3b12",
"classarm__compute_1_1_kernel.xhtml#a1d1e423ba4b26fdf460aa45bbc4457d1",
"classarm__compute_1_1_n_e_cast.xhtml#a0424c2a623ad4e6000e78e62e5366f3a",
"classarm__compute_1_1_n_e_elementwise_squared_diff.xhtml",
"classarm__compute_1_1_n_e_l2_normalize_layer_kernel.xhtml#a8028a2c0f5d9fedccbaa8b10c91fcacc",
"classarm__compute_1_1_n_e_r_o_i_pooling_layer_kernel.xhtml#ac5110dfc692959c2e852b827976e138b",
"classarm__compute_1_1_normalization_layer_info.xhtml#a9f8e7c7833f47804091414a46bef67d6",
"classarm__compute_1_1_sub_tensor_info.xhtml#a5f63b63606dbbbe54474e6e970a6738c",
"classarm__compute_1_1cl__dwc_1_1_i_cl_d_w_c_native_kernel_config.xhtml#a87536cafff1416614eeeade8309df3b7",
"classarm__compute_1_1cpu_1_1_cpu_softmax_generic.xhtml#a74ae0d6e96f38fecd38471431786b870",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_gemm_lowp_quantize_down_int32_scale_kernel.xhtml#af82adac7a80adfdfd03558e1bbc7310c",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_argument_pack.xhtml#af51575e50393ea37d24c67383fd61d09",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_cl_template_pool2d.xhtml",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_gpu_workload_argument.xhtml#a370bcd54ccd7afe8eedb3be0c47bd689",
"classarm__compute_1_1graph_1_1_default_node_visitor.xhtml#a5f7336fecd376c56e48449a647a6d552",
"classarm__compute_1_1graph_1_1_i_node.xhtml#a3fd5d1d4dea9bb355f9baa9b4d44402f",
"classarm__compute_1_1graph_1_1_split_layer_node.xhtml#aa5059ce7798ddeeb40b9f6659b851b8b",
"classarm__compute_1_1graph_1_1frontend_1_1_flatten_layer.xhtml",
"classarm__compute_1_1mlgo_1_1_heuristic_tree.xhtml#a7c4d11e37e7830c1c1488b8983c57d0f",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_convert_fully_connected_weights_kernel.xhtml",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_weights_reshape_kernel.xhtml#ad1a674e8ee046179a5ede04eb4ad2c1d",
"classarm__compute_1_1test_1_1_simple_tensor.xhtml#a057b52c2d0c51f410da5e48f47706c4e",
"classarm__compute_1_1test_1_1framework_1_1_scheduler_clock.xhtml#a501ef37fef441a0a6a8ea8b8e30a7181",
"classarm__compute_1_1utils_1_1_toggle_option.xhtml#aba9622f1f4bc66fcba246f30d4175a4f",
"classarm__conv_1_1depthwise_1_1sme2__fp16__nhwc__3x3__s1__output4x4__mla__depthfirst.xhtml#a8aa1345e82d1d526f7d2f38eae781360",
"classarm__gemm_1_1_gemm_interleaved_pretransposed2d.xhtml#a468e9c50d4decc2ba86f9bd393ba27d6",
"core_2_c_l_2_c_l_helpers_8cpp.xhtml#adc51892eeef112d44a4f9bb0b988c9b9",
"cpu_2kernels_2elementwise__binary_2list_8h.xhtml#a49b16389b2a82a84265a682a75c010d1",
"cpu_2kernels_2select_2list_8h.xhtml#a1178c7489ce4729ff482c5d7b06355e8",
"dir_19295c3848cb9fbcf1155f42ab5752ba.xhtml",
"dir_8b93d571e43f0b90a570a7b1731747a1.xhtml",
"direct__convolution3d_8cl.xhtml#ad0c87478355e0870d5b29497a0854d8f",
"elementwise__binary_2generic_2neon_2integer_8cpp.xhtml#a8109015040dabe0728f37ebf7facf3df",
"ext_8h.xhtml",
"gemm__helpers_8h.xhtml",
"gemm__helpers_8h.xhtml#ae58745045a3b663252c3b200bc378547",
"globals_func_b.xhtml",
"helpers__asymm_8h.xhtml#a5c2adf93c82648a1bec7db33c697a655",
"intrinsics_2add_8h.xhtml#a71dafd2b464c557b576244f9da4d5d93",
"maxunpool_2generic_2neon_2impl_8h.xhtml",
"namespacearm__compute_1_1utils_1_1detail.xhtml",
"neon__scale_8cpp_source.xhtml",
"qlstm__layer__normalization_8cl.xhtml#af4d1bdad50c7cb172247cc7967e40bcd",
"reference_2_g_e_m_m_lowp_8cpp.xhtml#a4b20cb8ea2fd8d04e21d0221ff24e677",
"reference_2_winograd_8cpp.xhtml",
"select_2generic_2neon_2impl_8cpp.xhtml#a75bd92c9d8cb4c889488cf6501301ca2",
"sme2__u8s8u8q__planar__3x3__s1__4rows__dot__za_8hpp_source.xhtml",
"src_2common_2utils_2_utils_8h.xhtml",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a87a9d201ccc6b81332852d562bb7ee13",
"src_2core_2_helpers_8cpp.xhtml#a79ff77f9b4506ad55c680f8849317b9f",
"struct_acl_activation_descriptor.xhtml#a83fc1af92e29717b4513d121b0c72c7d",
"structarm__compute_1_1_g_e_m_m_l_h_s_matrix_info.xhtml#a25a62de4b18dc349803bf5447052d673",
"structarm__compute_1_1cpu_1_1kernels_1_1_cpu_depthwise_conv2d_native_kernel_1_1_depthwise_conv2d_native_kernel.xhtml#a8f8f80d37794cde9472343e4487ba3eb",
"structarm__compute_1_1experimental_1_1dynamic__fusion_1_1_gpu_kernel_argument_info.xhtml#a17016d41439c2e42399cc92d8a95ad1da963c0c0999b3c937f36565fae3b0ef54",
"structarm__compute_1_1test_1_1framework_1_1_test_result.xhtml#a67a0db04d321a74b7e7fcfd3f1a3f70bab9e14d9b2886bcff408b85aefa780419",
"structarm__conv_1_1depthwise_1_1interleaves_1_1_packing_arguments.xhtml#ab38845ad7e40f15a8ad13498d52b1374",
"structgemm__tuner_1_1_common_gemm_example_params.xhtml#af0337b6054d4462adeb69c1b2a5387e6",
"sve__s8q__nhwc__max__generic__depthfirst_8hpp.xhtml",
"tests_2framework_2_utils_8h.xhtml#a1ce487275e4d9e2072af217789dcdcc5",
"tests_2validation_2cpu_2unit_2_tensor_pack_8cpp_source.xhtml",
"tile__helpers_8h.xhtml#a9016ec473d8d6297d715eba783b0cedb",
"utils_2_type_printer_8h.xhtml#a6e422c610c91f611590173cd595c94c5",
"validation_2_n_e_o_n_2_scale_8cpp.xhtml#aeffd5a0d37b930ce0676d973ae2046e1"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';