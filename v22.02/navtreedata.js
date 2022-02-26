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
"_c_l_2_cast_8cpp.xhtml#afd68cdfe92763f60825c3c355b9af9ce",
"_c_l_2_depthwise_convolution_layer_8cpp.xhtml#a0dabc3ee4c06c701e99214462233a594",
"_c_l_2_fuse_batch_normalization_8cpp.xhtml#aad02f34f2471c5d477e44f9d79a2f83b",
"_c_l_2_normalization_layer_8cpp.xhtml#a09929aa316ea25237bfff2f344f05634",
"_c_l_2_reverse_8cpp.xhtml#a512eb649fdb115f2dee5df9f1d156b16",
"_c_l_array_8h.xhtml#a8a401a071524761c661a75969c951cf5",
"_c_l_gather_8cpp_source.xhtml",
"_c_l_strided_slice_8cpp_source.xhtml",
"_channel_shuffle_layer_node_8cpp_source.xhtml",
"_cl_pool2d_kernel_8h.xhtml",
"_cpu_concatenate_8cpp_source.xhtml",
"_cpu_gemm_lowp_quantize_down_int32_to_int8_scale_by_fixed_point_kernel_8cpp_source.xhtml",
"_depthwise_convolution_layer_8h.xhtml#afa25b85fdf3eb7960f4e6febd70909d0",
"_execution_helpers_8cpp.xhtml#aaf7f4e1e2142802dd1c5a983ee156e77",
"_g_e_m_m_matrix_multiply_reshaped_8cpp.xhtml#a2b316619273b1ab85393e1375c8eacae",
"_i_c_l_simple_function_8h.xhtml",
"_legacy_support_8cpp.xhtml",
"_n_e_bitwise_and_8cpp_source.xhtml",
"_n_e_floor_8cpp.xhtml",
"_n_e_o_n_2_arithmetic_subtraction_8cpp.xhtml#a74eed3fcb9c814047858d96c0c1b3159",
"_n_e_o_n_2_deconvolution_layer_8cpp.xhtml#a2b44dc47dc1150bfa937370888d35dac",
"_n_e_o_n_2_elementwise_min_8cpp.xhtml#adb46067dcdcf2d2cb4a4417f49a8bd37",
"_n_e_o_n_2_p_relu_layer_8cpp.xhtml#a3f8cd7b6be1221d0d09ddbadde261042",
"_n_e_o_n_2_softmax_layer_8cpp.xhtml#a84405e2911e417969ee3f5adf020f4b9",
"_n_e_select_kernel_8cpp.xhtml#a054acbc5c4508672a80cd7eb9cb62bb0",
"_open_c_l_8h_source.xhtml",
"_saturate_cast_8h.xhtml",
"_tensor_allocator_8h.xhtml",
"a64__fp16__nhwc__3x3__s1__output2x2__mla__depthfirst_8hpp.xhtml",
"a64__s8__nhwc__max__2x2__s1__output2x2__depthfirst_8hpp_source.xhtml",
"add_2generic_2neon_2qasymm8__signed_8cpp.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a82b8ac759c804bc1fb4e2d21e178fb6fa4729d95f983955f0d93a30179deb2b86",
"arm__compute_2core_2experimental_2_types_8h.xhtml#a08e287b5f0197ce8c7c84dde6db24828aff6d498c21961fa2f6228178fa16b179",
"arm__conv_2depthwise_2kernels_2a64__u8q__nhwc__generic__output9__mla__depthfirst_2generic_8cpp_source.xhtml",
"arm__gemm_2kernels_2sve__hybrid__u8u32__dot__6x4_v_l_2generic_8cpp_source.xhtml",
"cl__gemmlowp__reshaped_8cpp.xhtml#a3c04138a5bfe5d72780bb7e82a18e627",
"classarm__compute_1_1_bounding_box_transform_info.xhtml#a491fddb771387738421f6d8e2b05f17f",
"classarm__compute_1_1_c_l_compile_context.xhtml#a86409f541991c3ac377c2a6b7d1f832d",
"classarm__compute_1_1_c_l_elementwise_min.xhtml#a0e3ec9eefda31243fc6f594e4d2771c9",
"classarm__compute_1_1_c_l_gather_kernel.xhtml#a93babbb8c4fde8b36d6b2627d36bd153",
"classarm__compute_1_1_c_l_pad_layer_kernel.xhtml#af8b7f2369f7d54396f1dedb06bb7f2af",
"classarm__compute_1_1_c_l_select.xhtml#afb3c068b9a3cd4c394055bd28a3df6f8",
"classarm__compute_1_1_c_l_tuner.xhtml#a7993b367bfe0275459eafb09059da1d8",
"classarm__compute_1_1_generate_proposals_info.xhtml#aa7d375782a8b80e3d4efee296a059855",
"classarm__compute_1_1_i_scheduler.xhtml#a64cafb079598059f1ee04f3076f1035e",
"classarm__compute_1_1_n_e_arg_min_max_layer.xhtml#ab25ca78794a2d3c7e924e102a6c8b832",
"classarm__compute_1_1_n_e_crop_resize.xhtml#aeba0f6c60cf9ecafb60928b6c6b4c547",
"classarm__compute_1_1_n_e_fully_connected_layer.xhtml#a9845f83f6f3bc45f9fb57ea1345e3dd3",
"classarm__compute_1_1_n_e_pad_layer.xhtml#abe00a9a7d28550ec86c73fa8f015c085",
"classarm__compute_1_1_n_e_space_to_batch_layer.xhtml#ae38c9a24129778ff49c13ddd144471de",
"classarm__compute_1_1_quantization_info.xhtml#aa0075a0b6df3da4c9993922e58a304d0",
"classarm__compute_1_1_tensor_shape.xhtml#a0fdcb923dfd4c74858cc2bc326321efb",
"classarm__compute_1_1cpu_1_1_cpu_mul.xhtml#a684a54d1fb1634a348a585c6b5e76df0",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_division_kernel.xhtml",
"classarm__compute_1_1experimental_1_1_n_e_strided_slice.xhtml#aee85a5053abeb2b5ab2d9ae72aaa74e3",
"classarm__compute_1_1graph_1_1_eltwise_layer_node.xhtml#af7af5fb47f0648c30118689c9b1e022c",
"classarm__compute_1_1graph_1_1_l2_normalize_layer_node.xhtml#a65d13dc93e2df5e8ab725263cf9f4ac5",
"classarm__compute_1_1graph_1_1backends_1_1_c_l_sub_tensor_handle.xhtml#adfcd04c831f29841dd197ffe95dd091b",
"classarm__compute_1_1graph_1_1frontend_1_1_stream.xhtml#a4be4b83054829b12ec645bb454e6b9cc",
"classarm__compute_1_1opencl_1_1_cl_elementwise_min.xhtml#a27c6792cf43071f3153769112bd0f8f9",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_gemm_matrix_multiply_reshaped_only_rhs_kernel.xhtml#a2563bcc3dcf9406cbf572d206a99d801",
"classarm__compute_1_1test_1_1_c_l_array_accessor.xhtml#ad0aeb1c3b2795a2008703a500d05f7f1",
"classarm__compute_1_1test_1_1framework_1_1_mali_counter.xhtml#ab3536e22848ce87b16a9b96d6d824d45",
"classarm__compute_1_1utils_1_1_enum_option.xhtml",
"classarm__conv_1_1pooling_1_1_pooling_depthfirst_generic.xhtml#ac54b55daf0f672146144be1e4239ea5c",
"clt_8h.xhtml#a28724874f41631f5dacaacc8fc145e66",
"cpu_2kernels_2elementwise__binary_2generic_2neon_2fp16_8cpp_source.xhtml",
"cpu_2kernels_2sub_2neon_2list_8h.xhtml#acee71620d39960df76ed85d99151f255",
"dir_5fcd67f9a14af02cd68579241d75cb4c.xhtml",
"direct__convolution3d_8cl.xhtml#ac06b8cd53a4a658f0cc88becd6e64208",
"ext_8h.xhtml#a75f2ece30fddc460ded2a3af6b871df4",
"gemm__helpers_8h.xhtml#a2c52a5415bb1333c04353b2f191fc5b5",
"gemm__hybrid__quantized_8hpp.xhtml",
"graph__depthwiseconvolution_8cpp.xhtml#a3c04138a5bfe5d72780bb7e82a18e627",
"helpers__asymm_8h.xhtml#af06991bb67792a6fec9c426923f39745",
"load__store__utility_8h.xhtml#a270acb3e45f182ebb6526f3e85eadd14",
"namespacearm__compute_1_1helpers_1_1float__ops.xhtml",
"nchw_2scale_8cl.xhtml#a42131df7a754ea279bea023c2e18b077",
"pooling__s8_8cpp.xhtml#a6d48fc2e894ccf10d2a3894f412f3ed0",
"reference_2_dequantization_layer_8cpp.xhtml#afb55c366bbc0df5025d95b736ac14ff4",
"reference_2_transpose_8cpp.xhtml#a9f3ed5e09dcdb43d381a70aba1700bd4",
"select_2generic_2neon_2impl_8cpp.xhtml#a22f928654f8c222e3fe675e59b94e54b",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a3a8c18725a03fd3c89626a8e06d1e5b4",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#ac1a4253aef5720be080de0282be96e2f",
"src_2gpu_2cl_2operators_2_c_l_g_e_m_m_lowp_output_stage_8h.xhtml",
"structarm__compute_1_1_c_l_quantization.xhtml#af7e03963753b604b47aa20603c0cffe5",
"structarm__compute_1_1cl__gemm_1_1auto__heuristics_1_1_common_query.xhtml#a6118865e7371b107bd5fb4dd4a9e99a3",
"structarm__compute_1_1graph_1_1_memory_manager_context.xhtml#a2a7ca82c5e74421cb45f17e936abf964",
"structarm__compute_1_1test_1_1traits_1_1promote_3_01int16__t_01_4.xhtml#a4f14ffc6d6f6ca92afb1baf996fc9305",
"structarm__conv_1_1pooling_1_1_pooling_implementation.xhtml#a2da787fc1e7d185135bbc0ca070cd8b8",
"sve__fp32__nhwc__3x3__s1__output2x2__mla__depthfirst__strided_8hpp_source.xhtml",
"tensor__transform_8cpp.xhtml#a396d80389277ad8cd13a5e0567776191",
"tests_2validation_2_c_l_2fft_8cpp.xhtml#ab43e1ef278ffc6fc35b71b5c246c9f36",
"tile__helpers_8h.xhtml#a17fa80fa2db3c0ae40e56b59c470d354",
"utils_2_type_printer_8h.xhtml#a1f3b9f93958bbf939d6889d96c0fc043",
"validation_2_n_e_o_n_2_scale_8cpp.xhtml#a2e6db4015404fa3fbdb74beca9bf03d0"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';