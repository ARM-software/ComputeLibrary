/*
@licstart  The following is the entire license notice for the
JavaScript code in this file.

Copyright (C) 1997-2019 by Dimitri van Heesch

This program is free software; you can redistribute it and/or modify
it under the terms of version 2 of the GNU General Public License as published by
the Free Software Foundation

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

@licend  The above is the entire license notice
for the JavaScript code in this file
*/
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
      [ "Compute Library architecture", "architecture.xhtml#architecture_compute_library", null ],
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
    [ "Convolution 2D heuristic", "conv2d_heuristic.xhtml", [
      [ "Convolution 2D heuristic: algorithm selection", "conv2d_heuristic.xhtml#conv2d_heuristic_algorithms_used", [
        [ "Convolution 2D heuristic: Arm® Cortex®-based CPUs", "conv2d_heuristic.xhtml#conv2d_heuristic_on_cpu", null ],
        [ "Convolution 2D heuristic: Arm® Mali™-based GPUs", "conv2d_heuristic.xhtml#conv2d_heuristic_on_gpu", null ]
      ] ]
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
    [ "Gemm Tuner", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml", [
      [ "Introduction", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md2", null ],
      [ "Step-by-step example", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md3", [
        [ "Step1: Prepare the shape and configs files", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md4", null ],
        [ "Step2: Push relevant files to the target device", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md5", null ],
        [ "Step3: Collect benchmark data", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md6", null ]
      ] ],
      [ "Reshaped benchmark on 3 different platforms", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md7", [
        [ "Platform 1", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md8", null ],
        [ "Platform 2", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md9", null ],
        [ "Platform 3", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md10", [
          [ "Step4: Generate the heuristics", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md11", null ]
        ] ],
        [ "Prerequisite", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md12", null ],
        [ "Usage", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml#autotoc_md13", null ]
      ] ]
    ] ],
    [ "Deprecated List", "deprecated.xhtml", null ],
    [ "Namespaces", "namespaces.xhtml", [
      [ "Namespace List", "namespaces.xhtml", "namespaces_dup" ],
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
    [ "Files", "files.xhtml", [
      [ "File List", "files.xhtml", "files_dup" ],
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
"_abs_layer_8cpp.xhtml",
"_acl_types_8h.xhtml#adc3017359ff84d03850580266cb18256",
"_c_l_2_activation_layer_8cpp.xhtml#a146990bdb7bc1df6e90e6dca4eaa764c",
"_c_l_2_cast_8cpp.xhtml#a77ce12a0a1ad23db0b51dfb721b15495",
"_c_l_2_deconvolution_layer_8cpp.xhtml#a7a11a3a9dc989784e3b77d3e0ab77439",
"_c_l_2_direct_convolution_layer_8cpp.xhtml#af44c067127e8c4062e9f2ec24c4b1098",
"_c_l_2_im2_col_8cpp.xhtml#aa191b6260447d3d5cd069993a375fa17",
"_c_l_2_pooling3d_layer_8cpp.xhtml#a3e736cd458ec14287e25bcea4434782a",
"_c_l_2_space_to_depth_layer_8cpp.xhtml#ad0c23b0170c87fec4d92c28e1dc41c3e",
"_c_l_comparison_kernel_8h.xhtml",
"_c_l_memory_region_8cpp.xhtml",
"_c_l_utils_8h.xhtml#a41a1889b2fdf01056017b9f8b58dda6ea53dfe5089ae9e65887cb0215cb0098f5",
"_cl_component_store_8h.xhtml",
"_cl_indirect_conv2d_kernel_8h_source.xhtml",
"_color_convert_helper_8h.xhtml#ab7d52900468f442e85c01d73722d181c",
"_cpu_add_mul_add_kernel_8cpp.xhtml",
"_cpu_gemm_lowp_quantize_down_int32_to_uint8_scale_by_fixed_point_kernel_8cpp.xhtml",
"_data_type_utils_8h.xhtml#ae69217acf0f0b5d4de030a09ad50a0bc",
"_elementwise_round_8cpp.xhtml#a703688d7ea06a4c7d746993fd46e6090",
"_function_helpers_8h.xhtml#a0e51b62035e79b0f12964cae17ce0480",
"_g_e_m_m_matrix_multiply_reshaped_only_r_h_s_8cpp.xhtml#a6ac14a0f9fc0f3c9d2af7469826915c4",
"_gpu_resize_8h_source.xhtml",
"_i_post_op_8h.xhtml#a3411c74a9d92ff2e88330fce49a425eaa56ce6a9123158f9f5836a8d8f854bb21",
"_m_l_g_o_parser_8cpp.xhtml#a8f6e0cacae0c93845d42effbaa6a4ef4",
"_n_e_asymm_8h.xhtml#a3109788e20d3a6113e4b7d2d937c1f64",
"_n_e_elementwise_unary_layer_8cpp.xhtml",
"_n_e_o_n_2_activation_layer_8cpp.xhtml#a5812a14fb9d2b782178a7f87250adcd5",
"_n_e_o_n_2_convolution3_d_8cpp_source.xhtml",
"_n_e_o_n_2_depthwise_convolution_layer_8cpp.xhtml#ac75707a8adda5011b4a25d628280f9c0",
"_n_e_o_n_2_im2_col_8cpp.xhtml#a548b2da4872df2c11721b7068cab5b8a",
"_n_e_o_n_2_pooling_layer_8cpp.xhtml#a80bdb9a0c10fc454a2e8ff1afeb120f6",
"_n_e_o_n_2_u_n_i_t_2_dynamic_tensor_8cpp.xhtml#a1e59c3c6e6b81537d6098183dcb44a67",
"_n_e_unstack_8h_source.xhtml",
"_pool_manager_8cpp.xhtml",
"_saturate_cast_8h.xhtml#aff2fdac1ef048c4a668437d172fa6b1a",
"_string_support_8h.xhtml#a77af73f4abd7eeca01501fa3828f725a",
"_version_8h.xhtml",
"a64__hybrid__s8s32__dot__6x16_2a55_8cpp_source.xhtml",
"a64__u8s8u8q__packed__to__nhwc__generic__with__multiplier__output2x8__mla__depthfirst_8hpp_source.xhtml",
"architecture.xhtml#programming_model",
"arm__compute_2core_2_validate_8h.xhtml#ae7eed178dac535c6e727061b1f5bc6eb",
"arm__compute_2runtime_2_c_l_2functions_2_c_l_logical_not_8h.xhtml",
"arm__conv_2pooling_2kernels_2sve__fp16__nhwc__avg__3x3__s1__output2x2__depthfirst_2generic_8cpp.xhtml",
"arm__gemm_8hpp.xhtml#a23ab0e5c6b5d13e084628686c4f282d5a06eeee52deca91c5f2d378c409143626",
"cl__gemmlowp__reshaped_8cpp_source.xhtml",
"classarm__compute_1_1_box_n_m_s_limit_info.xhtml#a5d68f71fde6fb8401332cbe3973368a2",
"classarm__compute_1_1_c_l_compile_context.xhtml#ae712947a608a9fbcbfb4c2d43a56da58",
"classarm__compute_1_1_c_l_elementwise_min.xhtml#a60766816045864d8fbb0a9e63dac06b8",
"classarm__compute_1_1_c_l_generate_proposals_layer.xhtml",
"classarm__compute_1_1_c_l_pad_layer.xhtml#a4474e177ef7d9d37019486f447b8f051",
"classarm__compute_1_1_c_l_scale.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_c_l_tile.xhtml#a96dec2794e43968411736c28f192a4aa",
"classarm__compute_1_1_g_e_m_m_info.xhtml#a11d8f855e323a8396fe6944edcef4238",
"classarm__compute_1_1_i_n_e_simple_function.xhtml#a745211a32b7afb5725f7d9a475816288",
"classarm__compute_1_1_memory.xhtml#a190be71167f70cb0c6b7a53504c8c189",
"classarm__compute_1_1_n_e_concatenate_layer.xhtml#a53cc12aaa9ea3a3080438cf5fd2014e5",
"classarm__compute_1_1_n_e_f_f_t_digit_reverse_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_mat_mul.xhtml#a5e6bab93b7a67fb041db43ef7d9a9abb",
"classarm__compute_1_1_n_e_reorg_layer_kernel.xhtml#a48da94ae894674d8009281d0bda17f62",
"classarm__compute_1_1_pixel_value.xhtml#a43fd30dcf0471d3194662a798cab5b07",
"classarm__compute_1_1_tensor_allocator.xhtml#a1468b0adb6ec3f9d38aa7d60b8a91974",
"classarm__compute_1_1cpu_1_1_cpu_add.xhtml",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_activation_kernel.xhtml#a883429dd6cf828bfdd64b255afc458da",
"classarm__compute_1_1cpu_1_1kernels_1_1_cpu_logits1_d_softmax_kernel.xhtml#a821291f03339db38362cc00e83208e79",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_cl_component_direct_conv2d.xhtml#a16e2118618778baf41d73ac332567db4",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_cl_template_store.xhtml#ad71b92a129e0deeaa8365c4e5f1c9c73",
"classarm__compute_1_1experimental_1_1dynamic__fusion_1_1_gpu_pool2d_settings.xhtml#a78dfef5ca7b638549a2c137c3164c616",
"classarm__compute_1_1graph_1_1_conv_post_op_info_activation.xhtml#abe3f4fb4f063604571dc8906dab9c520",
"classarm__compute_1_1graph_1_1_graph_context.xhtml#a52ee8d390e25147e776bfcb39b010947",
"classarm__compute_1_1graph_1_1_reduction_layer_node.xhtml#adb48b5745c55605a2d4ec6f665bb7f3f",
"classarm__compute_1_1graph_1_1frontend_1_1_batch_normalization_layer.xhtml#a225fb162b43cbef27d5a1dbd7a51c60d",
"classarm__compute_1_1logging_1_1_logger.xhtml#a421744d0932b362cbff55c07488c1356",
"classarm__compute_1_1opencl_1_1_cl_transposed_convolution.xhtml#adccd10e6a9130dcf71a87a57445efb89",
"classarm__compute_1_1opencl_1_1kernels_1_1_cl_permute_kernel.xhtml#a3705812f8d718586e36f2972b3772f77",
"classarm__compute_1_1test_1_1_i_accessor.xhtml#aa983ddaeded5756189f46428be79c631",
"classarm__compute_1_1test_1_1framework_1_1_p_m_u_counter.xhtml#a73cabccc9d4406bf48f1db033dc35005",
"classarm__compute_1_1utils_1_1_i_image_loader.xhtml#a25c255918a7306769a9e3886166af935",
"classarm__conv_1_1depthwise_1_1_generic_depthfirst_multiplier_kernel_strategy.xhtml#a3fcdbc0eeb82102e92f066c1b65473ab",
"classarm__gemm_1_1_gemm_hybrid_quantized.xhtml#a8f983ca114ef72d25515a8651ea1d0e7",
"conv2d_heuristic.xhtml#conv2d_heuristic_algorithms_used",
"cpu_2kernels_2elementwise__binary_2generic_2neon_2fp32_8cpp.xhtml#a000160c59f0735bda44082db86c5d415",
"cpu_2kernels_2roialign_2list_8h.xhtml#a334bfe81953ded3a839cc8918fc9d84b",
"dir_01cd49ee06e3027dd8f33ba76ebb4c81.xhtml",
"dir_759ad130c4d2ade5bdd75789c156321e.xhtml",
"dir_eaa5e6f503e04e7a59bfa7670ffbd507.xhtml",
"elementwise__binary_2generic_2neon_2impl_8h.xhtml#a1d55687f004c1a3302485b2cb8da093b",
"elementwise__unary_2generic_2neon_2impl_8cpp.xhtml#ad4fe05571f88fca394e13bfa99280135",
"functions_~.xhtml",
"gemm__helpers_8h.xhtml#ac129cf1abd938a824dd0bfdfc88b1270",
"globals.xhtml",
"helpers__asymm_8h.xhtml#a2d9cac9c87356624d52b00c2e6950ec9",
"interleave__indirect__impl_8hpp.xhtml#af49ac1de6bfb52f17065e939891cb511",
"load__store__utility_8h.xhtml#ae07d54f575d6a2f3139e03aacbb7ba65",
"namespacearm__compute_1_1mlgo.xhtml",
"nchw_2space__to__depth_8cl.xhtml",
"pooling__fp32_8cpp.xhtml#a2dbb9d67088387c8cb61b8d7d063afd1",
"reference_2_depth_convert_layer_8cpp.xhtml#a97ffa19c49dbe0b8d0e47b0ab15c51dd",
"reference_2_select_8cpp_source.xhtml",
"runtime_2_c_l_2mlgo_2_common_8h.xhtml#abf73f846abd6dfb414f6b51f8286ba09a1149b156e352294778e3596869f2f799",
"sme2__interleaved__nomerge__fp16fp32fp16__mopa__1_v_lx4_v_l_8hpp.xhtml",
"softmax__layer__quantized_8cl.xhtml",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a6561bc663cadc3d42b431fec7466add7",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#af6b3c4ee10fd382103ab085f26f66dff",
"src_2runtime_2_c_p_p_2_c_p_p_scheduler_8cpp.xhtml#a564b46e9815aaa6b0570d9d4ff1533ed",
"structarm__compute_1_1_direct_convolution_layer_output_stage_kernel_info.xhtml#aa3136196619ea94a6bbb98f377524e97",
"structarm__compute_1_1cpu_1_1_act_fp_impl_params.xhtml#a1ee4a05b54419bfadd66ee351d05812e",
"structarm__compute_1_1experimental_1_1_post_op_eltwise_add.xhtml#abe3f4fb4f063604571dc8906dab9c520",
"structarm__compute_1_1test_1_1common__promoted__signed__type.xhtml#aeab553c4422f2a498954db0b4eb15789",
"structarm__conv_1_1depthwise_1_1_depthwise_implementation.xhtml#a653bb8ceb55aa22e0c5963de011fae2e",
"support_2_rounding_8h_source.xhtml",
"sve__transpose__interleave__6_v_l__4x2_8hpp.xhtml",
"tests_2validation_2_c_l_2_fully_connected_layer_8cpp.xhtml#a69e759559510a9150f2fd8b8d59b635a",
"tests_2validation_2reference_2_reshape_layer_8cpp.xhtml#afea45dc2b397b75108739c4300aa18ad",
"tile__helpers_8h.xhtml#ab14939f7adf7ba956501983d626519b8",
"utils_2_type_printer_8h.xhtml#a96b47511b549b48d2ead05b5c757ccc9",
"working__space_8hpp.xhtml#a790736ad1b8b128e7fc4ba58334af4d2"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';