/*
@ @licstart  The following is the entire license notice for the
JavaScript code in this file.

Copyright (C) 1997-2017 by Dimitri van Heesch

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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
          [ "How to manually build the examples ?", "index.xhtml#S3_2_2_examples", null ]
        ] ],
        [ "Building for Android", "index.xhtml#S3_3_android", [
          [ "How to build the library ?", "index.xhtml#S3_3_1_library", null ],
          [ "How to manually build the examples ?", "index.xhtml#S3_3_2_examples", null ]
        ] ],
        [ "Building for bare metal", "index.xhtml#S3_4_bare_metal", [
          [ "How to build the library ?", "index.xhtml#S3_4_1_library", null ],
          [ "How to manually build the examples ?", "index.xhtml#S3_4_2_examples", null ]
        ] ],
        [ "Building on a Windows host system", "index.xhtml#S3_5_windows_host", [
          [ "Bash on Ubuntu on Windows", "index.xhtml#S3_5_1_ubuntu_on_windows", null ],
          [ "Cygwin", "index.xhtml#S3_5_2_cygwin", null ]
        ] ],
        [ "OpenCL DDK Requirements", "index.xhtml#S3_6_cl_requirements", [
          [ "Hard Requirements", "index.xhtml#S3_6_1_cl_hard_requirements", null ],
          [ "Performance improvements", "index.xhtml#S3_6_2_cl_performance_requirements", null ]
        ] ],
        [ "OpenCL Tuner", "index.xhtml#S3_7_cl_tuner", [
          [ "How to use it", "index.xhtml#S3_7_1_cl_tuner_how_to", null ]
        ] ]
      ] ]
    ] ],
    [ "Library architecture", "architecture.xhtml", [
      [ "Core vs Runtime libraries", "architecture.xhtml#S4_1_1", null ],
      [ "Thread-safety", "architecture.xhtml#S4_1_2", null ],
      [ "Windows, kernels, multi-threading and functions", "architecture.xhtml#S4_2_windows_kernels_mt_functions", [
        [ "Windows", "architecture.xhtml#S4_2_1_windows", null ],
        [ "Kernels", "architecture.xhtml#S4_2_2", null ],
        [ "Multi-threading", "architecture.xhtml#S4_2_3", null ],
        [ "Functions", "architecture.xhtml#S4_2_4", null ],
        [ "OpenCL Scheduler and kernel library", "architecture.xhtml#S4_4_1_cl_scheduler", null ],
        [ "OpenCL events and synchronization", "architecture.xhtml#S4_4_2_events_sync", null ],
        [ "OpenCL / NEON interoperability", "architecture.xhtml#S4_4_2_cl_neon", null ]
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
        [ "Directory structure", "tests.xhtml#tests_overview_structure", null ],
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
        ] ]
      ] ],
      [ "How to submit a patch", "contribution_guidelines.xhtml#S5_2_how_to_submit_a_patch", null ],
      [ "Patch acceptance and code review", "contribution_guidelines.xhtml#S5_3_code_review", null ]
    ] ],
    [ "List of functions", "functions_list.xhtml", [
      [ "NEON functions", "functions_list.xhtml#S6_1", null ],
      [ "OpenCL functions", "functions_list.xhtml#S6_2", null ],
      [ "GLES Compute functions", "functions_list.xhtml#S6_3", null ],
      [ "CPP functions", "functions_list.xhtml#S6_4", null ]
    ] ],
    [ "Errata", "errata.xhtml", [
      [ "Errata", "errata.xhtml#S7_1_errata", null ]
    ] ],
    [ "Gemm Tuner", "md_examples_gemm_tuner__r_e_a_d_m_e.xhtml", null ],
    [ "Namespaces", "namespaces.xhtml", [
      [ "Namespace List", "namespaces.xhtml", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.xhtml", [
        [ "All", "namespacemembers.xhtml", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.xhtml", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.xhtml", "namespacemembers_vars" ],
        [ "Typedefs", "namespacemembers_type.xhtml", "namespacemembers_type" ],
        [ "Enumerations", "namespacemembers_enum.xhtml", null ]
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
        [ "Macros", "globals_defs.xhtml", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_abs_layer_8cpp.xhtml",
"_c_l_2_arg_min_max_8cpp.xhtml#a989dc9a1cdb2b8f8f0fe805278155e88",
"_c_l_2_channel_combine_8cpp.xhtml#a8e77ed62c2a00d8d5107bd0c0f8aeb91",
"_c_l_2_depth_convert_layer_8cpp.xhtml#a4d04b987d8740c922911e59e99dd6e6a",
"_c_l_2_fill_8cpp.xhtml#af6c649617ab823f51e329840320def77",
"_c_l_2_l2_normalize_layer_8cpp.xhtml#a04af0f9ee366f68a8e53d5e7128a3930",
"_c_l_2_pixel_wise_multiplication_8cpp.xhtml#a65ddaf835fb36c6b21dd8a53d0c99451",
"_c_l_2_space_to_batch_layer_8cpp.xhtml#a2c4738de69048aebf8954d231f66a4d0",
"_c_l_2_winograd_8cpp.xhtml#af94bf82577407bd38165f4e84c4c360a",
"_c_l_core_runtime_context_8cpp_source.xhtml",
"_c_l_g_e_m_m_kernel_selection_valhall_8cpp_source.xhtml",
"_c_l_l2_normalize_layer_kernel_8h_source.xhtml",
"_c_l_range_kernel_8cpp.xhtml",
"_c_l_warp_perspective_8cpp.xhtml",
"_channel_shuffle_8h.xhtml",
"_dummy_node_8cpp.xhtml",
"_file_printer_8cpp_source.xhtml",
"_g_c_memory_8cpp.xhtml",
"_g_l_e_s___c_o_m_p_u_t_e_2_convolution_layer_8cpp.xhtml#aa9dbfc3b5de017a7a44b18d0fdf973e2",
"_i_accessor_8h.xhtml",
"_image_loader_8h.xhtml",
"_n_e_arithmetic_subtraction_kernel_8cpp_source.xhtml",
"_n_e_depth_concatenate_layer_kernel_8h_source.xhtml",
"_n_e_g_e_m_m_assembly_base_kernel_8h_source.xhtml",
"_n_e_locally_connected_matrix_multiply_kernel_8cpp_source.xhtml",
"_n_e_o_n_2_batch_to_space_layer_8cpp.xhtml#a62b00368dacbe493692481b8527f3520",
"_n_e_o_n_2_convolution_layer_8cpp.xhtml#acbf90c23b4569e5994cf16898a254078",
"_n_e_o_n_2_dilated_convolution_layer_8cpp.xhtml#a6813132c943295888972727864ea5c2f",
"_n_e_o_n_2_height_concatenate_layer_8cpp.xhtml#a2b012a0b9c5875a28c6ee20d7fb8f9e8",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml#a45a9fdf5736081ef4cada3c6a13fefd1",
"_n_e_o_n_2_stack_layer_8cpp.xhtml",
"_n_e_reduction_operation_kernel_8cpp_source.xhtml",
"_n_e_y_o_l_o_layer_kernel_8cpp_source.xhtml",
"_open_g_l_e_s_8cpp.xhtml#af9416600d973cf090c41f584c874dd9d",
"_schaar_8cpp.xhtml",
"_toolchain_support_8h.xhtml#a73e352c61baaf9c1178da2d30105b04e",
"_version_8cpp_source.xhtml",
"abs_8h.xhtml#ad8275425767f39b246862f11cef3328d",
"arm__compute_2core_2_types_8h.xhtml#a683661ae75dcb7aef16b9c9bde31517d",
"arm__compute_2graph_2_types_8h.xhtml#acac9cbaeea226ed297804c012dc12b16aab5cbd6c92ca9262ddd46d573bd2d2a5",
"class_gemm_tuner_1_1_g_e_m_m_benchmark_result_recorder.xhtml",
"classarm__compute_1_1_c_l_array.xhtml#aef6598a7ef623f1d4cb051d85176997b",
"classarm__compute_1_1_c_l_compile_context.xhtml#a3dd891a07efcd9a5ae685f834973666d",
"classarm__compute_1_1_c_l_depthwise_convolution_layer_native_kernel.xhtml#a838d7458ec7f5048d2304360de0095b2",
"classarm__compute_1_1_c_l_flatten_layer_kernel.xhtml#a199665cd64ca9f0d9b4bd9d48d59fbdc",
"classarm__compute_1_1_c_l_g_e_m_m_matrix_multiply_reshaped_kernel.xhtml#a5cbfc1be8e84436d97e5ba33b80987a3",
"classarm__compute_1_1_c_l_im2_col_kernel.xhtml#a118b70265d99428bf7d068d686d0cb80",
"classarm__compute_1_1_c_l_mean_std_dev_normalization_kernel.xhtml#a6776dab70234938527ed26acd8c6e33b",
"classarm__compute_1_1_c_l_q_l_s_t_m_layer_normalization_kernel.xhtml#ab24719b9763a7ac51210aab47e7c40c1",
"classarm__compute_1_1_c_l_sobel5x5_hor_kernel.xhtml#a0d2de1f5a7010147dc1d6c11eaaeda37",
"classarm__compute_1_1_c_l_transpose.xhtml#aa047ad145604827aa3f55253664ed61b",
"classarm__compute_1_1_c_p_p_top_k_v_kernel.xhtml",
"classarm__compute_1_1_g_c_dropout_layer_kernel.xhtml#a03febcf2730ff389e09fc47672e4fac7",
"classarm__compute_1_1_g_c_tensor.xhtml#ac4a7cca5d7e6a0028379b9fa874ce338",
"classarm__compute_1_1_i_c_l_simple_function.xhtml#ab1b50b6d8c633c9f858260b5960747b7",
"classarm__compute_1_1_i_n_e_winograd_layer_transform_output_kernel.xhtml#a2795ccfd548a0a3f8f680e2c2a878e33",
"classarm__compute_1_1_memory_region.xhtml#a32d4629fe7aa6093bbe54599319d08fa",
"classarm__compute_1_1_n_e_channel_shuffle_layer.xhtml#a2a36146a04397f2f7e22c940178894f8",
"classarm__compute_1_1_n_e_depthwise_convolution_layer3x3_kernel.xhtml#a423f9a45a52983b4de5e2b347f4369c7",
"classarm__compute_1_1_n_e_fill_inner_border_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_gather_kernel.xhtml#ac54444a80c4fc9141ca52479ea7650a2",
"classarm__compute_1_1_n_e_locally_connected_layer.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_n_e_q_l_s_t_m_layer_normalization_kernel.xhtml",
"classarm__compute_1_1_n_e_sobel7x7_hor_kernel.xhtml#ad4bba8f7619ade93cc4220dbcc72c35a",
"classarm__compute_1_1_normalization_layer_info.xhtml#a9f8e7c7833f47804091414a46bef67d6",
"classarm__compute_1_1_sub_tensor_info.xhtml#a5f1ca9d674346287cae57a6c5b5c24ec",
"classarm__compute_1_1cl__tuner_1_1_c_l_l_w_s_list_normal.xhtml#a32e58c10b8f30e8dd7b10980dbee2d11",
"classarm__compute_1_1graph_1_1_graph.xhtml#a538f789bf074c367457a6f8f32b83d2d",
"classarm__compute_1_1graph_1_1_reorg_layer_node.xhtml#a47d010db0ab9940009209db7cf529f36",
"classarm__compute_1_1graph_1_1backends_1_1_n_e_tensor_handle.xhtml#a7ab0a61cbf616378316288ffde1b7e04",
"classarm__compute_1_1logging_1_1_logger.xhtml#a75f49292d95d30b860d000005cf179a1",
"classarm__compute_1_1test_1_1_raw_lut_accessor.xhtml#aedcfdd4c3b92fe0d63b5463c7ad1d21e",
"classarm__compute_1_1test_1_1framework_1_1_printer.xhtml#adac997f56174b23d4e4ec5adccf1d836",
"classarm__compute_1_1utils_1_1_option.xhtml#a564afafa4220cd36afbe388fcd24549b",
"classarm__gemm_1_1_gemv_pretransposed.xhtml#ac5dec5063a75604fdc60555f15577b94",
"core_2_g_l_e_s___c_o_m_p_u_t_e_2_g_c_helpers_8cpp_source.xhtml",
"direct__convolution3x3_8cl.xhtml#a1f15728672380ade7a238f5e783d54d2",
"functions_vars_h.xhtml",
"gemm__hybrid__quantized_8hpp_source.xhtml",
"graph__squeezenet__v1__1_8cpp.xhtml#a3c04138a5bfe5d72780bb7e82a18e627",
"intrinsics_2add_8h.xhtml#a71dafd2b464c557b576244f9da4d5d93",
"namespacearm__compute_1_1logging.xhtml",
"normalization__layer_8cl_source.xhtml",
"reference_2_comparisons_8cpp.xhtml#ad9bfac7b9e4c2ff4299e279ae829791f",
"reference_2_magnitude_8cpp.xhtml",
"reference_2_y_o_l_o_layer_8cpp.xhtml#a3931061e2eae2c2a5caa8807cee9e517",
"softmax__layer__quantized_8cl.xhtml#a44206a4e5783c7aabacec88aad878c88",
"struct_image.xhtml",
"structarm__compute_1_1_optical_flow_parameters.xhtml#aca4f010655c8e5534162dfbfdce31d84",
"structarm__compute_1_1test_1_1framework_1_1_test_result.xhtml#a67a0db04d321a74b7e7fcfd3f1a3f70babdb2c34681be3432a08e0450b707a014",
"sve__hybrid__bf16fp32__mmla__4_v_lx4_2generic_8cpp_source.xhtml",
"tests_2framework_2_macros_8h.xhtml#acdd2ac75560db81371f4053e6465a0eb",
"tests_2validation_2reference_2_utils_8h.xhtml#a3bb27cb73ac0ecd83cbb28d01b5edc2c",
"validation_2_g_l_e_s___c_o_m_p_u_t_e_2_scale_8cpp.xhtml"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';