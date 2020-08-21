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
        [ "Macros", "globals_defs.xhtml", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_abs_layer_8cpp.xhtml",
"_c_l_2_arg_min_max_8cpp.xhtml#a2c0eb4c5a56da85631ef9ec732312303",
"_c_l_2_channel_combine_8cpp.xhtml",
"_c_l_2_depth_convert_layer_8cpp.xhtml",
"_c_l_2_fill_8cpp.xhtml#a7bd401bf5ea1dbfeb8817f09fa224b83",
"_c_l_2_instance_normalization_layer_8cpp.xhtml#a2c8cf066c5c57ab026598ccc0ec55a7b",
"_c_l_2_permute_8cpp_source.xhtml",
"_c_l_2_sobel_8cpp_source.xhtml",
"_c_l_2_winograd_8cpp.xhtml#aa3d3f69e3800e4033c4442fbc87bcd21",
"_c_l_convolution_kernel_8h.xhtml#a4751499f1c526c7285e7e946c2e84541",
"_c_l_g_e_m_m_convolution_layer_8h.xhtml",
"_c_l_instance_normalization_layer_kernel_8h.xhtml",
"_c_l_q_l_s_t_m_layer_normalization_kernel_8h_source.xhtml",
"_c_l_tuner_types_8h.xhtml#ab567edaca959e3f5013abf442b3235a2",
"_c_p_p_types_8h.xhtml#aa41d7415a5386798147cccae2333d5d4aaf79f683ceaea4650b20e898d6c4c373",
"_depthwise_separable_convolution_layer_8h.xhtml#a79f9dfec3d4a3a6dfe7d084ee6b2b32e",
"_exceptions_8h_source.xhtml",
"_g_c_function_factory_8h_source.xhtml",
"_g_e_m_m_matrix_multiply_reshaped_8cpp.xhtml#a1a5d16d40c0be4ffa124fa5e8be70557",
"_graph_manager_8cpp_source.xhtml",
"_i_memory_group_8h.xhtml",
"_mean_std_dev_8h_source.xhtml",
"_n_e_color_convert_8h.xhtml",
"_n_e_erode_kernel_8cpp.xhtml",
"_n_e_h_o_g_detector_8h_source.xhtml",
"_n_e_o_n_2_accumulate_8cpp.xhtml#accab7177e8385e72312ca47c4130ab18",
"_n_e_o_n_2_color_convert_8cpp.xhtml#a1cb6baaad52a436770c6e1fe25443b33",
"_n_e_o_n_2_depth_to_space_layer_8cpp.xhtml",
"_n_e_o_n_2_g_e_m_m_8cpp.xhtml#adaa0f4b3f66514cee57da32f79467d95",
"_n_e_o_n_2_max_unpooling_layer_8cpp.xhtml",
"_n_e_o_n_2_remap_8cpp.xhtml#ae7e59d7511e09055d4fae115f37b8329",
"_n_e_optical_flow_8cpp.xhtml",
"_n_e_stack_layer_kernel_8h_source.xhtml",
"_open_c_l_memory_usage_8h.xhtml",
"_quantization_info_8h.xhtml#adcbb8a7cd81427846571f9bef039f953",
"_slice_operations_8cpp_source.xhtml",
"_validate_8h.xhtml#a5022878c0277f395dec8055b219e679c",
"a64__hybrid__fp32__mla__16x4_2generic_8cpp_source.xhtml",
"arm__compute_2core_2_helpers_8h.xhtml#a4267e0c675915357d30839718990fe4e",
"arm__compute_2core_2_utils_8h.xhtml#a23062881efd3855d61bae58f330c97db",
"bias__adder_8hpp.xhtml#ad5675a7992a4e9eb9b2bc67e5715146d",
"classarm__compute_1_1_activation_layer_info.xhtml#a56297e0f7b215eea46c818cb7528d9eaa143c8c6f51b9bb893ce71e38702e3cc1",
"classarm__compute_1_1_c_l_bitwise_or.xhtml",
"classarm__compute_1_1_c_l_convert_fully_connected_weights_kernel.xhtml#a91315c3662a7ae017f5494c184eba184",
"classarm__compute_1_1_c_l_dilate_kernel.xhtml#a67b0c2ccd2c37a8d29fa6cc4b26795d8",
"classarm__compute_1_1_c_l_floor_kernel.xhtml#acad9b66c8f47a3aa6ec172c2fb26117f",
"classarm__compute_1_1_c_l_g_e_m_m_reshape_l_h_s_matrix_kernel.xhtml#ae6390852e3e6fb9761855cd12797cfb3",
"classarm__compute_1_1_c_l_integral_image.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_c_l_memory.xhtml#af2d30bc4b38a0f4703ff154a39f1b422",
"classarm__compute_1_1_c_l_quantization_layer.xhtml",
"classarm__compute_1_1_c_l_slice.xhtml#a96af67cce70fabc6bebb3a8608b6122b",
"classarm__compute_1_1_c_l_tensor_allocator.xhtml#a99c07c946023f413991a249db8b99ab0",
"classarm__compute_1_1_c_p_p_permute_kernel.xhtml#a93c836ab36443b23753d99495761daf7",
"classarm__compute_1_1_g_c_depthwise_convolution_layer3x3_kernel.xhtml#a169af81ccfe8f51bf4376f72e74fa23a",
"classarm__compute_1_1_g_c_scale_kernel.xhtml#a423f9a45a52983b4de5e2b347f4369c7",
"classarm__compute_1_1_i_c_l_memory_region.xhtml#a3d9d196240e9b0062c953c3abd534187",
"classarm__compute_1_1_i_n_e_harris_score_kernel.xhtml#a98cf472e8c62403e1995673e9b8ed1c3",
"classarm__compute_1_1_lut_allocator.xhtml#a5fdb67ad7cf44fcbc5bf7bd0a7a1ca09",
"classarm__compute_1_1_n_e_bounding_box_transform.xhtml",
"classarm__compute_1_1_n_e_deconvolution_layer.xhtml#ad14903ce0bc1e71d5d80d05652391985",
"classarm__compute_1_1_n_e_elementwise_unary_kernel.xhtml#a8a80c8f36d195d1a11c876cefb67f6b1",
"classarm__compute_1_1_n_e_g_e_m_m_lowp_output_stage.xhtml#a215ad3877ff40dcc4e6e39f25f2a27f0",
"classarm__compute_1_1_n_e_instance_normalization_layer.xhtml",
"classarm__compute_1_1_n_e_normalization_layer_kernel.xhtml#ab3498b8beb89599cee12a9e2ac083d88",
"classarm__compute_1_1_n_e_scale.xhtml#a77a3c38b3a2b77391f83102eecb8776c",
"classarm__compute_1_1_n_e_transpose_kernel.xhtml#a3a5122ad042e8f04a6cd96980bce3d49",
"classarm__compute_1_1_pyramid.xhtml#a62c986dcfc8a16c2fbdc8bf33c77a7b3",
"classarm__compute_1_1_tensor_shape.xhtml#a091252b04a1c79d499dd6184f9f5d715",
"classarm__compute_1_1experimental_1_1_i_operator.xhtml#a8ef96ccfabfca0faf4c3914c85bc4786",
"classarm__compute_1_1graph_1_1_fused_depthwise_convolution_batch_normalization_node.xhtml#adb48b5745c55605a2d4ec6f665bb7f3f",
"classarm__compute_1_1graph_1_1_prior_box_layer_node.xhtml#a4b7be447530e7ead6e3f6836edee2dd4",
"classarm__compute_1_1graph_1_1backends_1_1_n_e_node_validator.xhtml",
"classarm__compute_1_1graph__utils_1_1_validation_output_accessor.xhtml#ad20897c5c8bd47f5d4005989bead0e55",
"classarm__compute_1_1test_1_1_n_e_synthetize_function_with_zero_constant_border.xhtml#a3c585dd8a650e6194b40c7aac6701a6f",
"classarm__compute_1_1test_1_1framework_1_1_pretty_printer.xhtml#a73c2cb1847e4110041efe047cb9421e6",
"classarm__compute_1_1utils_1_1_j_p_e_g_loader.xhtml",
"classarm__gemm_1_1_quantize_wrapper.xhtml#abf5a58f6feffeae31f48b750cba9303d",
"crop__tensor_8cl_source.xhtml",
"div_8h_source.xhtml",
"functions_vars_t.xhtml",
"gemm__helpers_8h.xhtml#aab82226c20d3ee2f80364d93e492f91a",
"globals_l.xhtml",
"helpers__asymm_8h.xhtml#ae1ed2d7eea67b60ff64ed119fe273e2b",
"mean__stddev_8cl.xhtml",
"namespacemembers_func_z.xhtml",
"pooling__layer_8cl.xhtml#abb8f7128361a6a1965b1b2a5b3a719b2",
"reference_2_dequantization_layer_8cpp.xhtml#abd43030e06efec1c26997107b7bd184d",
"reference_2_pooling_layer_8cpp.xhtml#a81789776e276f1b0b394b6a72b55f179",
"rev64_8h.xhtml#a77b10f14845f83d937ac88959aee2f53",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a4090567b3adb034c7cc1af308cb45670",
"struct_keypoint.xhtml#a0a2f84ed7838f07779ae24c5a9086d33",
"structarm__compute_1_1_pooling_layer_info.xhtml#a93e2246c8f5500552788602f344b32f3",
"structarm__compute_1_1test_1_1framework_1_1_measurement_1_1_value.xhtml",
"structgemm__tuner_1_1_common_gemm_example_params.xhtml#af169764abf85f85bec428e43dc60274b",
"tests_2framework_2_macros_8h.xhtml#adc163370cc6af9884013477920955038",
"tests_2validation_2reference_2_utils_8h.xhtml#a5d1175c32ed7ea771e8ea46c936ea5c7",
"validation_2_c_l_2_scale_8cpp.xhtml#a948c5ab946718c1c6b7f2dc71780ddea"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';