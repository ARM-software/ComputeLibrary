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
        [ "The OpenCL stub library", "index.xhtml#S3_6_cl_stub_library", null ],
        [ "The Linux OpenGLES and EGL stub libraries", "index.xhtml#S3_7_gles_stub_library", null ],
        [ "OpenCL DDK Requirements", "index.xhtml#S3_8_cl_requirements", [
          [ "Hard Requirements", "index.xhtml#S3_8_1_cl_hard_requirements", null ],
          [ "Performance improvements", "index.xhtml#S3_8_2_cl_performance_requirements", null ]
        ] ],
        [ "OpenCL Tuner", "index.xhtml#S3_9_cl_tuner", [
          [ "How to use it", "index.xhtml#S3_9_1_cl_tuner_how_to", null ]
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
"_c_l_2_arithmetic_addition_8cpp.xhtml#a5684fd635df10ab5b0ebc921a825128b",
"_c_l_2_channel_shuffle_8cpp.xhtml#a5ec14cb26a1f3dbfbc35064f1c560bdc",
"_c_l_2_depth_convert_layer_8cpp.xhtml#a19e1fbe931820770862d454fc05d2f1c",
"_c_l_2_flatten_8cpp.xhtml#acdd086d8a6a088acde2e91a617b8fccb",
"_c_l_2_im2_col_8cpp.xhtml#ab05fa88a2f816dce047f4d47acc276fc",
"_c_l_2_phase_8cpp.xhtml#a951f0fc437f1981a9df35290872df0a5",
"_c_l_2_space_to_batch_layer_8cpp.xhtml#a3cbddfe65d4cf806d066eb3a38248cdf",
"_c_l_2_winograd_8cpp.xhtml#af94bf82577407bd38165f4e84c4c360a",
"_c_l_crop_kernel_8cpp_source.xhtml",
"_c_l_g_e_m_m_lowp_matrix_multiply_reshaped_only_r_h_s_kernel_8cpp_source.xhtml",
"_c_l_locally_connected_layer_8cpp_source.xhtml",
"_c_l_reverse_8cpp.xhtml",
"_c_l_winograd_input_transform_kernel_8cpp.xhtml",
"_color_convert_helper_8h.xhtml#ae5035a982d39c4905c9ed3ac150f9917",
"_elementwise_negation_8cpp.xhtml#a3b7a703f0d348a36ff12a20529c944ec",
"_function_helpers_8h.xhtml#a627f6bdc4a7de6dbb03acb3d8b3a4d6d",
"_g_c_scale_8h.xhtml",
"_g_l_e_s___c_o_m_p_u_t_e_2_pooling_layer_8cpp.xhtml",
"_i_c_l_kernel_8h.xhtml#a6e51ab3789678d3e0b0b72178dd6c4c6",
"_instruments_stats_8h_source.xhtml",
"_n_e_bitwise_and_8h_source.xhtml",
"_n_e_dilate_kernel_8cpp_source.xhtml",
"_n_e_g_e_m_m_lowp_quantize_down_int32_to_uint8_scale_by_fixed_point_kernel_8cpp.xhtml",
"_n_e_median3x3_kernel_8h_source.xhtml",
"_n_e_o_n_2_cast_8cpp.xhtml#ae02c6fc90d9c60c634bfa258049eb46b",
"_n_e_o_n_2_depth_convert_layer_8cpp.xhtml#a3ed17ec92926b05f647ae5dc75f18397",
"_n_e_o_n_2_fill_border_8cpp_source.xhtml",
"_n_e_o_n_2_laplacian_pyramid_8cpp_source.xhtml",
"_n_e_o_n_2_reduce_mean_8cpp.xhtml",
"_n_e_o_n_2_y_o_l_o_layer_8cpp.xhtml",
"_n_e_split_8cpp_source.xhtml",
"_open_c_l_8h.xhtml#af6e65f36e2be6bd9a1cac10ba6e82f7c",
"_quantization_info_8h.xhtml#af7586ba11d82cc0232d4246a61a4a4f3",
"_softmax_layer_node_8cpp_source.xhtml",
"_validate_8h.xhtml#a89059fcfb27bba71edffcc22c6e1a1a2",
"a64__merge__fp16__24x8_8hpp_source.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a14d24d90ab4ba2956e92e27890ba4c91a8d6b5cada83510220f59e00ce86d4d92",
"arm__compute_2core_2utils_2logging_2_printers_8h_source.xhtml",
"cge_8h.xhtml#aed06c8595807184acaeac3075dc9495a",
"classarm__compute_1_1_c_l_accumulate_squared.xhtml#ade36d9a0c92807caac072618a6d9a643",
"classarm__compute_1_1_c_l_comparison_kernel.xhtml#a9eb86fbe4825b43b638171bb6dd398c3",
"classarm__compute_1_1_c_l_derivative_kernel.xhtml#ab24f49526202babfe7df925cd326427b",
"classarm__compute_1_1_c_l_g_e_m_m_lowp_matrix_a_reduction_kernel.xhtml",
"classarm__compute_1_1_c_l_generate_proposals_layer.xhtml#a5238a3faae0338e0f8cba6d62e1ad94e",
"classarm__compute_1_1_c_l_locally_connected_matrix_multiply_kernel.xhtml#a0a3641bb70273c2005e53942c51db9bb",
"classarm__compute_1_1_c_l_pixel_wise_multiplication_kernel.xhtml#a18767cd831fa27f6c0d28e847509b4cf",
"classarm__compute_1_1_c_l_sobel3x3_kernel.xhtml#ab30b6bb92fb3a1cb0cda6dc08e9fa160",
"classarm__compute_1_1_c_l_tuner.xhtml#aba10acdb2d58e3e0a96364c487a71d40",
"classarm__compute_1_1_c_p_p_upsample_kernel.xhtml#ade0b22ebd85d13fa476cbb3260b675b8",
"classarm__compute_1_1_g_c_fully_connected_layer.xhtml#a840ecacfbb054d2245e1864e61178421",
"classarm__compute_1_1_g_c_tensor_shift_kernel.xhtml",
"classarm__compute_1_1_i_c_l_tuner.xhtml",
"classarm__compute_1_1_i_runtime_context.xhtml#aa2c17793f71ecb166b947f0eab98bfc5",
"classarm__compute_1_1_n_e_absolute_difference_kernel.xhtml#a8f920e3071a96d1cb1bf77dabced479e",
"classarm__compute_1_1_n_e_color_convert_kernel.xhtml#adc4007361101a56f38f1686590c3a6b5",
"classarm__compute_1_1_n_e_derivative.xhtml#a06a119e5e329e9283c4c884a1d927914",
"classarm__compute_1_1_n_e_fully_connected_layer_reshape_weights.xhtml#a83a344e60eb7db895953a942abf16628",
"classarm__compute_1_1_n_e_gaussian_pyramid.xhtml#af34c199e11afef4c68af773a9fd40150",
"classarm__compute_1_1_n_e_magnitude.xhtml#a1aa524e83bd73b149831dc0bb6e39e7e",
"classarm__compute_1_1_n_e_r_o_i_pooling_layer_kernel.xhtml",
"classarm__compute_1_1_n_e_space_to_depth_layer.xhtml#abe7a463ff78322ea575c7efcaacd66a2",
"classarm__compute_1_1_pixel_value.xhtml#a1234f83efa812e259523c91799614a3c",
"classarm__compute_1_1_tensor_allocator.xhtml#a1468b0adb6ec3f9d38aa7d60b8a91974",
"classarm__compute_1_1graph_1_1_concatenate_layer_node.xhtml#a6272eb9643b239dd0c1e7adde3ee093b",
"classarm__compute_1_1graph_1_1_i_node.xhtml#a7069f1901ff50cd2ac6ba152cb106058",
"classarm__compute_1_1graph_1_1_upsample_layer_node.xhtml#a6272eb9643b239dd0c1e7adde3ee093b",
"classarm__compute_1_1graph_1_1frontend_1_1_input_layer.xhtml#a8c543d7a0c50bd07dfba4bbc6ba1ee53",
"classarm__compute_1_1test_1_1_assets_library.xhtml#aa13fcfba9d7f0433db83255bd1f0638a",
"classarm__compute_1_1test_1_1framework_1_1_common_options.xhtml#a20a9bb6edc108f6ee20c822b68531b5d",
"classarm__compute_1_1test_1_1framework_1_1dataset_1_1_container_dataset.xhtml#af9593d4a5ff4274efaf429cb4f9e57cc",
"classarm__compute_1_1utils_1_1random_1_1_ranged_uniform_distribution.xhtml#a50f59630d99c12348ad67924da225bb6",
"combine_8h.xhtml#ac0a3fb58591449faff844901e8191027",
"dir_5e492e6e3fe2ac2919691e9872c36ea9.xhtml",
"fft_8cl.xhtml#ad4dcc4a8b94f263cd19c59fdc2cec3d2",
"gemm__helpers_8h.xhtml#a985131db76a28d171da0810a8b0f70ee",
"graph__convolution_8cpp.xhtml#a3c04138a5bfe5d72780bb7e82a18e627",
"index.xhtml#S3_7_gles_stub_library",
"mul_8h.xhtml#a34bb12fc418530504162b300bf417ee9",
"neon__permute_8cpp_source.xhtml",
"reference_2_box3x3_8cpp_source.xhtml",
"reference_2_integral_image_8cpp_source.xhtml",
"reference_2_warp_perspective_8cpp.xhtml",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a0f6ad555d097377151003a5119ccee45",
"struct_vector.xhtml",
"structarm__compute_1_1detail_1_1brelu.xhtml#abd5aa7b37cf4c0a02b4bf2d4692d3750",
"structarm__compute_1_1test_1_1validation_1_1_canny_edge_parameters.xhtml#ab8a8d1ea5336156261d2fa39d65bd3f3",
"tbl_8h.xhtml#a0a230ea3527f48b002a8ea9ea08aa840",
"tests_2validation_2_helpers_8cpp.xhtml#ab845254df9508ca05606545601bebb96",
"utils_2_type_printer_8h.xhtml#aae534105c7ea67999ccbb34a0ed567cd"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';