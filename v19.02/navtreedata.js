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
    [ "List of functions", "functions_list.xhtml", [
      [ "NEON functions", "functions_list.xhtml#S5_1", null ],
      [ "OpenCL functions", "functions_list.xhtml#S5_2", null ],
      [ "GLES Compute functions", "functions_list.xhtml#S5_3", null ],
      [ "CPP functions", "functions_list.xhtml#S5_4", null ]
    ] ],
    [ "Library architecture", "architecture.xhtml", [
      [ "Core vs Runtime libraries", "architecture.xhtml#S4_1", null ],
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
      [ "OpenCL Tuner", "architecture.xhtml#S4_8_opencl_tuner", null ]
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
    [ "Deprecated List", "deprecated.xhtml", null ],
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
"_absolute_difference_8h.xhtml",
"_c_l_2_batch_to_space_layer_8cpp.xhtml#a157af9e1f21df8785893e442409a4435",
"_c_l_2_g_e_m_m_reshape_l_h_s_matrix_8cpp.xhtml#a15438f2dd36e0e42c2e8aaf4b19e4f3c",
"_c_l_2_scharr_8cpp.xhtml",
"_c_l_absolute_difference_8cpp_source.xhtml",
"_c_l_depth_convert_layer_kernel_8h_source.xhtml",
"_c_l_g_e_m_m_transpose1x_w_kernel_8cpp.xhtml",
"_c_l_normalize_planar_y_u_v_layer_kernel_8h.xhtml",
"_c_l_tensor_handle_8cpp.xhtml",
"_cast_8cpp.xhtml#a0ed7521147c7992d5f65b360cd2812f5",
"_derivative_8h.xhtml#aeae8f44225b61c5a6b05fdfcd82ae3d1",
"_fully_connected_layer_8h_source.xhtml",
"_g_c_scheduler_8cpp.xhtml",
"_helpers_8inl.xhtml#a46e938020a3ac8c926d0590b7fe957db",
"_i_tensor_info_8h_source.xhtml",
"_n_e_bitwise_not_8cpp_source.xhtml",
"_n_e_direct_convolution_detail_8h.xhtml#ab7e696498dc262dc55dc90e8d4a2fc77",
"_n_e_h_o_g_detector_8cpp.xhtml",
"_n_e_o_n_2_arithmetic_subtraction_8cpp.xhtml#acb5454ca4d9a7aac50c9b24ab9a0f514",
"_n_e_o_n_2_gaussian3x3_8cpp.xhtml#afad045c7bea57340a4532a78a6b04e7c",
"_n_e_o_n_2_stack_layer_8cpp.xhtml#aefbc705cede4006a11ee179b95d6468b",
"_n_e_sobel5x5_kernel_8h_source.xhtml",
"_node_fusion_mutator_8cpp_source.xhtml",
"_p_m_u_8h_source.xhtml",
"_single_thread_scheduler_8h_source.xhtml",
"_validate_8h.xhtml#a693decaffb042b3ff76e274f983e8ac1",
"a64__sgemm__12x8_2a53_8cpp.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a683661ae75dcb7aef16b9c9bde31517d",
"arm__compute_2graph_2frontend_2_types_8h_source.xhtml",
"benchmark_2_c_l_2_normalization_layer_8cpp.xhtml#acec794dbb7f6c77bdefa188d83fc80c9",
"benchmark_2_n_e_o_n_2_convolution_8cpp_source.xhtml",
"canny_8cl.xhtml#a64ee229d1bcee88c8017356d5d485650",
"classarm__compute_1_1_c_l_arithmetic_operation_kernel.xhtml",
"classarm__compute_1_1_c_l_convolution_rectangle_kernel.xhtml#a423f9a45a52983b4de5e2b347f4369c7",
"classarm__compute_1_1_c_l_exp_layer.xhtml",
"classarm__compute_1_1_c_l_g_e_m_m_reshape_r_h_s_matrix_kernel.xhtml#a493987e85723a8000eb26d1f00e2ad0e",
"classarm__compute_1_1_c_l_l_k_tracker_init_kernel.xhtml#a493987e85723a8000eb26d1f00e2ad0e",
"classarm__compute_1_1_c_l_permute_kernel.xhtml",
"classarm__compute_1_1_c_l_sobel5x5_hor_kernel.xhtml#a0d2de1f5a7010147dc1d6c11eaaeda37",
"classarm__compute_1_1_c_l_warp_perspective_kernel.xhtml",
"classarm__compute_1_1_dimensions.xhtml#a0d3c59537291735849c740364496a41c",
"classarm__compute_1_1_g_c_kernel.xhtml#af29ae815590ed07fc2ce2dc3f77a23a7",
"classarm__compute_1_1_i_array.xhtml#aac8e28a698cd201286d75eb3f5ad3e1c",
"classarm__compute_1_1_i_lut.xhtml#a5eeb94d22b8366d1b68d0614384802fe",
"classarm__compute_1_1_lut_allocator.xhtml#a5fdb67ad7cf44fcbc5bf7bd0a7a1ca09",
"classarm__compute_1_1_n_e_channel_shuffle_layer_kernel.xhtml#a3a050bd26e07d30bf7e578ecf1174d05",
"classarm__compute_1_1_n_e_direct_convolution_layer_kernel.xhtml#a537ba0d35bcc8d5488da55d1a27c89a3",
"classarm__compute_1_1_n_e_g_e_m_m_lowp_matrix_b_reduction_kernel.xhtml",
"classarm__compute_1_1_n_e_im2_col.xhtml",
"classarm__compute_1_1_n_e_pixel_wise_multiplication_kernel.xhtml#ac23503429643fa0415fd64b1fc17e40c",
"classarm__compute_1_1_n_e_sobel7x7_hor_kernel.xhtml#ace7523f9c3073ad82b77e46318a1ea77",
"classarm__compute_1_1_pad_stride_info.xhtml#ad71c061b948d43c30e489e15fee6dc8b",
"classarm__compute_1_1_tensor_info.xhtml",
"classarm__compute_1_1graph_1_1_dummy_node.xhtml#aefa24b710045e042672a5e887c3efaef",
"classarm__compute_1_1graph_1_1_prior_box_layer_node.xhtml#adb48b5745c55605a2d4ec6f665bb7f3f",
"classarm__compute_1_1graph_1_1frontend_1_1_concat_layer.xhtml",
"classarm__compute_1_1test_1_1_array_accessor.xhtml#aa733629a56800aca0b5cb4608069718b",
"classarm__compute_1_1test_1_1framework_1_1_fixture.xhtml",
"classarm__compute_1_1test_1_1framework_1_1dataset_1_1_singleton_dataset.xhtml#a0c62c15c8ed609e7e5e9518cf5f5c712",
"classarm__gemm_1_1_gemv_pretransposed.xhtml#aa7cfff39cbb2be65cd40042a75e5ae1c",
"dir_3cbe5b6455504b72bdad57647abe41ab.xhtml",
"functions_v.xhtml",
"graph__mobilenet_8cpp_source.xhtml",
"magnitude__phase_8cl.xhtml#a02ff978b574e44604d625dbd470ab870",
"namespacemembers_i.xhtml",
"pooling__layer__quantized_8cl.xhtml#a5c74cefcb6318451b5785dd00bd9b05f",
"reference_2_reorg_layer_8cpp_source.xhtml",
"softmax__layer_8cl.xhtml#af5b2e33e3c5fcaab3a213f26c2300170",
"structarm__compute_1_1_g_e_m_m_lowp_output_stage_info.xhtml#a7d24a8e7bad9b4d51aae3af31072d896",
"structarm__compute_1_1test_1_1framework_1_1_test_result.xhtml#a67a0db04d321a74b7e7fcfd3f1a3f70ba9c51674930e03c276344d19f9e4398fb",
"tests_2_utils_8h.xhtml",
"tests_2validation_2_n_e_o_n_2_u_n_i_t_2_tensor_allocator_8cpp_source.xhtml",
"validation_2_c_l_2_activation_layer_8cpp.xhtml#a0b8743d7c78a9fa2c303aa1b255a1b8e",
"validation_2_c_l_2_g_e_m_m_8cpp.xhtml",
"validation_2_c_l_2_softmax_layer_8cpp.xhtml",
"validation_2_n_e_o_n_2_convolution_layer_8cpp.xhtml#af3c1de77fd86df539395c75c17ec230e",
"validation_2_n_e_o_n_2_sobel_8cpp.xhtml",
"validation_2reference_2_transpose_8cpp.xhtml#af5f82318aa0982e38535d512accf3177"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';