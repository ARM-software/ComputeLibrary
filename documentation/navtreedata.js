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
"_c_l_2_arithmetic_addition_8cpp.xhtml#af39fb33d73c7cc13659cb7c5bcf2c6f1",
"_c_l_2_crop_resize_8cpp.xhtml#a04cbd579ffebdb4072c71ce71208d948",
"_c_l_2_g_e_m_m_reshape_r_h_s_matrix_8cpp.xhtml#a4cffb1d9cb2dc0bc65b752b757eef4e5",
"_c_l_2_r_o_i_align_layer_8cpp.xhtml#a18e24f4dd3a3c8927fd48e9b1e65d617",
"_c_l_2_upsample_layer_8cpp.xhtml#a778b3df465ed2c9457feb9ea872f0ca9",
"_c_l_channel_extract_8cpp.xhtml",
"_c_l_f_f_t_radix_stage_kernel_8h_source.xhtml",
"_c_l_harris_corners_8h_source.xhtml",
"_c_l_prior_box_layer_8cpp.xhtml",
"_c_l_transpose_kernel_8h_source.xhtml",
"_c_p_p_upsample_kernel_8cpp_source.xhtml",
"_detection_output_layer_node_8h_source.xhtml",
"_execution_helpers_8h.xhtml#a82d07d3d612bee8ff703226ff9d5d452",
"_g_c_g_e_m_m_transpose1x_w_8h.xhtml",
"_g_l_e_s___c_o_m_p_u_t_e_2_arithmetic_addition_8cpp_source.xhtml",
"_i_c_l_multi_image_8h.xhtml#aa80145f30ddae0c2ccbcaa910e3e71dd",
"_join_dataset_8h_source.xhtml",
"_n_e_bitwise_xor_kernel_8cpp_source.xhtml",
"_n_e_direct_convolution_detail_8h.xhtml#a5db34f1ae85326efebbe1dadb65f0301",
"_n_e_g_e_m_m_matrix_multiply_kernel_8cpp.xhtml",
"_n_e_non_maxima_suppression3x3_8cpp_source.xhtml",
"_n_e_o_n_2_convert_fully_connected_weights_8cpp.xhtml#ac2ed31007ae463a3cec24a581f3651f6",
"_n_e_o_n_2_gather_8cpp.xhtml#ae3804b9ad1989bb4c14c64b4cb7b36e2",
"_n_e_o_n_2_r_n_n_layer_8cpp.xhtml#a5179d211b87d8c1e650219c9288b2d67",
"_n_e_pad_layer_kernel_8cpp_source.xhtml",
"_n_e_softmax_layer_kernel_8cpp.xhtml#a93020987fbede1b8da7db6a1a855ddd4",
"_normalize_planar_y_u_v_layer_8h.xhtml",
"_pool_manager_8cpp.xhtml",
"_shape_calculator_8h.xhtml#ad16b366db486fec63b6d962937ec4545",
"_utility_8h.xhtml#a1e384f81bb641de61df2800a432c51fe",
"a64__gemm__u16__12x8_8hpp.xhtml",
"arm__compute_2core_2_helpers_8h.xhtml#a6f698fa1629f7f800b3c8cd77a3d4b4a",
"arm__compute_2core_2_utils_8h.xhtml#a8b82da7b5e0f6192f415ac347c9e0555",
"benchmark_2_c_l_2_convolution_layer_8cpp.xhtml#a2aed5930b7e89685d562842c50cd1522",
"benchmark_2_g_l_e_s___c_o_m_p_u_t_e_2_convolution_layer_8cpp.xhtml",
"benchmark_2_n_e_o_n_2_harris_corners_8cpp.xhtml#aad65e9549e68a0293552a8b73130d8fe",
"channel__extract_8cl_source.xhtml",
"classarm__compute_1_1_c_l_arithmetic_subtraction.xhtml#a7af75ca77a6e9eb53532e7ab1317bdc3",
"classarm__compute_1_1_c_l_convolution3x3.xhtml#a26e1b4686b1f2d591d62d11585114a82",
"classarm__compute_1_1_c_l_distribution1_d.xhtml#a454323653553eb092330e739ce3882eb",
"classarm__compute_1_1_c_l_g_e_m_m_lowp_matrix_multiply_reshaped_only_r_h_s_kernel.xhtml#ad12963f1b6b9bcdb92da8de2e5491b97",
"classarm__compute_1_1_c_l_h_o_g_gradient.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_c_l_lut_allocator.xhtml#aa1e425aebd0805be1916fddde364aaa1",
"classarm__compute_1_1_c_l_quantization_layer_kernel.xhtml#aec2ad990f91ebeb1c5e26426bcb9529c",
"classarm__compute_1_1_c_l_space_to_batch_layer.xhtml#a948804ad61ca2c4d419876e6dec1c024",
"classarm__compute_1_1_c_l_width_concatenate4_tensors_kernel.xhtml#a26b9ccaf52c73ed92fc20f8456315592",
"classarm__compute_1_1_detection_post_process_layer_info.xhtml#adda82c28c368106734620f105bb0e1e3",
"classarm__compute_1_1_g_c_im2_col_kernel.xhtml#a430228b1b1392252b298974038b69abd",
"classarm__compute_1_1_generate_proposals_info.xhtml#ac2a327bd7d58d9fa1c49c302fefbbcba",
"classarm__compute_1_1_i_g_c_memory_region.xhtml#a9b586073abda157c8aa9cfc1d1a6baf7",
"classarm__compute_1_1_i_tensor_info.xhtml#a0cd5aea514f03afb48ebe22581ea4c66",
"classarm__compute_1_1_n_e_arithmetic_subtraction.xhtml",
"classarm__compute_1_1_n_e_convolution_kernel.xhtml",
"classarm__compute_1_1_n_e_edge_non_max_suppression_kernel.xhtml#a57a36344e261edfdfa97b19ac40b7c30",
"classarm__compute_1_1_n_e_g_e_m_m_convolution_layer.xhtml#ac13c7e80f7e9e6f1e2912ef50678c5c9",
"classarm__compute_1_1_n_e_h_o_g_detector_kernel.xhtml#ac8fd84865e94c9be09c574eb9abdfde6",
"classarm__compute_1_1_n_e_min_max_layer_kernel.xhtml#aa517db8efb13c3f64c91e7ce2de9c5e7",
"classarm__compute_1_1_n_e_reshape_layer_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_tile_kernel.xhtml#ad226085cfab135a5367504930fdd189b",
"classarm__compute_1_1_program.xhtml#a1d89c28bd42ba9a52da008bb69367171",
"classarm__compute_1_1_tensor_info.xhtml#ad03af3eeb6f3666d6282ca689c1b2ce8",
"classarm__compute_1_1graph_1_1_detection_output_layer_node.xhtml#a747fc2559359c5004e3034a30ec7579c",
"classarm__compute_1_1graph_1_1_input_node.xhtml#a45cded3088bd7dcefdf86b17b143aa16",
"classarm__compute_1_1graph_1_1backends_1_1_g_c_device_backend.xhtml#ad02dea2a998dcd64725620b72022ba56",
"classarm__compute_1_1graph__utils_1_1_image_accessor.xhtml#a872e7ef3563a74e35a6912d12706c012",
"classarm__compute_1_1test_1_1_g_c_accessor.xhtml#ad7701a09a964eab360a8e51fa7ad2c16",
"classarm__compute_1_1test_1_1framework_1_1_mali_counter.xhtml",
"classarm__compute_1_1utils_1_1_common_graph_validate_options.xhtml#a9dbe8cb92349b9aec5281e4e37a2f487",
"classarm__gemm_1_1_gemm_interleaved.xhtml#a468e9c50d4decc2ba86f9bd393ba27d6",
"core_2_c_l_2_c_l_helpers_8h.xhtml#a0019d1de2500c73f16b673d8a883a767",
"direct__convolution9x9_8cl.xhtml",
"gaussian__pyramid_8cl.xhtml#ae8b283818e0ddb9edf271668d2682dc3",
"generate__proposals__quantized_8cl.xhtml",
"helpers__asymm_8h.xhtml#a35b98dc80eefc6ce799720861a668691",
"magnitude__phase_8cl.xhtml#ab0d7e891c8a09824c46baa2393d6961f",
"namespacemembers_func_d.xhtml",
"pmin_8h.xhtml#ab12ca96a30ec693404ee462fa9972441",
"reference_2_im2_col_8cpp.xhtml#a77f0a04bc3f800ccd330df1170e76344",
"scalar_2add_8h.xhtml",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#adc30a7690418156dd429314c58634328",
"structarm__compute_1_1_g_e_m_m_lowp_output_stage_info.xhtml#ae5bbe76a801cb3001c3ea60bd9fb71d8",
"structarm__compute_1_1test_1_1framework_1_1_measurement_1_1_value.xhtml#aea625d488d9eafbc023001f74f15f7af",
"sve__hybrid__fp32__mla__4_v_lx4_2generic_8cpp_source.xhtml",
"tests_2framework_2_macros_8h.xhtml#a9c14b58feb41d3702ce3ed63822ac2a8",
"tests_2validation_2reference_2_utils_8h.xhtml#ab50b3b23d5b3e67cca71a12b91c2a8db",
"validation_2_c_l_2_batch_normalization_layer_8cpp.xhtml#a8d9a1eec393939154689c980ef44cfca",
"validation_2_c_l_2_floor_8cpp.xhtml",
"validation_2_c_l_2_remap_8cpp.xhtml#aa2468a9d9dc7b6450258bbb8eff28390",
"validation_2_n_e_o_n_2_convolution_8cpp.xhtml#ac613cfb8cf4e990c42e0e7ef57bf6fad",
"validation_2_n_e_o_n_2_min_max_location_8cpp.xhtml#a2717a3e83eb53f04462a3b678b9b06cd",
"validation_2reference_2_l2_normalize_layer_8cpp.xhtml#a238d3ab8cdae100387631d6491204c6b"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';