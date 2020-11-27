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
      [ "Data-type and Data-layout support", "architecture.xhtml#S4_1_2", null ],
      [ "Fast-math support", "architecture.xhtml#S4_1_3", null ],
      [ "Thread-safety", "architecture.xhtml#S4_1_4", null ],
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
        [ "Macros", "globals_defs.xhtml", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"_abs_layer_8cpp.xhtml",
"_c_l_2_arg_min_max_8cpp.xhtml#a47d01269d8515680e7a30ec8ab19bc31",
"_c_l_2_cast_8cpp.xhtml#ad268d70596992c77669e6bf32f8bacc3",
"_c_l_2_depth_convert_layer_8cpp.xhtml#a656df9dd7c5ab11fed1ea622b1d30880",
"_c_l_2_floor_8cpp.xhtml#a4a6c09a72db979494ed1b4b62db649cc",
"_c_l_2_l_s_t_m_layer_quantized_8cpp.xhtml#a1bb4ab3817a5bf7c596a3a7774c891d9",
"_c_l_2_prior_box_layer_8cpp.xhtml",
"_c_l_2_stack_layer_8cpp.xhtml#a7298d6d944879e8266a0699808c18d16",
"_c_l_array_8h.xhtml#ab8d2c7efb7643ce8f9e5446dbe1da0b4",
"_c_l_depthwise_convolution_layer_8h_source.xhtml",
"_c_l_g_e_m_m_matrix_multiply_native_kernel_8cpp_source.xhtml",
"_c_l_magnitude_phase_kernel_8h.xhtml",
"_c_l_reverse_kernel_8cpp_source.xhtml",
"_c_l_winograd_output_transform_kernel_8cpp.xhtml",
"_color_convert_helper_8h.xhtml#ac3138daaa2a074986406b22ec6f3d909",
"_elementwise_division_8cpp.xhtml#aa9575b681374a73e698577888adc8aef",
"_function_helpers_8h.xhtml#a0331e6b2b68ea76e9415f7f148d92601",
"_g_c_pixel_wise_multiplication_8h_source.xhtml",
"_g_l_e_s___c_o_m_p_u_t_e_2_depth_concatenate_layer_8cpp_source.xhtml",
"_i_array_8h_source.xhtml",
"_in_place_operation_mutator_8cpp_source.xhtml",
"_n_e_arithmetic_subtraction_8cpp.xhtml",
"_n_e_deconvolution_layer_8h.xhtml",
"_n_e_floor_kernel_8h.xhtml",
"_n_e_integral_image_kernel_8h_source.xhtml",
"_n_e_o_n_2_arithmetic_subtraction_8cpp.xhtml#a2fbc975f1ce454a53c018dd9d4f19e87",
"_n_e_o_n_2_convolution_layer_8cpp.xhtml#aa28d2cc5d04ff5315c274f3399af6239",
"_n_e_o_n_2_direct_convolution_layer_8cpp.xhtml",
"_n_e_o_n_2_im2_col_8cpp.xhtml#a5e304f37dd22343195d1c1803bea4b78",
"_n_e_o_n_2_pixel_wise_multiplication_8cpp.xhtml#af7bb763a0b5768bc11201378eb9709ef",
"_n_e_o_n_2_threshold_8cpp.xhtml",
"_n_e_sobel3x3_8cpp_source.xhtml",
"_offset_lifetime_manager_8cpp_source.xhtml",
"_pool_manager_8h_source.xhtml",
"_shape_calculator_8h.xhtml#a585529133e437dc5f935d33de17c4abb",
"_trace_point_8h.xhtml#a9f1fd6408b8fa1dfd1179ffd7cbe8c93",
"a64__gemm__u8__8x12_2a55r1_8cpp_source.xhtml",
"activation__quant__helpers_8h.xhtml#ac48bfc2edd6f1195fbbd228ae1a85ef8",
"arm__compute_2core_2_types_8h.xhtml#ab4e88c89b3b7ea1735996cc4def22d58ab08f0cb36474118c5bbc03b3a172a778",
"arm__compute_2graph_2_type_printer_8h.xhtml",
"canny_8cl.xhtml#a333a1d2eee75220d5161fdb0e778867d",
"classarm__compute_1_1_access_window_transpose.xhtml#ae56df086bde04273b067b844c378db01",
"classarm__compute_1_1_c_l_batch_to_space_layer_kernel.xhtml#a6dcb3fb08a1947b68158ed175d0b60c3",
"classarm__compute_1_1_c_l_compute_all_anchors_kernel.xhtml#a4aaba95024a47f19a87894e21ef6022b",
"classarm__compute_1_1_c_l_dequantization_layer.xhtml",
"classarm__compute_1_1_c_l_fast_corners_kernel.xhtml",
"classarm__compute_1_1_c_l_g_e_m_m_matrix_multiply_native_kernel.xhtml#a5cc0a5be72202db1773b7198993495a6",
"classarm__compute_1_1_c_l_histogram.xhtml#a406ea1342bf3679752ba91525c59aaf6",
"classarm__compute_1_1_c_l_logits1_d_norm_kernel.xhtml#a19373a3162839643484511cec353d34b",
"classarm__compute_1_1_c_l_p_relu_layer.xhtml#adb00c00c17a8c28e3ac0921aec0259dc",
"classarm__compute_1_1_c_l_reverse_kernel.xhtml#a1acfeaa60695d4df61d8d4b5c905aa53",
"classarm__compute_1_1_c_l_space_to_depth_layer.xhtml#ab6c521c37a2109b7985b31dfcae4dd73",
"classarm__compute_1_1_c_l_weights_reshape_kernel.xhtml#ad012198a049187b766ca44b356821918",
"classarm__compute_1_1_detection_output_layer_info.xhtml#acf35ae15a9350f47bcba0d0cedeb3e7c",
"classarm__compute_1_1_g_c_g_e_m_m_matrix_multiply_kernel.xhtml#a0b0ecc60be6600a55aa46dffaa803ed9",
"classarm__compute_1_1_g_e_m_m_reshape_info.xhtml#a757197ffaf53ef6b284c6ceb24fdb688",
"classarm__compute_1_1_i_function.xhtml",
"classarm__compute_1_1_i_simple_lifetime_manager.xhtml",
"classarm__compute_1_1_n_e_absolute_difference_kernel.xhtml#a5d6e77370a140f10010d3cb9e2a11857",
"classarm__compute_1_1_n_e_box3x3_kernel.xhtml#a423f9a45a52983b4de5e2b347f4369c7",
"classarm__compute_1_1_n_e_crop_resize.xhtml#a05254b9c1c9d4a736764d8d45eb97210",
"classarm__compute_1_1_n_e_elementwise_max.xhtml#a6ad355ed142997ff0232e84112b473e1",
"classarm__compute_1_1_n_e_g_e_m_m.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_n_e_gaussian5x5.xhtml#ae1a0b9bad33c9ffab96843aff2e1e218",
"classarm__compute_1_1_n_e_integral_image_kernel.xhtml#a5a3e4e8ed3b5e4c7b68ea6a02827d396",
"classarm__compute_1_1_n_e_min_max_kernel.xhtml#aa977f349fb2c7d5ae7cd41fee339668a",
"classarm__compute_1_1_n_e_r_o_i_pooling_layer_kernel.xhtml#a5224515065ba1467fd0cf092b08cc614",
"classarm__compute_1_1_n_e_sobel7x7_vert_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_n_e_winograd_layer_transform_weights_kernel.xhtml#a4370ae5fda7bd455a171fc8ed4d3f283",
"classarm__compute_1_1_steps.xhtml#ad5ac35535f5df9d92f9a77bf4d772f76",
"classarm__compute_1_1cl__gemm_1_1_c_l_g_e_m_m_native_kernel_configuration_midgard.xhtml#a656562ef74f9c5be24abde6f6c8fd92a",
"classarm__compute_1_1gles_1_1_n_d_range.xhtml#a79ccb8003ef6ebe1c8e5b624fad982a6",
"classarm__compute_1_1graph_1_1_graph_context.xhtml#a22acc02bd87a26f9a08eb821bb35e75f",
"classarm__compute_1_1graph_1_1_reorg_layer_node.xhtml",
"classarm__compute_1_1graph_1_1backends_1_1_n_e_sub_tensor_handle.xhtml#a5d9a543899a9f7c93a950a1d080f2437",
"classarm__compute_1_1graph__utils_1_1_validation_input_accessor.xhtml",
"classarm__compute_1_1test_1_1_lut_accessor.xhtml#a1963ce09163dcfe7437106500bc2482d",
"classarm__compute_1_1test_1_1framework_1_1_pretty_printer.xhtml",
"classarm__compute_1_1utils_1_1_i_image_loader.xhtml#a25c255918a7306769a9e3886166af935",
"classarm__gemm_1_1_gemm_interleaved.xhtml#ae9fc7380e533c43a1d91e48a7578e412",
"core_2_c_l_2_c_l_helpers_8cpp.xhtml#a0019d1de2500c73f16b673d8a883a767",
"dir_a0f71818ea2c5c08950cb56389fab822.xhtml",
"fft__scale_8cl.xhtml#aab20f99ce45fbea55b054515089b52da",
"gemm__helpers_8h.xhtml#a519be13ff5823b035d64a1bd413680eb",
"getlane_8h.xhtml#aa16ace001ab8287faa46d6962f369219",
"helpers__asymm_8h.xhtml#a44e4d74ed42006c4153a9cb6c97285de",
"intrinsics_2add_8h.xhtml#ac061b47e13fa4832ee51b203de487325",
"mla_8h.xhtml#ac1a53ae8a71f3589b9af290a8a181c14",
"namespacemembers_vars_d.xhtml",
"quantization__layer_8cl.xhtml",
"reference_2_fuse_batch_normalization_8cpp.xhtml#afabcd35cc620facb8b136711b97085df",
"reference_2_range_8cpp.xhtml#af736e2b03b30fa5c7c98b387d3625f7c",
"runtime_2_c_l_2_c_l_helpers_8cpp.xhtml#aca9062d049299f125a442eac466402fd",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#a58f2fc2ba79422bbba9e538c7b8dc5d8",
"src_2core_2_utils_8cpp.xhtml#aa6d4f0b9fedd979c5b768f9b34fda9f6",
"structarm__compute_1_1_f_f_t_radix_stage_kernel_info.xhtml#a3f305b6a19c882fc747c346aa7636bb2",
"structarm__compute_1_1experimental_1_1_memory_info.xhtml",
"structarm__compute_1_1test_1_1validation_1_1is__floating__point.xhtml",
"sve__merge__fp32__3_v_lx8_8hpp.xhtml",
"tests_2validation_2_c_l_2fft_8cpp_source.xhtml",
"utils_2_type_printer_8h.xhtml#a29cd1c3e1b33d61a479fa49465c5a7e7",
"warp__helpers_8h.xhtml#aba42442a4c991cdbb52727fa370676f1"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';