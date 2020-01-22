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
"_shape_calculator_8h.xhtml#acedb0877d41f2ae0591a2d4e84318140",
"_utility_8h.xhtml#a0c1a72cd100958036bb59f0625f0613f",
"a64__gemm__u16__12x8_2generic_8cpp_source.xhtml",
"arm__compute_2core_2_helpers_8h.xhtml#a6174495b626531de015ae2b810859287",
"arm__compute_2core_2_utils_8h.xhtml#a84437d80241f6a31e1a07c231ee8e3ac",
"benchmark_2_c_l_2_convolution_layer_8cpp.xhtml#a22fb86f77a20d298ea600a2d24713c5e",
"benchmark_2_g_l_e_s___c_o_m_p_u_t_e_2_batch_normalization_layer_8cpp_source.xhtml",
"benchmark_2_n_e_o_n_2_harris_corners_8cpp.xhtml#a1aa05c5a190c320fbb0e259c9af430d2",
"channel__extract_8cl.xhtml#aeecb8084159259d10790df3e0a602cbf",
"classarm__compute_1_1_c_l_arithmetic_subtraction.xhtml",
"classarm__compute_1_1_c_l_convolution3x3.xhtml",
"classarm__compute_1_1_c_l_distribution1_d.xhtml#a1b056f14381ab6eda09ed227a3d9e9e5",
"classarm__compute_1_1_c_l_g_e_m_m_lowp_matrix_multiply_reshaped_only_r_h_s_kernel.xhtml#aa7967bee001197b8642e30d4efaa96cc",
"classarm__compute_1_1_c_l_h_o_g_gradient.xhtml#a229bdf53bc1dad3b2408d48db8bcaa96",
"classarm__compute_1_1_c_l_lut_allocator.xhtml#a65d09435ca3956fb0c4409e622bfd5da",
"classarm__compute_1_1_c_l_quantization_layer_kernel.xhtml#a493987e85723a8000eb26d1f00e2ad0e",
"classarm__compute_1_1_c_l_softmax_layer_generic.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_c_l_width_concatenate2_tensors_kernel.xhtml#a9474659ca4a4bf972731b0465dd28afd",
"classarm__compute_1_1_detection_post_process_layer_info.xhtml#a634bf1faedaec956983af4d41940239e",
"classarm__compute_1_1_g_c_g_e_m_m_transpose1x_w_kernel.xhtml",
"classarm__compute_1_1_generate_proposals_info.xhtml#a2a3e74a86ee9c520cc1a43191145edf6",
"classarm__compute_1_1_i_g_c_memory_region.xhtml#a3d9d196240e9b0062c953c3abd534187",
"classarm__compute_1_1_i_tensor_allocator.xhtml#af3461a6d7c86f09c854cbb7ab5578c4e",
"classarm__compute_1_1_n_e_arithmetic_addition_kernel.xhtml#ad336181c06f2c2983212849bd9ebcd35",
"classarm__compute_1_1_n_e_convert_quantized_signedness_kernel.xhtml#a83a344e60eb7db895953a942abf16628",
"classarm__compute_1_1_n_e_division_operation_kernel.xhtml#afc96fa047235667233e73836c97aa774",
"classarm__compute_1_1_n_e_g_e_m_m_convolution_layer.xhtml#a0a7030af14bb803820435ca75009b0f9",
"classarm__compute_1_1_n_e_h_o_g_detector_kernel.xhtml#a2c0f4bf7e37cb04bf2384f4bc71b84d5",
"classarm__compute_1_1_n_e_min_max_layer_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_n_e_reshape_layer.xhtml",
"classarm__compute_1_1_n_e_tile_kernel.xhtml#a158e0205c27d5d36f5353f7ade72040d",
"classarm__compute_1_1_prior_box_layer_info.xhtml#a9752ea0a44b131caa9384231944a2b8d",
"classarm__compute_1_1_tensor_info.xhtml#ab54246abe670b06f5624add7e7022904",
"classarm__compute_1_1graph_1_1_detection_output_layer_node.xhtml",
"classarm__compute_1_1graph_1_1_in_place_operation_mutator.xhtml",
"classarm__compute_1_1graph_1_1backends_1_1_g_c_device_backend.xhtml#a5100fa4901700c5ce71b142e4b1b8d14",
"classarm__compute_1_1graph__utils_1_1_i_preprocessor.xhtml",
"classarm__compute_1_1test_1_1_g_c_accessor.xhtml#a9a3e72153aeb3ed212e9c3698774e881",
"classarm__compute_1_1test_1_1framework_1_1_j_s_o_n_printer.xhtml#aa520a585d43bb30a6c483e69d40f01a1",
"classarm__compute_1_1utils_1_1_common_graph_validate_options.xhtml#a770e3e82cb8e4632d754e651fd29b9cb",
"classarm__gemm_1_1_gemm_hybrid_quantized.xhtml#ae9fc7380e533c43a1d91e48a7578e412",
"core_2_c_l_2_c_l_helpers_8cpp.xhtml#ac07e02c0066cf540a5a2665fa7d54934",
"direct__convolution3x3_8cl.xhtml#ad3cc858846806e6b1d3694b9d0a2e6da",
"functions_z.xhtml",
"gemv__native__transposed_8hpp_source.xhtml",
"helpers__asymm_8h.xhtml#a1d51b02a83af2a152fa52755f572f5a6",
"magnitude__phase_8cl.xhtml",
"namespacemembers_f.xhtml",
"pmin_8h.xhtml#a14642064b6d61c95588ff03d2bf3e522",
"reference_2_gaussian5x5_8cpp.xhtml#ae74885d96f07f242e6974979cce5673b",
"runtime_2_tensor_8cpp.xhtml",
"src_2core_2_c_l_2cl__kernels_2_helpers_8h.xhtml#acfabbc4d3d33d2e66a69e0dbe42309f2",
"structarm__compute_1_1_g_e_m_m_lowp_output_stage_info.xhtml#a6e019ad85979fd73c74f97e5483faf35",
"structarm__compute_1_1test_1_1framework_1_1_measurement_1_1_value.xhtml#a82dc1e06c88a4412fd40f1750487c2ae",
"sve__hybrid__fp16__mla__4_v_lx4_2generic_8cpp.xhtml",
"tests_2framework_2_macros_8h.xhtml#a8b3c06c9e7676202a34f711b1a7625fc",
"tests_2validation_2reference_2_utils_8h.xhtml#a4f489943d8618d47b1ad4611f0b9b7ff",
"validation_2_c_l_2_batch_normalization_layer_8cpp.xhtml#a4da3a5c737246cbb15c2c55c8b9f4f0f",
"validation_2_c_l_2_fast_corners_8cpp.xhtml#a23837c7359608bfada46c0d1d6500818",
"validation_2_c_l_2_quantization_layer_8cpp_source.xhtml",
"validation_2_n_e_o_n_2_convolution_8cpp.xhtml#aae364968da28604884452ee304580304",
"validation_2_n_e_o_n_2_median3x3_8cpp.xhtml#a4dee97faf3f5b0b27bbe0f37753c9f5b",
"validation_2reference_2_integral_image_8cpp.xhtml#a6d4691ec28061cbb6e38600f5f78a9bf"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';