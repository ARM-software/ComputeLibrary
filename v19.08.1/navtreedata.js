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
      [ "Import Memory Interface", "architecture.xhtml#S4_8_import_memory", null ],
      [ "OpenCL Tuner", "architecture.xhtml#S4_9_opencl_tuner", null ]
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
"_abs_layer_8cpp.xhtml",
"_c_l_2_arithmetic_division_8cpp.xhtml#ac91c705162f84cdbfdce90bdb135e2e4",
"_c_l_2_crop_resize_8cpp.xhtml#afcdff1aca709a6f9831f5592c24aba30",
"_c_l_2_gather_8cpp.xhtml#ac2ed31007ae463a3cec24a581f3651f6",
"_c_l_2_select_8cpp.xhtml#acb8fd2b1ee77b9b379853485bb429217",
"_c_l_absolute_difference_8cpp.xhtml",
"_c_l_deconvolution_reshape_output_kernel_8h_source.xhtml",
"_c_l_g_e_m_m_lowp_offset_contribution_output_stage_kernel_8h_source.xhtml",
"_c_l_magnitude_phase_kernel_8cpp_source.xhtml",
"_c_l_scharr3x3_kernel_8h_source.xhtml",
"_c_p_p_2_d_f_t_8cpp.xhtml#a78f90e18c925576e2bd4d196a72e201f",
"_convert_fully_connected_weights_8h.xhtml",
"_elementwise_square_diff_8cpp.xhtml#a6c0efec1f9ab32b14febc1f8fdd69bb3",
"_g_c_batch_normalization_layer_8cpp_source.xhtml",
"_g_e_m_m_lowp_8h.xhtml#a37da19bcb7b529afd0af2eb85edbb110",
"_helpers_8inl.xhtml",
"_i_tensor_handle_8h_source.xhtml",
"_n_e_asymm_8h.xhtml#a695a8105065167f7e4596d31f23a3573",
"_n_e_depthwise_vector_to_tensor_kernel_8cpp_source.xhtml",
"_n_e_g_e_m_m_interleave4x4_kernel_8h.xhtml",
"_n_e_median3x3_8h.xhtml",
"_n_e_o_n_2_channel_shuffle_8cpp.xhtml",
"_n_e_o_n_2_gather_8cpp.xhtml#ac2ed31007ae463a3cec24a581f3651f6",
"_n_e_o_n_2_reorg_layer_8cpp.xhtml#a32d10d3c62058cc25b4a8240b2c0db70",
"_n_e_range_8h.xhtml",
"_n_e_split_8cpp_source.xhtml",
"_open_c_l_8h.xhtml#af6e65f36e2be6bd9a1cac10ba6e82f7c",
"_range_8h_source.xhtml",
"_stream_8cpp_source.xhtml",
"_validate_helpers_8h.xhtml#a366fca5f1c0a2ed6ee50e9d619d4f2e5",
"a64__transpose__interleave__12way__16bit_8hpp_source.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a673665b4587a2956fcbad5f0e9ba89d3a80485e54c7a3c93a0f74637c6b918ce5",
"arm__compute_2graph_2_types_8h.xhtml#acac9cbaeea226ed297804c012dc12b16af2ee14b628c0a45e1682de8f33983dc1",
"benchmark_2_c_l_2_laplacian_pyramid_8cpp_source.xhtml",
"benchmark_2_n_e_o_n_2_color_convert_8cpp.xhtml#ad14a5471a936562e38e9443b3fc80dc9",
"bit__ops_8h_source.xhtml",
"classarm__compute_1_1_c_l_abs_layer.xhtml",
"classarm__compute_1_1_c_l_comparison_kernel.xhtml#a9eb86fbe4825b43b638171bb6dd398c3",
"classarm__compute_1_1_c_l_dequantization_layer_kernel.xhtml#ab475d3d6b3279119cb22ac054bbd0ec1",
"classarm__compute_1_1_c_l_g_e_m_m.xhtml#aebefa7576807b8b316ea46c242949039",
"classarm__compute_1_1_c_l_gaussian_pyramid_orb.xhtml",
"classarm__compute_1_1_c_l_locally_connected_layer.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_c_l_pixel_wise_multiplication_kernel.xhtml#a8de93f833e4e851a3867b100e32a56cd",
"classarm__compute_1_1_c_l_sobel5x5_vert_kernel.xhtml#ab6450493472e18b0f6696826464a07fd",
"classarm__compute_1_1_c_l_warp_affine_kernel.xhtml#aa6f6097800a87984c4fab8ab1fffd65e",
"classarm__compute_1_1_detection_output_layer_info.xhtml#a381583deeb7c92f3b86d959c1e6c8185",
"classarm__compute_1_1_g_c_g_e_m_m_transpose1x_w_kernel.xhtml#a8fd12b95bdde3f93db96bc9b1598db69",
"classarm__compute_1_1_h_o_g_info.xhtml#a5a4a2ab2f4396cc64b4a5676a90d7f07",
"classarm__compute_1_1_i_g_c_simple_kernel.xhtml#a81af259374ad902ad2b9e735b3b380be",
"classarm__compute_1_1_kernel.xhtml",
"classarm__compute_1_1_n_e_bitwise_not_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_n_e_depth_concatenate_layer_kernel.xhtml#ad6e98c5ccc23df7ba5781c8497e83947",
"classarm__compute_1_1_n_e_f_f_t1_d.xhtml",
"classarm__compute_1_1_n_e_g_e_m_m_lowp_quantize_down_int32_to_int16_scale_by_fixed_point.xhtml",
"classarm__compute_1_1_n_e_im2_col_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_permute_kernel.xhtml#a93c836ab36443b23753d99495761daf7",
"classarm__compute_1_1_n_e_sobel5x5_hor_kernel.xhtml#ab5656bb5b6334bdbe6e606c715872828",
"classarm__compute_1_1_n_e_winograd_layer_transform_weights_kernel.xhtml#a5dbb04cc8d65cafca5884f9c2c5a82b8",
"classarm__compute_1_1_sub_tensor.xhtml#af3913a3ffefed788b1fa3fb91407472a",
"classarm__compute_1_1cl__tuner_1_1_i_c_l_l_w_s_list.xhtml",
"classarm__compute_1_1graph_1_1_graph_context.xhtml#a9c24c14abfd0ab9d1f4ce8cd5cc20819",
"classarm__compute_1_1graph_1_1_tensor.xhtml#a0839be1697496b3a951e030627f9e69b",
"classarm__compute_1_1graph_1_1frontend_1_1_i_stream.xhtml#a75ce23fbe59cc58db22e1b6d072b10c6",
"classarm__compute_1_1test_1_1_assets_library.xhtml#ac4cb5f95f1d720ef0cc94b74152cf50b",
"classarm__compute_1_1test_1_1framework_1_1_framework.xhtml#a0fa5f7d6bcff8bfd18ec2aadf660a489",
"classarm__compute_1_1test_1_1framework_1_1dataset_1_1_zip_dataset.xhtml",
"classarm__gemm_1_1_gemv_batched.xhtml#a0b21ce5febc3f7eed9c7b37c0b9a3560",
"depthwise__convolution_8cl.xhtml#a128f47cb6aacde29e07fde2c4b9f5dd2",
"exp_8h.xhtml#a3d35fe714076b40450ba6ba870fae322",
"gemm__helpers_8h.xhtml#a6b46fdbc87c7084bdc63a2f9520ddb83",
"globals_func_t.xhtml",
"intrinsics_2add_8h.xhtml#a297de761eb0705e6059856e4a8e6e1d2",
"namespacearm__compute_1_1test_1_1benchmark_1_1_c_l_suite_1_1_n_i_g_h_t_l_y_suite.xhtml",
"optical__flow__pyramid__lk_8cl.xhtml#a3955a0fbedb5b3eae9a38801fb01bac5",
"reference_2_d_f_t_8cpp.xhtml#afc02bc2a9b06db328e63c6bfbbdb23a8",
"repeat_8h.xhtml#a88950092f54faf75d508703c229018e5",
"src_2core_2_utils_8cpp_source.xhtml",
"structarm__compute_1_1_optical_flow_parameters.xhtml#a2dceb63003a97fa431d06d1d07edc449",
"structarm__compute_1_1test_1_1framework_1_1dataset_1_1_singleton_dataset_1_1iterator.xhtml#aeeb0d03fad59d8f94bb8dea95825550e",
"tests_2_utils_8h.xhtml#a000a9b9acb976222ee72275cf8852a3f",
"tests_2validation_2_helpers_8h.xhtml",
"utils_2_type_printer_8h.xhtml#adac3c1908846c6361e664a353ad1c3cf",
"validation_2_c_l_2_convolution_layer_8cpp_source.xhtml",
"validation_2_c_l_2_min_max_location_8cpp.xhtml#a2fb5fe73a5c8287a2b91c01e3de7b41a",
"validation_2_n_e_o_n_2_batch_normalization_layer_8cpp.xhtml",
"validation_2_n_e_o_n_2_h_o_g_descriptor_8cpp.xhtml#a0abbec087553630b76284d3800c79602",
"validation_2reference_2_floor_8cpp.xhtml"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';