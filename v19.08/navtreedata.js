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
"_n_e_split_8h.xhtml",
"_open_c_l_8h_source.xhtml",
"_range_dataset_8h.xhtml",
"_stream_8h_source.xhtml",
"_validate_helpers_8h.xhtml#a40f232193e819dfc9dc9edbabefd5b7a",
"a64__transpose__interleave__12way__half__to__float_8hpp_source.xhtml",
"arm__compute_2core_2_types_8h.xhtml#a683661ae75dcb7aef16b9c9bde31517da4c5d06b02c97731aaa976179c62dcf76",
"arm__compute_2graph_2_types_8h_source.xhtml",
"benchmark_2_c_l_2_laplacian_reconstruct_8cpp.xhtml#a33ccd0c3fcaab9d1bb2c6a20bfd5ec6f",
"benchmark_2_n_e_o_n_2_convolution_8cpp.xhtml",
"bitwise__op_8cl.xhtml#a4b8ef6799be6362c31b39a159cd9f48b",
"classarm__compute_1_1_c_l_absolute_difference.xhtml#af53d66a8f8dd368d3c06b43c0c6a12f1",
"classarm__compute_1_1_c_l_comparison_static.xhtml",
"classarm__compute_1_1_c_l_derivative.xhtml",
"classarm__compute_1_1_c_l_g_e_m_m_convolution_layer.xhtml#a78f1fff174957ab8dd876ee696d5a749",
"classarm__compute_1_1_c_l_gaussian_pyramid_vert_kernel.xhtml",
"classarm__compute_1_1_c_l_locally_connected_matrix_multiply_kernel.xhtml#a0cc875c804d3f82bd6eba503ede72a86",
"classarm__compute_1_1_c_l_pooling_layer_kernel.xhtml#a1acfeaa60695d4df61d8d4b5c905aa53",
"classarm__compute_1_1_c_l_sobel7x7_hor_kernel.xhtml",
"classarm__compute_1_1_c_l_weights_reshape_kernel.xhtml#a32630202628891fb30458ef84581ad2a",
"classarm__compute_1_1_detection_output_layer_info.xhtml#af14fc4cf24dfb69a0f225a582ef01d54",
"classarm__compute_1_1_g_c_im2_col_kernel.xhtml#aa9f940db93f2843ff4694e64177e9b49",
"classarm__compute_1_1_h_o_g_info.xhtml#ac3f834a0c4744ccbc55d3b8bc810fdff",
"classarm__compute_1_1_i_g_c_tensor.xhtml#a15242f453f1928c3dbc2a30ec08480e6",
"classarm__compute_1_1_kernel.xhtml#a8b2bcac315357d36294d68948e51b925",
"classarm__compute_1_1_n_e_bitwise_not_kernel.xhtml#aed139c8b2173519c398b47c2cdddced3",
"classarm__compute_1_1_n_e_depth_convert_layer.xhtml#ad641b4160974dd335faa633d22d9ca7b",
"classarm__compute_1_1_n_e_f_f_t2_d.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_n_e_g_e_m_m_lowp_quantize_down_int32_to_int16_scale_by_fixed_point_kernel.xhtml#a95ba17151fa6b8b6771079441844e711",
"classarm__compute_1_1_n_e_integral_image_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_n_e_pixel_wise_multiplication.xhtml#a080bc209f47babf3bc587f49b4286e53",
"classarm__compute_1_1_n_e_sobel5x5_vert_kernel.xhtml#a59d1a2b1ef3d2b7e3bc5e942714d102a",
"classarm__compute_1_1_n_e_winograd_layer_transform_weights_kernel.xhtml#ae43e71907cc8f09be3338c84c18701bb",
"classarm__compute_1_1_sub_tensor_info.xhtml#a3504ac3cf390bdc4eadbc5dca081e07a",
"classarm__compute_1_1detail_1_1compare__dimension.xhtml",
"classarm__compute_1_1graph_1_1_graph_manager.xhtml#a7e6bc69b0ad4be79c10f72debc89dfc8",
"classarm__compute_1_1graph_1_1_tensor.xhtml#a8c1a8dc99f338372d7dc6dd33672ec03",
"classarm__compute_1_1graph_1_1frontend_1_1_normalize_planar_y_u_v_layer.xhtml",
"classarm__compute_1_1test_1_1_c_l_accessor.xhtml#a0189be0971a32a0171c51b90a1ab1020",
"classarm__compute_1_1test_1_1framework_1_1_framework.xhtml#a47fbbce86aea0ba4a9d539c069f840c9",
"classarm__compute_1_1test_1_1framework_1_1detail_1_1_test_suite_registrar.xhtml",
"classarm__gemm_1_1_gemv_batched.xhtml#abf5a58f6feffeae31f48b750cba9303d",
"depthwise__convolution_8cl.xhtml#ae40b0f25b3985d4853b944151ffddb44",
"fast__corners_8cl.xhtml#aea4e2a64dfdec6a9ecf62f8cd518fc17",
"gemm__helpers_8h.xhtml#a7249d30c01773ebe7a7c15899550967f",
"globals_k.xhtml",
"intrinsics_2add_8h.xhtml#a5facaa0c056dc73dfa6a49cde91240c4",
"namespacearm__compute_1_1test_1_1framework.xhtml",
"optical__flow__pyramid__lk_8cl_source.xhtml",
"reference_2_depth_convert_layer_8cpp.xhtml#a2790b7ea17fe89ea8befa5a0c657f4ae",
"repeat_8h.xhtml#affa3fc076d4917193cd8d0642def145d",
"src_2graph_2_utils_8cpp.xhtml#a1df15aed3ed531f442ecea2a131d65a4",
"structarm__compute_1_1_thread_info.xhtml",
"structarm__compute_1_1test_1_1traits_1_1promote.xhtml",
"tests_2_utils_8h.xhtml#a5351f49d449d49f82e2bf942b7a13da6",
"tests_2validation_2_helpers_8h.xhtml#a9a1aa7d40588cbc644eb7d45d008b1f0",
"utils_2_type_printer_8h.xhtml#ae8c2a3451bcf739a75aa7438e7a78d45",
"validation_2_c_l_2_depth_concatenate_layer_8cpp.xhtml#aed82cc7360b0d6831d7ce0ac9aa12c8b",
"validation_2_c_l_2_min_max_location_8cpp.xhtml#aa49dc5afff07b86be9d319b2947a10cd",
"validation_2_n_e_o_n_2_batch_normalization_layer_8cpp_source.xhtml",
"validation_2_n_e_o_n_2_h_o_g_multi_detection_8cpp.xhtml",
"validation_2reference_2_g_e_m_m_8cpp_source.xhtml"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';