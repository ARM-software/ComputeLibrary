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
"_c_l_2_arithmetic_subtraction_8cpp_source.xhtml",
"_c_l_2_elementwise_min_8cpp.xhtml#a8438b034546f24d3d38a7758e6c88abe",
"_c_l_2_reduction_operation_8cpp.xhtml#a6a5f79b26e653290a01b9b4d931ff268",
"_c_l_2_winograd_8cpp.xhtml#a9a6f542aa33954cc0c391ebc079bc0ef",
"_c_l_convolution_kernel_8h.xhtml#ac4cfbdf439d7bf5d420546298bd5ca0d",
"_c_l_g_e_m_m_deconvolution_layer_8h_source.xhtml",
"_c_l_laplacian_pyramid_8h_source.xhtml",
"_c_l_reverse_kernel_8h_source.xhtml",
"_c_p_p_2_d_f_t_8cpp.xhtml#a03c574d14dd0623ae30cf069c53a7f25",
"_color_convert_helper_8h.xhtml",
"_elementwise_operations_8cpp_source.xhtml",
"_g_c_batch_normalization_layer_8cpp.xhtml",
"_g_e_m_m_interleave_blocked_8h_source.xhtml",
"_i_array_8h.xhtml#a32e4b9083329486a06960006af89a84c",
"_instruments_8cpp.xhtml",
"_n_e_bitwise_or_8h.xhtml",
"_n_e_direct_convolution_detail_8h.xhtml#a93001798dca50920585380cf5ae07a75",
"_n_e_gather_8cpp.xhtml",
"_n_e_o_n_2_arg_min_max_8cpp.xhtml#a38fe4b20a05bbaa1c844f3d7a19791ae",
"_n_e_o_n_2_dilated_convolution_layer_8cpp.xhtml#a3af28ab49b5bab61e9ab1d819b2bc076",
"_n_e_o_n_2_reduce_mean_8cpp.xhtml#a13b0c3565b30f4234d850c64a2a6ea89",
"_n_e_r_o_i_pooling_layer_kernel_8cpp_source.xhtml",
"_n_e_stack_layer_8cpp_source.xhtml",
"_open_g_l_e_s_8cpp.xhtml#a1c5dc7cc04f7c0b1bd7c181b0da84d9b",
"_reverse_8h_source.xhtml",
"_toolchain_support_8h.xhtml#aebc1ea57c14482b68dac0c0dab766126a027219c7702304969c898aeffed504af",
"_window_iterator_8cpp.xhtml#acd2ef152b1743c15de3f67238d0eac43",
"architecture.xhtml#S4_7_2_working_with_memory_manager",
"arm__compute_2core_2_utils_8h.xhtml#aa2f22cc01532e0236e438324310fdb94",
"benchmark_2_c_l_2_depth_concatenate_layer_8cpp.xhtml",
"benchmark_2_g_l_e_s___c_o_m_p_u_t_e_2_depthwise_convolution_layer_8cpp.xhtml#ae809ef8d55ba29d3a7636231607183ef",
"benchmark_2_n_e_o_n_2_laplacian_pyramid_8cpp.xhtml#ad85f4086eab6de98194f326a4b635450",
"classarm__compute_1_1_access_window_rectangle.xhtml#aa8bd157ea64c2b37dec8b035b1b78a07",
"classarm__compute_1_1_c_l_canny_edge.xhtml",
"classarm__compute_1_1_c_l_depth_concatenate_layer_kernel.xhtml#ac0005a72b0a709d73612b1f7b8118d3e",
"classarm__compute_1_1_c_l_fill_border_kernel.xhtml",
"classarm__compute_1_1_c_l_g_e_m_m_reshape_l_h_s_matrix_kernel.xhtml#aa65a9cf45dc5c7c2c96fd57ecc8d2ec8",
"classarm__compute_1_1_c_l_l2_normalize_layer.xhtml#afc0747175608c02e3f6e74ef1c573f50",
"classarm__compute_1_1_c_l_normalize_planar_y_u_v_layer_kernel.xhtml#afda135e29153ae1cc8946bf974427189",
"classarm__compute_1_1_c_l_sobel3x3_kernel.xhtml#a453a89015ac4f8c719f40d98a24de478",
"classarm__compute_1_1_c_l_upsample_layer_kernel.xhtml",
"classarm__compute_1_1_detection_output_layer_info.xhtml#a025a49ad16e9d5d59d3919c25a17d1ae",
"classarm__compute_1_1_g_c_im2_col_kernel.xhtml#a8fd12b95bdde3f93db96bc9b1598db69",
"classarm__compute_1_1_h_o_g_info.xhtml#aeaa7d619922de47d6239b0167a58e2c7",
"classarm__compute_1_1_i_g_c_tensor.xhtml#a95acb432c46c1f3a6b77eb88944e6c02",
"classarm__compute_1_1_l_s_t_m_params.xhtml#a35e4b6311397e1f9532fb37560aa9996",
"classarm__compute_1_1_n_e_box3x3_kernel.xhtml",
"classarm__compute_1_1_n_e_depthwise_convolution_layer.xhtml#aa9b93ef660fc3c5b4b19d3fc7b891b77",
"classarm__compute_1_1_n_e_fill_array_kernel.xhtml#a1885e2bee760c881e4d8bc5747bb3200",
"classarm__compute_1_1_n_e_g_e_m_m_transpose1x_w_kernel.xhtml#a83a344e60eb7db895953a942abf16628",
"classarm__compute_1_1_n_e_logits1_d_softmax_kernel.xhtml#a112b35dd205c62ea6ed1447ef226da82",
"classarm__compute_1_1_n_e_reduction_operation.xhtml#ad1717410afd0be936c6213a63c8005fb",
"classarm__compute_1_1_n_e_threshold.xhtml",
"classarm__compute_1_1_pooling_layer_info.xhtml#a825657ba220662927b90d1ca2c19fbd7",
"classarm__compute_1_1_tensor_shape.xhtml#a8e15e87871211f98c2b566137e38ef99",
"classarm__compute_1_1graph_1_1_edge.xhtml#ad00e584f78c622e5ee9ec9613f6d6633",
"classarm__compute_1_1graph_1_1_pooling_layer_node.xhtml#a65d13dc93e2df5e8ab725263cf9f4ac5",
"classarm__compute_1_1graph_1_1backends_1_1_n_e_tensor_handle.xhtml#a5d9a543899a9f7c93a950a1d080f2437",
"classarm__compute_1_1logging_1_1_thread_id_decorator.xhtml",
"classarm__compute_1_1test_1_1_simple_tensor.xhtml#af6124c81d1e81f182d64ae76caa3fa52",
"classarm__compute_1_1test_1_1framework_1_1_test_error.xhtml#a01ff177012854b9a35d86825a3a8eabd",
"classarm__gemm_1_1_gemm_hybrid.xhtml#ae385dce1a98e03f0a3ca311f031e5110",
"depthwise__convolution_8cl.xhtml#a0916b921e5c01cc64afede6dc7bd5caa",
"fft_8cl.xhtml#a008d11872b90493790f933f82c9f05b5",
"getlow_8h.xhtml#a6b68d30ce46c4c3194066823acacd442",
"histogram_8cl.xhtml",
"namespacearm__compute.xhtml",
"non__linear__filter__helpers_8h.xhtml#ae3e09114bb887de4e20122eed42671a1",
"reference_2_copy_8cpp.xhtml#ad9000ce99b9ffcec5722cade36d7e757",
"repeat_8h.xhtml#a25dcf84375f5bef322aba053e054986d",
"src_2graph_2_utils_8cpp.xhtml#aaef0102f19fc08d670766506b1a0b66e",
"structarm__compute_1_1_valid_region.xhtml#a1fcd64682b37ed3c2098d0094ce788d8",
"structarm__compute_1_1test_1_1traits_1_1promote_3_01uint32__t_01_4.xhtml#a9659bbcc7fc016eda242219021c7980b",
"tests_2benchmark_2_c_l_2_reshape_layer_8cpp_source.xhtml",
"tests_2validation_2_n_e_o_n_2fft_8cpp.xhtml#aedc846191741fb759460952ef3dbf6d2",
"utils_8hpp.xhtml#a8f6fbf8b243a10af40ce8d47a1013384",
"validation_2_c_l_2_direct_convolution_layer_8cpp.xhtml#a6af2459356e4644f2683ca2dd4b74f01",
"validation_2_c_l_2_scale_8cpp.xhtml#a27d2376584b1bfbce2de6c046d119d00",
"validation_2_n_e_o_n_2_convolution_layer_8cpp.xhtml",
"validation_2_n_e_o_n_2_phase_8cpp.xhtml#ad3aca7b8dd427c7a78e479e41a7cb6a7",
"validation_2reference_2_phase_8cpp.xhtml"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';