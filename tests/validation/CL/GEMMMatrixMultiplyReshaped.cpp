/*
 * Copyright (c) 2018-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeLhsMatrixKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::opencl::kernels;

// Create function for ClGemmReshapeLhsMatrixKernel
using CLGEMMReshapeLHSMatrix = CLSynthetizeOperator<ClGemmReshapeLhsMatrixKernel>;

// Create function for ClGemmReshapeRhsMatrixKernel
using CLGEMMReshapeRHSMatrix = CLSynthetizeOperator<ClGemmReshapeRhsMatrixKernel>;

// Create function for ClGemmMatrixMultiplyReshapedKernel
using CLGEMMMatrixMultiplyReshaped = CLSynthetizeOperator<ClGemmMatrixMultiplyReshapedKernel>;

// Fixture for CLGEMMMatrixMultiplyReshaped
template <typename T>
using CLGEMMMatrixMultiplyReshapedFixture = GEMMMatrixMultiplyReshapedValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped>;

// Fixture for CLGEMMMatrixMultiplyReshaped mixed precision
template <typename T>
using CLGEMMMatrixMultiplyReshapedMixedPrecisionFixture =
    GEMMMatrixMultiplyReshapedValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped, true>;

// Fixture for CLGEMMMatrixMultiplyReshaped3D
template <typename T>
using CLGEMMMatrixMultiplyReshaped3DFixture = GEMMMatrixMultiplyReshaped3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped>;

// Fixture for CLGEMMMatrixMultiplyReshaped3D mixed precision
template <typename T>
using CLGEMMMatrixMultiplyReshaped3DMixedPrecisionFixture =
    GEMMMatrixMultiplyReshaped3DValidationFixture<CLTensor, CLAccessor, T, CLGEMMReshapeLHSMatrix, CLGEMMReshapeRHSMatrix, CLGEMMMatrixMultiplyReshaped, true>;

namespace
{
// *INDENT-OFF*
// clang-format off
RelativeTolerance<float> rel_tolerance_f32(0.001f);
constexpr float          abs_tolerance_f32(0.0001f);

RelativeTolerance<float> rel_tolerance_f16_mixed_precision(0.001f);
constexpr float          abs_tolerance_f16_mixed_precision(0.01f);

RelativeTolerance<float> rel_tolerance_f16(0.001f);
constexpr float          abs_tolerance_f16(0.01f);

/** M values to test */
const auto m_values = framework::dataset::make("M", 17);

/** M_W values to test */
const auto m_w_values = framework::dataset::make("M_W", 5);

/** M_H values to test */
const auto m_h_values = framework::dataset::make("M_H", 7);

/** N values to test */
const auto n_values = framework::dataset::make("N", 21);

/** K values to test */
const auto k_values = framework::dataset::make("K", 13);

/** Batch size values to test */
const auto b_values = framework::dataset::make("batch_size", 2, 3);

/** Activation values to test */
const auto act_values = framework::dataset::make("Activation",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
});

/** Alpha values to test - Precommit */
const auto a_values_precommit = framework::dataset::make("alpha", {-0.75f} );

/** Beta values to test - Precommit */
const auto beta_values_precommit = framework::dataset::make("beta", {-0.35f} );

/** M0 values to test - Precommit */
const auto m0_values_precommit = framework::dataset::make("M0", { 4 });

/** N0 values to test - Precommit */
const auto n0_values_precommit = framework::dataset::make("N0", { 4 });

/** K0 values to test - Precommit */
const auto k0_values_precommit = framework::dataset::make("K0", { 4 });

/** V0 values to test - Precommit */
const auto v0_values_precommit = framework::dataset::make("V0", 1, 3);

/** H0 values to test - Precommit */
const auto h0_values_precommit = framework::dataset::make("H0", 1, 3);

/** Alpha values to test - Nightly */
const auto a_values_nightly = framework::dataset::make("alpha", {1.0f} );

/** Beta values to test - Nightly */
const auto beta_values_nightly = framework::dataset::make("beta", {1.0f} );

/** M0 values to test - Nightly */
const auto m0_values_nightly = framework::dataset::make("M0", { 8 });

/** N0 values to test - Nightly */
const auto n0_values_nightly = framework::dataset::make("N0", { 8 });

/** K0 values to test - Nightly */
const auto k0_values_nightly = framework::dataset::make("K0", { 4 });

/** N0 values to test with export to OpenCL image object - Nightly */
const auto n0_export_to_cl_image_values_nightly = framework::dataset::make("N0", { 4, 8, 16 });

/** K0 values to test with export to OpenCL image object - Nightly */
const auto k0_export_to_cl_image_values_nightly = framework::dataset::make("K0", { 4, 8, 16 });

/** V0 values to test - Nightly */
const auto v0_values_nightly = framework::dataset::make("V0", 1, 3);

/** H0 values to test - Nightly */
const auto h0_values_nightly = framework::dataset::make("H0", 1, 3);

/** Interleave values to test with LHS matrix */
const auto i_values_lhs = framework::dataset::make("interleave_lhs", { true, false });

/** Interleave values to test with RHS matrix */
const auto i_values_rhs = framework::dataset::make("interleave_rhs", { true, false });

/** Broadcast bias from vector to matrix */
const auto broadcast_bias_values = framework::dataset::make("broadcast_bias", { false, true } );

/** LHS transposed values */
const auto lhs_transpose_values = framework::dataset::make("lhs_transpose", { false, true } );

} // namespace

TEST_SUITE(CL)
TEST_SUITE(GEMMMatrixMultiplyReshaped)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("Input0Info", { TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::F32),      // OK
                                                        TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::F16),      // OK
                                                        TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::QASYMM8),  // Data type not supported
                                                        TensorInfo(TensorShape(10U, 5U, 2U), 1, DataType::F32),      // Incorrect dimension bias
                                                        TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::F32),      // Mismatching shapes
                                                        TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::F16),      // OK, do not broadcast bias
                                                        TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::F16),      // OK, wider accummulation
                                                        TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::F16),      // OK, RHS 4,4,2

                                                      }),
               framework::dataset::make("Input1Info",{ TensorInfo(TensorShape(64U, 6U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 6U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(64U, 5U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(64U, 6U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 6U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(64U, 6U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(128U, 3U, 2U), 1, DataType::F16),

                      })),
               framework::dataset::make("Input2Info", { TensorInfo(TensorShape(21U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(21U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(21U), 1, DataType::QASYMM8),
                                                        TensorInfo(TensorShape(21U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(21U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(21U,17U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(21U,17U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F16),

                                                      })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(21U,17U,2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(21U,17U,2U), 1, DataType::F16),

                           })),
               framework::dataset::make("LHSMInfo",{
                                                          GEMMLHSMatrixInfo(4,4,1,false,true),
                                                          GEMMLHSMatrixInfo(4,4,1,false,true),
                                                          GEMMLHSMatrixInfo(4,4,1,false,true),
                                                          GEMMLHSMatrixInfo(4,2,4,false,false),
                                                          GEMMLHSMatrixInfo(4,2,4,false,false),
                                                          GEMMLHSMatrixInfo(4,4,1,false,true),
                                                          GEMMLHSMatrixInfo(4,4,1,false,true),
                                                          GEMMLHSMatrixInfo(4,4,1,false,true),

                                })),
               framework::dataset::make("RHSMInfo",{
                                                          GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                          GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                          GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                          GEMMRHSMatrixInfo(2,2,1,true,false,false),
                                                          GEMMRHSMatrixInfo(2,2,1,true,false,false),
                                                          GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                          GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                          GEMMRHSMatrixInfo(4,4,2,true,false,false),


                           })),


               framework::dataset::make("GEMMInfo",{
                                                            GEMMKernelInfo( 17 /**<M Number of LHS rows*/,
                                                                            21 /**<N Number of RHS columns*/,
                                                                            13 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                                     false /**< reinterpret the input as 3D */,
                                                                     true  /**< Flag used to broadcast the bias addition */,
                                                                     false /**< wider accumm */,
                                                                     false /**< has pad y */,
                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                                     1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                                     1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                                     GEMMLHSMatrixInfo(4,4,1,false,true),
                                                                     GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                                     0  /**< Offset to be added to each element of the matrix A */,
                                                                     0 /**< Offset to be added to each element of the matrix B */),

                                                            GEMMKernelInfo( 17 /**<M Number of LHS rows*/,
                                                                            21 /**<N Number of RHS columns*/,
                                                                            13 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                                     false /**< reinterpret the input as 3D */,
                                                                     true  /**< Flag used to broadcast the bias addition */,
                                                                     false /**< wider accumm */,
                                                                     false /**< has pad y */,
                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                                     1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                                     1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                                     GEMMLHSMatrixInfo(4,4,1,false,true),
                                                                     GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                                     0  /**< Offset to be added to each element of the matrix A */,
                                                                     0 /**< Offset to be added to each element of the matrix B */),
                                                            GEMMKernelInfo(),
                                                            GEMMKernelInfo(),
                                                            GEMMKernelInfo(),

                                                            GEMMKernelInfo( 17 /**<M Number of LHS rows*/,
                                                                            21 /**<N Number of RHS columns*/,
                                                                            13 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                                     false /**< reinterpret the input as 3D */,
                                                                     false  /**< Flag used to broadcast the bias addition */,
                                                                     false /**< wider accumm */,
                                                                     false /**< has pad y */,
                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                                     1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                                     1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                                     GEMMLHSMatrixInfo(4,4,1,false,true),
                                                                     GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                                     0  /**< Offset to be added to each element of the matrix A */,
                                                                     0 /**< Offset to be added to each element of the matrix B */),


                                                            GEMMKernelInfo( 17 /**<M Number of LHS rows*/,
                                                                            21 /**<N Number of RHS columns*/,
                                                                            13 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                                     false /**< reinterpret the input as 3D */,
                                                                     false  /**< Flag used to broadcast the bias addition */,
                                                                     true /**< wider accumm */,
                                                                     true /**< has pad y */,
                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                                     1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                                     1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                                     GEMMLHSMatrixInfo(4,4,1,false,true),
                                                                     GEMMRHSMatrixInfo(4,4,1,true,true,false),
                                                                     0  /**< Offset to be added to each element of the matrix A */,
                                                                     0 /**< Offset to be added to each element of the matrix B */),

                                                            GEMMKernelInfo( 17 /**<M Number of LHS rows*/,
                                                                            21 /**<N Number of RHS columns*/,
                                                                            13 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                                     false /**< reinterpret the input as 3D */,
                                                                     false  /**< Flag used to broadcast the bias addition */,
                                                                     false /**< wider accumm */,
                                                                     false /**< has pad y */,
                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                                     1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                                     1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                                     GEMMLHSMatrixInfo(4,4,1,false,true),
                                                                     GEMMRHSMatrixInfo(4,4,2,true,false,false),
                                                                     0  /**< Offset to be added to each element of the matrix A */,
                                                                     0 /**< Offset to be added to each element of the matrix B */),
                                                    })),
               framework::dataset::make("Expected", { true, true, false, false, false, true, true,true})),
                    input0_info ,input1_info, input2_info, output_info, lhs_info, rhs_info, gemm_info, expected)
{
   ARM_COMPUTE_EXPECT(bool(ClGemmMatrixMultiplyReshapedKernel::validate(&input0_info.clone()->set_is_resizable(true),
                                                          &input1_info.clone()->set_is_resizable(true),
                                                          &input2_info.clone()->set_is_resizable(true),
                                                          &output_info.clone()->set_is_resizable(true),1.f,1.f,
                                                          lhs_info,
                                                          rhs_info,
                                                          gemm_info)) == expected, framework::LogLevel::ERRORS);
}
TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE(ExportToCLImage)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("Input0Info", { TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),  // OK or incorrect if cl_khr_image2d_from_buffer not supported
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),  // OK or incorrect if cl_khr_image2d_from_buffer not supported
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),  // OK or incorrect if cl_khr_image2d_from_buffer not supported
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),  // Incorrect k0
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),  // Incorrect n0

                                                      }),
               framework::dataset::make("Input1Info",{ TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(512U, 8U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(128U, 32U, 2U), 1, DataType::F32),

                      })),
               framework::dataset::make("Input2Info", { TensorInfo(TensorShape(64U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F32),

                                                      })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F32),

                           })),
               framework::dataset::make("LHSMInfo",{
                                                          GEMMLHSMatrixInfo(4, 4, 1, false, true),
                                                          GEMMLHSMatrixInfo(4, 8, 1, false, true),
                                                          GEMMLHSMatrixInfo(4, 4, 1, false, true),
                                                          GEMMLHSMatrixInfo(4, 2, 1, false, false),
                                                          GEMMLHSMatrixInfo(4, 4, 1, false, false),

                                })),
               framework::dataset::make("RHSMInfo",{
                                                          GEMMRHSMatrixInfo(4, 4, 1, true, true, true),
                                                          GEMMRHSMatrixInfo(4, 8, 1, true, true, true),
                                                          GEMMRHSMatrixInfo(8, 4, 1, true, true, true),
                                                          GEMMRHSMatrixInfo(4, 2, 1, true, false, true),
                                                          GEMMRHSMatrixInfo(2, 4, 1, true, false, true),
                           })),
               framework::dataset::make("GEMMInfo",{GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),
                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),
                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),

                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),
                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */)
                                                    })),
               framework::dataset::make("Expected", { true,
                                                      true,
                                                      true,
                                                      false,
                                                      false})),
                    input0_info ,input1_info, input2_info, output_info, lhs_info, rhs_info, gemm_info, expected)
{
   ARM_COMPUTE_EXPECT(bool(ClGemmMatrixMultiplyReshapedKernel::validate(&input0_info.clone()->set_is_resizable(true),
                                                          &input1_info.clone()->set_is_resizable(true),
                                                          &input2_info.clone()->set_is_resizable(true),
                                                          &output_info.clone()->set_is_resizable(true),1.f,1.f,
                                                          lhs_info,
                                                          rhs_info,
                                                          gemm_info)) == (expected && image2d_from_buffer_supported(CLKernelLibrary::get().get_device())), framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
     // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }

}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_export_to_cl_image_values_nightly),
                                                                   k0_export_to_cl_image_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
     // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
     // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<float>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_export_to_cl_image_values_nightly),
                                                                   k0_export_to_cl_image_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // ExportToCLImage
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
}

TEST_SUITE(ExportToCLImage)
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
               framework::dataset::make("Input0Info", { TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),  // OK or incorrect if cl_khr_image2d_from_buffer not supported
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),  // OK or incorrect if cl_khr_image2d_from_buffer not supported
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),  // OK or incorrect if cl_khr_image2d_from_buffer not supported
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),  // Incorrect k0
                                                        TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),  // Incorrect n0

                                                      }),
               framework::dataset::make("Input1Info",{ TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(512U, 8U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(256U, 16U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(128U, 32U, 2U), 1, DataType::F16),

                      })),
               framework::dataset::make("Input2Info", { TensorInfo(TensorShape(64U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(64U), 1, DataType::F16),

                                                      })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(64U, 64U, 2U), 1, DataType::F16),

                           })),
               framework::dataset::make("LHSMInfo",{
                                                          GEMMLHSMatrixInfo(4, 4, 1, false, true),
                                                          GEMMLHSMatrixInfo(4, 8, 1, false, true),
                                                          GEMMLHSMatrixInfo(4, 4, 1, false, true),
                                                          GEMMLHSMatrixInfo(4, 2, 1, false, false),
                                                          GEMMLHSMatrixInfo(4, 4, 1, false, false),

                                })),
               framework::dataset::make("RHSMInfo",{
                                                          GEMMRHSMatrixInfo(4, 4, 1, true, true, true),
                                                          GEMMRHSMatrixInfo(4, 8, 1, true, true, true),
                                                          GEMMRHSMatrixInfo(8, 4, 1, true, true, true),
                                                          GEMMRHSMatrixInfo(4, 2, 1, true, false, true),
                                                          GEMMRHSMatrixInfo(2, 4, 1, true, false, true),
                           })),
               framework::dataset::make("GEMMInfo",{GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),
                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),
                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),

                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */),
                                                    GEMMKernelInfo( 64 /**<M Number of LHS rows*/,
                                                                    64 /**<N Number of RHS columns*/,
                                                                    64 /**<K Number of LHS columns or RHS rows */, 0 /**< Depth of the output tensor in case is reinterpreted as 3D */,
                                                             false /**< reinterpret the input as 3D */,
                                                             true  /**< Flag used to broadcast the bias addition */,
                                                             false /**< wider accumm */,
                                                             false /**< has pad y */,
                                                           ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                             1   /**< Multiplication factor for the width of the 1xW transposed block */,
                                                             1   /**< Multiplication factor for the height of the 4x4 interleaved block */,
                                                             GEMMLHSMatrixInfo(),
                                                             GEMMRHSMatrixInfo(),
                                                             0  /**< Offset to be added to each element of the matrix A */,
                                                             0 /**< Offset to be added to each element of the matrix B */)
                                                    })),
               framework::dataset::make("Expected", { true,
                                                      true,
                                                      true,
                                                      false,
                                                      false})),
                    input0_info ,input1_info, input2_info, output_info, lhs_info, rhs_info, gemm_info, expected)
{
   ARM_COMPUTE_EXPECT(bool(ClGemmMatrixMultiplyReshapedKernel::validate(&input0_info.clone()->set_is_resizable(true),
                                                          &input1_info.clone()->set_is_resizable(true),
                                                          &input2_info.clone()->set_is_resizable(true),
                                                          &output_info.clone()->set_is_resizable(true),1.f,1.f,
                                                          lhs_info,
                                                          rhs_info,
                                                          gemm_info)) == (expected && image2d_from_buffer_supported(CLKernelLibrary::get().get_device())), framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }

}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_export_to_cl_image_values_nightly),
                                                                   k0_export_to_cl_image_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DFixture<half>, framework::DatasetMode::NIGHTLY,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_export_to_cl_image_values_nightly),
                                                                   k0_export_to_cl_image_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", true)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output only if validate() is successful
    if(validate_result)
    {
        validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.f, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_khr_image2d_from_buffer not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // ExportToCLImage
TEST_SUITE_END() // FP16

TEST_SUITE(MixedPrecision)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMMatrixMultiplyReshapedMixedPrecisionFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMMatrixMultiplyReshapedMixedPrecisionFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_values,
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   broadcast_bias_values),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}

FIXTURE_DATA_TEST_CASE(RunSmall3D, CLGEMMMatrixMultiplyReshaped3DMixedPrecisionFixture<half>, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_precommit),
                                                                   n0_values_precommit),
                                                                   k0_values_precommit),
                                                                   v0_values_precommit),
                                                                   h0_values_precommit),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_precommit),
                                                                   beta_values_precommit),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}

FIXTURE_DATA_TEST_CASE(RunLarge3D, CLGEMMMatrixMultiplyReshaped3DMixedPrecisionFixture<half>, framework::DatasetMode::DISABLED,
                combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                   m_w_values,
                                                                   m_h_values),
                                                                   n_values),
                                                                   k_values),
                                                                   b_values),
                                                                   m0_values_nightly),
                                                                   n0_values_nightly),
                                                                   k0_values_nightly),
                                                                   v0_values_nightly),
                                                                   h0_values_nightly),
                                                                   i_values_lhs),
                                                                   i_values_rhs),
                                                                   framework::dataset::make("export_to_cl_image_rhs", false)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                   a_values_nightly),
                                                                   beta_values_nightly),
                                                                   lhs_transpose_values),
                                                                   act_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16_mixed_precision, 0.f, abs_tolerance_f16_mixed_precision);
}
TEST_SUITE_END() // MixedPrecision
TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMMMatrixMultiplyReshaped
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
