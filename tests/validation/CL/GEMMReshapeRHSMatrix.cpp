/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMReshapeRHSMatrixFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// *INDENT-OFF*
// clang-format off
/** Data types */
const auto data_types = framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F16, DataType::F32 });

/** Batch size values to test */
const auto b_values = framework::dataset::make("batchsize", 1, 3);

/** N0 values to test */
const auto n0_values_nt_s32 = framework::dataset::make("N0", { 1, 2, 3 });
const auto n0_values_nt_s16 = framework::dataset::make("N0", { 4, 8 });
const auto n0_values_nt_s8 = framework::dataset::make("N0", { 16 });
const auto n0_values_t_s32 = framework::dataset::make("N0", { 4, 8 });
const auto n0_values_t_s16 = framework::dataset::make("N0", { 16 });
const auto n0_values_t_s8 = framework::dataset::make("N0", { 2, 3 });

/** K0 values to test */
const auto k0_values_nt_s32 = framework::dataset::make("K0", { 1, 2 });
const auto k0_values_nt_s16 = framework::dataset::make("K0", { 16 });
const auto k0_values_nt_s8 = framework::dataset::make("K0", { 3,4 });
const auto k0_values_t_s32 = framework::dataset::make("K0", { 2, 3 });
const auto k0_values_t_s16 = framework::dataset::make("K0", { 4, 8 });
const auto k0_values_t_s8 = framework::dataset::make("K0", { 16 });

/** H0 values to test */
const auto h0_values = framework::dataset::make("H0", 1, 4);

/** Interleave values to test */
const auto i_values = framework::dataset::make("interleave", { true, false });
} // namespace

using namespace arm_compute::misc::shape_calculator;

// Initialize the output tensor with zero and fill the border with zero
using CLGEMMReshapeRHSMatrix = CLSynthetizeFunctionInitOutputWithZeroAndWithZeroConstantBorder<CLGEMMReshapeRHSMatrixKernel, 16>;

template <typename T>
using CLGEMMReshapeRHSMatrixFixture = GEMMReshapeRHSMatrixValidationFixture<CLTensor, CLAccessor, CLGEMMReshapeRHSMatrix, T>;

TEST_SUITE(CL)
TEST_SUITE(GEMMReshapeRHSMatrix)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),    // Wrong n0 value
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),    // Wrong k0 value
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),    // Wrong h0 value
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),    // n0 > 16
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),    // k0 > 16
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),    // k0 == 1 && transpose
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(64U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 2U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 2U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 2U, 2U), 1, DataType::F32),
                                                     })),
                framework::dataset::make("N0",{ 4, 0, 4, 4, 4, 17, 4, 4 })),
                framework::dataset::make("K0",{ 4, 4, 0, 4, 4, 4, 17, 1 })),
                framework::dataset::make("H0",{ 4, 4, 4, 0, 4, 4, 4, 4 })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false})),
               input_info, output_info, n0, k0, h0, expected)
{
    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0 = n0;
    rhs_info.k0 = k0;
    rhs_info.h0 = h0;
    rhs_info.transpose = true;
    rhs_info.interleave = true;

    bool has_error = bool(CLGEMMReshapeRHSMatrixKernel::validate(&input_info.clone()->set_is_resizable(false), (output_info.total_size() == 0) ? nullptr : &output_info.clone()->set_is_resizable(false), rhs_info));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(ValidatePadding, framework::DatasetMode::ALL, combine(combine(combine(
               framework::dataset::make("InputShape", { TensorShape(32U, 16U, 1U),
                                                        TensorShape(32U, 16U, 2U)
                                                     }),
                framework::dataset::make("N0",{ 4 })),
                framework::dataset::make("K0",{ 4, 8, 16 })),
                framework::dataset::make("H0",{ 1, 2, 4 })),
               input_shape, n0, k0, h0)
{
    CLTensor input;
    CLTensor output;

    input.info()->init(input_shape, 1, DataType::F32);

    unsigned int padding = 0;

    GEMMRHSMatrixInfo rhs_info;
    rhs_info.n0 = n0;
    rhs_info.k0 = k0;
    rhs_info.h0 = h0;
    rhs_info.transpose = true;
    rhs_info.interleave = true;
    rhs_info.export_to_cl_image = image2d_from_buffer_supported(CLKernelLibrary::get().get_device()) && (get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device()) != 0);

    if(rhs_info.export_to_cl_image)
    {
        TensorShape output_shape = compute_rhs_reshaped_shape(*input.info(), rhs_info);
        constexpr unsigned int num_floats_per_pixel = 4;

        const unsigned int pixel_aligment      = get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device());
        const unsigned int row_pitch_alignment = pixel_aligment * num_floats_per_pixel;
        const unsigned int round_up_width      = ((output_shape[0] + row_pitch_alignment - 1) / row_pitch_alignment) * row_pitch_alignment;

        padding = round_up_width - output_shape[0];
    }

    CLGEMMReshapeRHSMatrixKernel kernel;

    kernel.configure(&input, &output, rhs_info);

    ARM_COMPUTE_EXPECT((output.info()->padding().right == padding), framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

// Run S32 tests only for transpose = false
FIXTURE_DATA_TEST_CASE(S32_NT, CLGEMMReshapeRHSMatrixFixture<int>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                               b_values),
                                                                       framework::dataset::make("DataType", DataType::S32)),
                                                               n0_values_nt_s32),
                                                       k0_values_nt_s32),
                                               h0_values),
                                       i_values),
                               framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// Run S32 tests only for transpose = true
FIXTURE_DATA_TEST_CASE(S32_T, CLGEMMReshapeRHSMatrixFixture<int>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                               b_values),
                                                                       framework::dataset::make("DataType", DataType::S32)),
                                                               n0_values_t_s32),
                                                       k0_values_t_s32),
                                               h0_values),
                                       i_values),
                               framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// Run S16 tests only for transpose = false
FIXTURE_DATA_TEST_CASE(S16_NT, CLGEMMReshapeRHSMatrixFixture<short>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                               b_values),
                                                                       framework::dataset::make("DataType", DataType::S16)),
                                                               n0_values_nt_s16),
                                                       k0_values_nt_s16),
                                               h0_values),
                                       i_values),
                               framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// Run S16 tests only for transpose = true
FIXTURE_DATA_TEST_CASE(S16_T, CLGEMMReshapeRHSMatrixFixture<short>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                               b_values),
                                                                       framework::dataset::make("DataType", DataType::S16)),
                                                               n0_values_t_s16),
                                                       k0_values_t_s16),
                                               h0_values),
                                       i_values),
                               framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// Run S8 tests only for transpose = false
FIXTURE_DATA_TEST_CASE(S8_NT, CLGEMMReshapeRHSMatrixFixture<char>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                               b_values),
                                                                       framework::dataset::make("DataType", DataType::S8)),
                                                               n0_values_nt_s8),
                                                       k0_values_nt_s8),
                                               h0_values),
                                       i_values),
                               framework::dataset::make("transpose", false)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

// Run S8 tests only for transpose = true
FIXTURE_DATA_TEST_CASE(S8_T, CLGEMMReshapeRHSMatrixFixture<char>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(datasets::SmallGEMMReshape2DShapes(),
                                                                               b_values),
                                                                       framework::dataset::make("DataType", DataType::S8)),
                                                               n0_values_t_s8),
                                                       k0_values_t_s8),
                                               h0_values),
                                       i_values),
                               framework::dataset::make("transpose", true)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // GEMMReshapeRHSMatrix
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
