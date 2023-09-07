/*
 * Copyright (c) 2023 Arm Limited.
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

#include "arm_compute/runtime/CL/CLTensor.h"

#include "src/gpu/cl/kernels/ClMatMulLowpNativeMMULKernel.h"

#include "tests/datasets/LargeMatMulDataset.h"
#include "tests/datasets/SmallMatMulDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/MatMulKernelFixture.h"
#include "tests/validation/reference/Permute.h"

#include <tuple>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// TODO: enable
// constexpr AbsoluteTolerance<float> tolerance_quant(1); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
}
template <typename T>
using CLMatMulLowpNativeMMULKernelFixture = MatMulKernelValidationFixture<T, ClMatMulLowpNativeMMULKernel>;

template <typename T>
using CLMatMulLowpKernelWithBiasFixture = MatMulKernelWithBiasValidation<T, ClMatMulLowpNativeMMULKernel>;

TEST_SUITE(CL)
TEST_SUITE(MatMulLowpNativeMMULKernel)
TEST_SUITE(Validate)

TEST_CASE(SupportedKernelConfigurations, framework::DatasetMode::ALL)
{
    using MatMulConfigurationPair = std::pair<MatMulKernelInfo, bool>;

    const std::vector<MatMulConfigurationPair> supported_block_sizes =
    {
        // MatMulKernelInfo(adj_lhs, adj_rhs, M0, N0, K0, export_rhs_to_cl_image = false)
        // Lhs not-transposed, Rhs-not-transposed
        // TODO: Test Cases
    };

    // Set big enough shapes so that block sizes are not truncated. Also, set all dimensions equal
    // so that it doesn't fail for different NT/T configurations. We aim to test the block sizes here,
    // not the shapes themselves.
    const TensorInfo lhs_info = TensorInfo(TensorShape(100U, 100U), 1, DataType::QASYMM8_SIGNED);
    const TensorInfo rhs_info = TensorInfo(TensorShape(100U, 100U), 1, DataType::QASYMM8_SIGNED);

    for(auto &pair : supported_block_sizes)
    {
        TensorInfo output_info;
        Status     status = ClMatMulLowpNativeMMULKernel::validate(&lhs_info, &rhs_info, nullptr, &output_info, pair.first);

        ARM_COMPUTE_EXPECT(bool(status) == pair.second, framework::LogLevel::ERRORS);
    }
}

TEST_CASE(ValidateInputShapes, framework::DatasetMode::ALL)
{
    // Configurations are assumed to be Nt/Nt, but will be transposed inside the test to test other configurations
    using ShapeConfigurationTuple = std::tuple<TensorShape, TensorShape, TensorShape, bool>;
    const std::vector<ShapeConfigurationTuple> shape_configurations =
    {
        { TensorShape(5U, 1U), TensorShape(3U, 5U), TensorShape(3U), true },
        { TensorShape(10U, 12U), TensorShape(3U, 10U), TensorShape(3U), true },
        { TensorShape(8U, 4U), TensorShape(2U, 8U), TensorShape(2U), true },
        { TensorShape(8U, 4U), TensorShape(2U, 5U), TensorShape(2U), false }, // Mismatch in the K dimension
        { TensorShape(5U, 0U), TensorShape(2U, 5U), TensorShape(2U), false }, // Invalid dimension
        { TensorShape(5U, 4U, 3U, 4U, 5U, 6U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U), TensorShape(2U), true },
        { TensorShape(5U, 4U, 3U, 4U, 5U, 1U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U), TensorShape(2U), false }, // no batch broadcasting
        { TensorShape(5U, 4U, 3U, 4U, 9U, 6U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U), TensorShape(2U), false }, // mismatch in batch dimension
        { TensorShape(5U, 1U), TensorShape(3U, 5U), TensorShape(1U), false },                                 // invalid broadcast of bias
        { TensorShape(5U, 1U), TensorShape(3U, 5U), TensorShape(3U, 3U), false },                             // 2d bias is invalid
    };

    for(auto &tuple : shape_configurations)
    {
        const bool expected = std::get<3>(tuple);

        for(bool adj_lhs :
            {
                false, true
            })
        {
            for(bool adj_rhs :
                {
                    false, true
                })
            {
                TensorShape lhs_shape = std::get<0>(tuple);
                TensorShape rhs_shape = std::get<1>(tuple);
                TensorShape bia_shape = std::get<2>(tuple);

                if(adj_lhs)
                {
                    permute(lhs_shape, PermutationVector(1U, 0U));
                }

                if(adj_rhs)
                {
                    permute(rhs_shape, PermutationVector(1U, 0U));
                }

                const TensorInfo lhs_info = TensorInfo(lhs_shape, 1, DataType::QASYMM8_SIGNED);
                const TensorInfo rhs_info = TensorInfo(rhs_shape, 1, DataType::QASYMM8_SIGNED);
                const TensorInfo bia_info = TensorInfo(bia_shape, 1, DataType::S32);
                TensorInfo       output_info;

                MatMulKernelInfo matmul_kernel_info{ adj_lhs, adj_rhs, 1, 1, 1, false /* export_rhs_to_cl_image */ };

                Status status = ClMatMulLowpNativeMMULKernel::validate(&lhs_info, &rhs_info, &bia_info, &output_info, matmul_kernel_info);
                ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
            }
        }
    }
}

TEST_CASE(ValidateDataTypes, framework::DatasetMode::ALL)
{
    using DataTypeConfigurationTuple = std::tuple<DataType, DataType, DataType, DataType, bool>;
    const std::vector<DataTypeConfigurationTuple> data_type_configurations =
    {
        { DataType::F32, DataType::F32, DataType::F32, DataType::F32, false }, // no floating point types
        { DataType::F16, DataType::F16, DataType::F16, DataType::F16, false }, // no floating point types
        { DataType::F64, DataType::F64, DataType::F64, DataType::F64, false }, // no double precision
        { DataType::QASYMM8, DataType::QASYMM8, DataType::S32, DataType::QASYMM8, true },
        { DataType::QASYMM8_SIGNED, DataType::QASYMM8_SIGNED, DataType::S32, DataType::QASYMM8_SIGNED, true },
        { DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8_PER_CHANNEL, DataType::S32, DataType::QSYMM8_PER_CHANNEL, false }, // only qasymm8/qasymm8_signed is supported
        { DataType::QASYMM16, DataType::QASYMM16, DataType::S32, DataType::QASYMM16, false },                               // only qasymm8/qasymm8_signed is supported
        { DataType::QSYMM16, DataType::QSYMM16, DataType::S32, DataType::QSYMM16, false },                                  // only qasymm8/qasymm8_signed is supported
        { DataType::QSYMM8, DataType::QSYMM8, DataType::S32, DataType::QSYMM8, false },                                     // only qasymm8/qasymm8_signed is supported
        { DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S32, DataType::QASYMM8, false },                           // no mixed data types
        { DataType::S64, DataType::S64, DataType::S64, DataType::S64, false },                                              // no integral types
        { DataType::S32, DataType::S32, DataType::S32, DataType::S32, false },                                              // no integral types
        { DataType::S16, DataType::S16, DataType::S16, DataType::S16, false },                                              // no integral types
        { DataType::S8, DataType::S8, DataType::S8, DataType::S8, false },                                                  // no integral types
        { DataType::U64, DataType::U64, DataType::U64, DataType::U64, false },                                              // no integral types
        { DataType::U32, DataType::U32, DataType::U32, DataType::U32, false },                                              // no integral types
        { DataType::U16, DataType::U16, DataType::U16, DataType::U16, false },                                              // no integral types
        { DataType::U8, DataType::U8, DataType::U8, DataType::U8, false },                                                  // no integral types
        { DataType::QASYMM8, DataType::QASYMM8, DataType::F32, DataType::QASYMM8, false }                                   // Only S32 bias is supported
    };

    // It's enough to test a single shape and block size configuration while checking data types
    const TensorShape      shape     = TensorShape(48U, 48U);
    const TensorShape      bia_shape = TensorShape(48U);
    const MatMulKernelInfo matmul_kernel_info{ false, false, 1, 1, 1, false };
    for(auto &tuple : data_type_configurations)
    {
        const bool expected = std::get<4>(tuple);

        const TensorInfo lhs_info(shape, 1, std::get<0>(tuple));
        const TensorInfo rhs_info(shape, 1, std::get<1>(tuple));
        const TensorInfo bia_info(bia_shape, 1, std::get<2>(tuple));
        TensorInfo       output_info(shape, 1, std::get<3>(tuple));

        Status status = ClMatMulLowpNativeMMULKernel::validate(&lhs_info, &rhs_info, &bia_info, &output_info, matmul_kernel_info);
        ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
    }
}

TEST_SUITE_END() // Validate

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)

// TODO: tests

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QASYMM8)

// TODO: tests

TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // MatMulLowpNativeMMULKernel
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
