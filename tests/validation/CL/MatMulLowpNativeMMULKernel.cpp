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

#include "tests/datasets/MatMulLowpMMULDataset.h"
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
constexpr AbsoluteTolerance<float> tolerance_quant(1); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
}
using framework::dataset::make;

template <typename T>
using CLMatMulLowpNativeMMULKernelFixture = MatMulKernelValidationFixture<T, ClMatMulLowpNativeMMULKernel, true /* use_mmul */>;

template <typename T>
using CLMatMulLowpNativeMMULKernelWithBiasFixture = MatMulKernelWithBiasValidation<T, ClMatMulLowpNativeMMULKernel, true /* use_mmul */>;

/** M0 values to test --precommit*/
const auto m0_values_precommit = framework::dataset::make("M0", { 1, 3 });

/** N0 values to test --precommit*/
const auto n0_values_precommit = framework::dataset::make("N0", { 2, 4 });

/** M0 values to test --nightly*/
const auto m0_values_nightly_lhs_nt = framework::dataset::make("M0", { 2, 4, 5, 8 });
const auto m0_values_nightly_lhs_t  = framework::dataset::make("M0", { 2, 4, 8 });

/** N0 values to test --nightly*/
const auto n0_values_nightly = framework::dataset::make("N0", { 1, 3, 8, 16 });

TEST_SUITE(CL)
TEST_SUITE(MatMulLowpNativeMMULKernel)
TEST_SUITE(Validate)

TEST_CASE(SupportedKernelConfigurations, framework::DatasetMode::ALL)
{
    using MatMulConfigurationPair = std::pair<MatMulKernelInfo, bool>;

    const std::vector<MatMulConfigurationPair> supported_block_sizes =
    {
        // MatMulKernelInfo(adj_lhs, adj_rhs, M0, N0, K0, export_rhs_to_cl_image = false)
        { MatMulKernelInfo(false, false, 0, 1, 4), false }, // M0 should be > 0
        { MatMulKernelInfo(false, true, 3, 5, 4), false },  // N0 not in {1, 2, 3, 4, 8, 16}
        { MatMulKernelInfo(false, false, 3, 6, 4), false }, // N0 not in {1, 2, 3, 4, 8, 16}
        { MatMulKernelInfo(false, false, 3, 3, 8), false }, // K0 not in 4
        { MatMulKernelInfo(true, false, 5, 3, 4), false },  // M0 not in {1, 2, 3, 4, 8, 16} when Lhs is transposed
        { MatMulKernelInfo(false, false, 9, 1, 4), true },
        { MatMulKernelInfo(false, true, 3, 16, 4), true },
        { MatMulKernelInfo(false, false, 7, 3, 4), true },
        { MatMulKernelInfo(true, false, 8, 3, 4), true },
        { MatMulKernelInfo(true, true, 4, 3, 4), true },
        { MatMulKernelInfo(false, false, 7, 3, 4, true), false }, // export to CLImage is unsupported for quantized types
    };

    // Set big enough shapes so that block sizes are not truncated. Also, set all dimensions equal
    // so that it doesn't fail for different NT/T configurations. We aim to test the block sizes here,
    // not the shapes themselves.
    const TensorInfo lhs_info = TensorInfo(TensorShape(64U, 64U), 1, DataType::QASYMM8_SIGNED);
    const TensorInfo rhs_info = TensorInfo(TensorShape(64U, 64U), 1, DataType::QASYMM8_SIGNED);

    for(auto &pair : supported_block_sizes)
    {
        TensorInfo output_info;
        Status     status   = ClMatMulLowpNativeMMULKernel::validate(&lhs_info, &rhs_info, nullptr, &output_info, pair.first);
        const bool expected = (pair.second && arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()));

        ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
    }
}

TEST_CASE(ValidateInputShapes, framework::DatasetMode::ALL)
{
    // Configurations are assumed to be Nt/Nt, but will be transposed inside the test to test other configurations
    using ShapeConfigurationTuple = std::tuple<TensorShape, TensorShape, TensorShape, bool>;
    const std::vector<ShapeConfigurationTuple> shape_configurations =
    {
        { TensorShape(32U, 1U), TensorShape(3U, 32U), TensorShape(3U), true },
        { TensorShape(16U, 12U), TensorShape(3U, 16U), TensorShape(3U), true },
        { TensorShape(64U, 4U), TensorShape(2U, 64U), TensorShape(2U), true },
        { TensorShape(16U, 4U), TensorShape(2U, 32U), TensorShape(2U), false }, // Mismatch in the K dimension
        { TensorShape(16U, 0U), TensorShape(2U, 16U), TensorShape(2U), false }, // Invalid dimension
        { TensorShape(32U, 4U, 3U, 4U, 5U, 6U), TensorShape(2U, 32U, 3U, 4U, 5U, 6U), TensorShape(2U), true },
        { TensorShape(32U, 4U, 3U, 4U, 5U, 1U), TensorShape(2U, 32U, 3U, 4U, 5U, 6U), TensorShape(2U), false }, // no batch broadcasting
        { TensorShape(32U, 4U, 3U, 4U, 9U, 6U), TensorShape(2U, 32U, 3U, 4U, 5U, 6U), TensorShape(2U), false }, // mismatch in batch dimension
        { TensorShape(32U, 1U), TensorShape(3U, 32U), TensorShape(1U), false },                                 // invalid broadcast of bias
        { TensorShape(32U, 1U), TensorShape(3U, 32U), TensorShape(3U, 3U), false },                             // 2d bias is invalid
        { TensorShape(12U, 12U), TensorShape(3U, 12U), TensorShape(3U), false },                                // K must be multiple of 16
    };

    for(auto &tuple : shape_configurations)
    {
        const bool expected = (std::get<3>(tuple) && arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()));

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

                MatMulKernelInfo matmul_kernel_info{ adj_lhs, adj_rhs, 1, 1, 4, false /* export_rhs_to_cl_image */ };

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
    const MatMulKernelInfo matmul_kernel_info{ false, false, 1, 1, 4, false };
    for(auto &tuple : data_type_configurations)
    {
        const bool expected = (std::get<4>(tuple) && arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()));

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

FIXTURE_DATA_TEST_CASE(RunSmall, CLMatMulLowpNativeMMULKernelFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallMatMulLowpMMULDataset(),
                               make("TransposeA", { false, true }),
                               make("TransposeB", { false, true }),
                               m0_values_precommit,
                               n0_values_precommit,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8_SIGNED)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

FIXTURE_DATA_TEST_CASE(RunWithBias, CLMatMulLowpNativeMMULKernelWithBiasFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallMatMulLowpMMULWithBiasDataset(),
                               make("TransposeA", { false, true }),
                               make("TransposeB", { false, true }),
                               m0_values_precommit,
                               n0_values_precommit,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8_SIGNED)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

FIXTURE_DATA_TEST_CASE(RunLargeLhsNotTransposed, CLMatMulLowpNativeMMULKernelFixture<int8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeMatMulLowpMMULDataset(),
                               make("TransposeA", { false }),
                               make("TransposeB", { false, true }),
                               m0_values_nightly_lhs_nt,
                               n0_values_nightly,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8_SIGNED)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

FIXTURE_DATA_TEST_CASE(RunLargeLhsTransposed, CLMatMulLowpNativeMMULKernelFixture<int8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeMatMulLowpMMULDataset(),
                               make("TransposeA", { true }),
                               make("TransposeB", { false, true }),
                               m0_values_nightly_lhs_t,
                               n0_values_nightly,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8_SIGNED)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

// Running High Dimensional test is enough for qasymm8_signed, because we're stressing the number of dimensions, not data type or M0/N0/K0
// It's a good idea to test for each Lhs/Rhs T/NT combinations because they're different CL kernels
FIXTURE_DATA_TEST_CASE(RunHighDimensional, CLMatMulLowpNativeMMULKernelFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::HighDimensionalMatMulLowpMMULDataset(),
                               make("TransposeA", { false, true }),
                               make("TransposeB", { false, true }),
                               make("M0", { 2 }),
                               make("N0", { 2 }),
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8_SIGNED)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QASYMM8)

FIXTURE_DATA_TEST_CASE(RunSmall, CLMatMulLowpNativeMMULKernelFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallMatMulLowpMMULDatasetSubset(),
                               make("TransposeA", { false, true }),
                               make("TransposeB", { false, true }),
                               m0_values_precommit,
                               n0_values_precommit,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

FIXTURE_DATA_TEST_CASE(RunWithBias, CLMatMulLowpNativeMMULKernelWithBiasFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallMatMulLowpMMULWithBiasDataset(),
                               make("TransposeA", { false, true }),
                               make("TransposeB", { false, true }),
                               m0_values_precommit,
                               n0_values_precommit,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

FIXTURE_DATA_TEST_CASE(RunLargeLhsNotTransposed, CLMatMulLowpNativeMMULKernelFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeMatMulLowpMMULDataset(),
                               make("TransposeA", { false }),
                               make("TransposeB", { false, true }),
                               m0_values_nightly_lhs_nt,
                               n0_values_nightly,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

FIXTURE_DATA_TEST_CASE(RunLargeLhsTransposed, CLMatMulLowpNativeMMULKernelFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeMatMulLowpMMULDataset(),
                               make("TransposeA", { true }),
                               make("TransposeB", { false, true }),
                               m0_values_nightly_lhs_t,
                               n0_values_nightly,
                               make("K0", { 4 }),
                               make("ExportRhsToCLImage", { false }),
                               make("DataType", DataType::QASYMM8)))
{
    if(_device_supports_mmul)
    {
        // Validate output
        validate(CLAccessor(_target), _reference, tolerance_quant);
    }
}

TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // MatMulLowpNativeMMULKernel
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
