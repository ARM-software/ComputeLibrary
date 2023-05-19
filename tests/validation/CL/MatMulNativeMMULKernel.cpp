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
#include "src/gpu/cl/kernels/ClMatMulNativeMMULKernel.h"
#include "tests/datasets/LargeMatMulMMULDataset.h"
#include "tests/datasets/SmallMatMulMMULDataset.h"
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
RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float          abs_tolerance_f32(
    0.0001f); /**< Absolute tolerance value for comparing reference's output against implementation's output for floating point data types in case using relative tolerance fails because of small values */
constexpr float abs_tolerance_f16(
    0.001f);                                                   /**< Absolute tolerance value for comparing reference's output against implementation's output for fp16  data types in case using relative tolerance fails because of small values */
RelativeTolerance<half_float::half> tolerance_f16(half(0.01)); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
} // namespace

/** M0 values to test --precommit*/
const auto m0_values_precommit = framework::dataset::make("M0", { 1, 3 });

/** N0 values to test --precommit*/
const auto n0_values_precommit = framework::dataset::make("N0", { 2, 4 });

/** M0 values to test --nightly*/
const auto m0_values_nightly_lhs_nt = framework::dataset::make("M0", { 1, 2, 3, 4, 5, 6, 7, 8 });

/** N0 values to test --nightly*/
const auto n0_values_nightly_rhs_nt = framework::dataset::make("N0", { 1, 2, 3, 4, 8, 16 });

/** K0 value -- Fixed to 1 */
const auto k0_value = framework::dataset::make("K0", { 1 });

template <typename T>
using CLMatMulNativeMMULKernelFixture = MatMulKernelValidationFixture<T, ClMatMulNativeMMULKernel, true /*use_mmul*/>;

TEST_SUITE(CL)
TEST_SUITE(MatMulNativeMMULKernel)
TEST_SUITE(Validate)

TEST_CASE(SupportedBlockSizes, framework::DatasetMode::ALL)
{
    if(arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()))
    {
        using MatMulConfigurationPair = std::pair<MatMulKernelInfo, bool>;

        const std::vector<MatMulConfigurationPair> supported_block_sizes =
        {
            // MatMulKernelInfo(adj_lhs, adj_rhs, M0, N0, K0, export_rhs_to_cl_image = false)
            // Lhs not-transposed, Rhs-not-transposed
            { MatMulKernelInfo(false, false, 0, 1, 1), false }, // M0 should be > 0
            { MatMulKernelInfo(false, false, 3, 5, 1), false }, // N0 not in {1, 2, 3, 4, 8, 16}
            { MatMulKernelInfo(false, false, 3, 6, 1), false }, // N0 not in {1, 2, 3, 4, 8, 16}
            { MatMulKernelInfo(false, false, 3, 3, 4), false }, // K0 not 1
            { MatMulKernelInfo(false, false, 9, 1, 2), true },
            { MatMulKernelInfo(false, false, 3, 16, 3), true },
            { MatMulKernelInfo(false, false, 7, 3, 4), true },

            // Lhs not-transposed, Rhs transposed
            // TODO: COMPMID-6195

            // Lhs transposed, Rhs-not-transposed
            // TODO: COMPMID-6196

            // Lhs transposed, Rhs-transposed
            // TODO: COMPMID-6197
        };

        // Set big enough shapes so that block sizes are not truncated. Also, set all dimensions equal
        // so that it doesn't fail for different NT/T configurations. We aim to test the block sizes here,
        // not the shapes themselves.
        const TensorInfo lhs_info = TensorInfo(TensorShape(100U, 100U), 1, DataType::F32);
        const TensorInfo rhs_info = TensorInfo(TensorShape(100U, 100U), 1, DataType::F32);

        for(auto &pair : supported_block_sizes)
        {
            TensorInfo output_info;
            Status     status = ClMatMulNativeMMULKernel::validate(&lhs_info, &rhs_info, &output_info, pair.first);
        }
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_CASE(ValidateInputShapes, framework::DatasetMode::ALL)
{
    if(arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()))
    {
        // Configurations are assumed to be Nt/Nt, but will be transposed inside the test to test other configurations
        using ShapeConfigurationTuple = std::tuple<TensorShape, TensorShape, bool>;
        const std::vector<ShapeConfigurationTuple> shape_configurations =
        {
            { TensorShape(4U, 1U), TensorShape(3U, 4U), true },
            { TensorShape(12U, 12U), TensorShape(3U, 12U), true },
            { TensorShape(8U, 4U), TensorShape(2U, 8U), true },
            { TensorShape(8U, 4U), TensorShape(2U, 4U), false }, // Mismatch in the K dimension
            { TensorShape(5U, 0U), TensorShape(2U, 5U), false }, // Invalid dimension
            { TensorShape(5U, 7U), TensorShape(2U, 5U), false }, // K not a multiple of 4 (MMUL_K0)
            { TensorShape(8U, 4U, 3U, 4U, 5U, 6U), TensorShape(2U, 8U, 3U, 4U, 5U, 6U), true },
            { TensorShape(5U, 4U, 3U, 4U, 5U, 1U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U), false }, // No batch broadcasting
            { TensorShape(5U, 4U, 3U, 4U, 9U, 6U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U), false }, // Mismatch in batch dimension
        };

        for(auto &tuple : shape_configurations)
        {
            const bool expected = std::get<2>(tuple);

            for(bool adj_lhs :
                {
                    false // TODO: COMPMID-6195, COMPMID-6196, COMPMID-6197
                })
            {
                for(bool adj_rhs :
                    {
                        false // TODO: COMPMID-6195, COMPMID-6196, COMPMID-6197
                    })
                {
                    TensorShape lhs_shape = std::get<0>(tuple);
                    TensorShape rhs_shape = std::get<1>(tuple);

                    if(adj_lhs)
                    {
                        permute(lhs_shape, PermutationVector(1U, 0U));
                    }

                    if(adj_rhs)
                    {
                        permute(rhs_shape, PermutationVector(1U, 0U));
                    }

                    const TensorInfo lhs_info = TensorInfo(lhs_shape, 1, DataType::F32);
                    const TensorInfo rhs_info = TensorInfo(rhs_shape, 1, DataType::F32);
                    TensorInfo       output_info;

                    MatMulKernelInfo matmul_kernel_info{ adj_lhs, adj_rhs, 1, 1, 1, false /* export_rhs_to_cl_image */ };

                    Status status = ClMatMulNativeMMULKernel::validate(&lhs_info, &rhs_info, &output_info, matmul_kernel_info);
                    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
                }
            }
        }
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_CASE(ValidateDataTypes, framework::DatasetMode::ALL)
{
    if(arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()))
    {
        // Configurations are assumed to be Nt/Nt, but will be transposed inside the test to test other configurations
        using DataTypeConfigurationTuple = std::tuple<DataType, DataType, DataType, bool>;
        const std::vector<DataTypeConfigurationTuple> data_type_configurations =
        {
            { DataType::F32, DataType::F32, DataType::F32, true },
            { DataType::F16, DataType::F16, DataType::F16, true },
            { DataType::F16, DataType::F32, DataType::F32, false },                                              // no mixed precision
            { DataType::F64, DataType::F64, DataType::F64, false },                                              // no double precision
            { DataType::QASYMM8, DataType::QASYMM8, DataType::QASYMM8, false },                                  // no quantized types
            { DataType::QASYMM8_SIGNED, DataType::QASYMM8_SIGNED, DataType::QASYMM8_SIGNED, false },             // no quantized types
            { DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8_PER_CHANNEL, false }, // no quantized types
            { DataType::QASYMM16, DataType::QASYMM16, DataType::QASYMM16, false },                               // no quantized types
            { DataType::QSYMM16, DataType::QSYMM16, DataType::QSYMM16, false },                                  // no quantized types
            { DataType::QSYMM8, DataType::QSYMM8, DataType::QSYMM8, false },                                     // no quantized types
            { DataType::S64, DataType::S64, DataType::S64, false },                                              // no integral types
            { DataType::S32, DataType::S32, DataType::S32, false },                                              // no integral types
            { DataType::S16, DataType::S16, DataType::S16, false },                                              // no integral types
            { DataType::S8, DataType::S8, DataType::S8, false },                                                 // no integral types
            { DataType::U64, DataType::U64, DataType::U64, false },                                              // no integral types
            { DataType::U32, DataType::U32, DataType::U32, false },                                              // no integral types
            { DataType::U16, DataType::U16, DataType::U16, false },                                              // no integral types
            { DataType::U8, DataType::U8, DataType::U8, false },                                                 // no integral types
        };

        const TensorShape      shape = TensorShape(8U, 8U);
        const MatMulKernelInfo matmul_kernel_info{ false, false, 1, 1, 1, false };
        for(auto &tuple : data_type_configurations)
        {
            const bool expected = std::get<3>(tuple);

            const TensorInfo lhs_info(shape, 1, std::get<0>(tuple));
            const TensorInfo rhs_info(shape, 1, std::get<1>(tuple));
            TensorInfo       output_info(shape, 1, std::get<2>(tuple));

            Status status = ClMatMulNativeMMULKernel::validate(&lhs_info, &rhs_info, &output_info, matmul_kernel_info);
            ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
        }
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("cl_arm_matrix_multiply not supported. TEST skipped");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_SUITE_END() // Validate

TEST_SUITE(Float)
TEST_SUITE(FP32)
TEST_SUITE(Buffer)
FIXTURE_DATA_TEST_CASE(RunTiny, CLMatMulNativeMMULKernelFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::TinyMatMulMMULDataset(),
                                                                                                                     framework::dataset::make("TransposeA", { false })),
                                                                                                                     framework::dataset::make("TransposeB", { false })),
                                                                                                                     m0_values_precommit),
                                                                                                                     n0_values_precommit),
                                                                                                                     k0_value),
                                                                                                                     framework::dataset::make("ExportRhsToCLImage", { false })),
                                                                                                             framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    if(_device_supports_mmul)
    {
        validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
    }
}
FIXTURE_DATA_TEST_CASE(RunSmall, CLMatMulNativeMMULKernelFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallMatMulMMULDataset(),
                                                                                                                      framework::dataset::make("TransposeA", { false })),
                                                                                                                      framework::dataset::make("TransposeB", { false })),
                                                                                                                      m0_values_precommit),
                                                                                                                      n0_values_precommit),
                                                                                                                      k0_value),
                                                                                                                      framework::dataset::make("ExportRhsToCLImage", { false })),
                                                                                                              framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    if(_device_supports_mmul)
    {
        validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLMatMulNativeMMULKernelFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(datasets::LargeMatMulMMULDataset(),
                                                                                                                  framework::dataset::make("TransposeA", { false })),
                                                                                                                  framework::dataset::make("TransposeB", { false })),
                                                                                                                  m0_values_nightly_lhs_nt),
                                                                                                                  n0_values_nightly_rhs_nt),
                                                                                                                  k0_value),
                                                                                                                  framework::dataset::make("ExportRhsToCLImage", { false })),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    if(_device_supports_mmul)
    {
        validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
    }
}
// Running High Dimensional test is enough for FP32, because we're stressing the number of dimensions, not data type or M0/N0/K0
// It's a good idea to test for each Lhs/Rhs T/NT combinations because they're different CL kernels
FIXTURE_DATA_TEST_CASE(RunHighDimensional, CLMatMulNativeMMULKernelFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(datasets::HighDimensionalMatMulMMULDataset(),
                                                                               framework::dataset::make("TransposeA", { false })),
                                                                       framework::dataset::make("TransposeB", { false })),
                                                               framework::dataset::make("M0", { 2 })),
                                                       framework::dataset::make("N0", { 2 })),
                                               framework::dataset::make("K0", { 1 })),
                                       framework::dataset::make("ExportRhsToCLImage", { false })),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    if(_device_supports_mmul)
    {
        validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
    }
}
TEST_SUITE_END() // Buffer

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
TEST_SUITE(Buffer)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMatMulNativeMMULKernelFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(combine(combine(combine(datasets::SmallMatMulMMULDataset(),
                                                                                                                     framework::dataset::make("TransposeA", { false })),
                                                                                                                     framework::dataset::make("TransposeB", { false })),
                                                                                                                     m0_values_precommit),
                                                                                                                     n0_values_precommit),
                                                                                                                     k0_value),
                                                                                                                     framework::dataset::make("ExportRhsToCLImage", { false })),
                                                                                                             framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    if(_device_supports_mmul)
    {
        validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLMatMulNativeMMULKernelFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(datasets::LargeMatMulMMULDataset(),
                                                                                                                 framework::dataset::make("TransposeA", { false })),
                                                                                                                 framework::dataset::make("TransposeB", { false })),
                                                                                                                 m0_values_nightly_lhs_nt),
                                                                                                                 n0_values_nightly_rhs_nt),
                                                                                                                 k0_value),
                                                                                                                 framework::dataset::make("ExportRhsToCLImage", { false })),
                                                                                                                 framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    if(_device_supports_mmul)
    {
        validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
    }
}
TEST_SUITE_END() // Buffer

TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // MatMulNativeMMULKernel
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
