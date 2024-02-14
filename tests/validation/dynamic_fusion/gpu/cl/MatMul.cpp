/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#ifdef ACL_INTERNAL_TEST_CKW_IN_DF
#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/LargeMatMulDataset.h"
#include "tests/datasets/MatMulDataset.h"
#include "tests/datasets/SmallMatMulDataset.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/validation/fixtures/dynamic_fusion/gpu/cl/MatMulKernelFixture.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/Validation.h"

#include <tuple>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_f32(
    0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float abs_tolerance_f32(
    0.0001f); /**< Absolute tolerance value for comparing reference's output against implementation's output for floating point data types in case using relative tolerance fails because of small values */
constexpr float abs_tolerance_f16(
    0.001f); /**< Absolute tolerance value for comparing reference's output against implementation's output for fp16  data types in case using relative tolerance fails because of small values */
RelativeTolerance<half_float::half> tolerance_f16(half(
    0.02)); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
} // namespace

/** M0 values to test - precommit */
const auto m0_values_lhs_nt_precommit = framework::dataset::make("M0", {1, 2, 3});

/** N0 values to test - precommit */
const auto n0_values_rhs_t_precommit = framework::dataset::make("N0", {1, 2, 4});

/** K0 values to test - precommit */
const auto k0_values_rhs_t_precommit = framework::dataset::make("K0", {1, 2, 4});

/** M0 values to test - nightly */
const auto m0_values_lhs_nt_nightly = framework::dataset::make("M0", {1, 2, 3, 4});

/** N0 values to test - nightly */
const auto n0_values_rhs_t_nightly = framework::dataset::make("N0", {1, 2, 3, 4, 8});

/** K0 values to test - nightly */
const auto k0_values_rhs_t_nightly = framework::dataset::make("K0", {1, 2, 3, 4, 8});

class DFMatMulDataset final : public datasets::MatMulDataset
{
public:
    DFMatMulDataset()
    {
        // LHS = [K, M], RHS = [N, K], DST = [N, M]
        add_config(TensorShape(1U, 1U), TensorShape(1U, 1U), TensorShape(1U, 1U));
        add_config(TensorShape(1U, 2U), TensorShape(2U, 1U), TensorShape(2U, 2U));
        add_config(TensorShape(9U, 6U), TensorShape(5U, 9U), TensorShape(5U, 6U));
        add_config(TensorShape(32U, 37U), TensorShape(17U, 32U), TensorShape(17U, 37U));
    }
};

TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)

TEST_SUITE(MatMul)

TEST_SUITE(Validate)
TEST_CASE(SupportedBlockSizes, framework::DatasetMode::ALL)
{
    using MatMulConfigurationPair = std::pair<MatMulKernelInfo, bool>;

    const std::vector<MatMulConfigurationPair> supported_block_sizes = {
        // MatMulKernelInfo(adj_lhs, adj_rhs, M0, N0, K0, export_rhs_to_cl_image = false)

        // Lhs not-transposed, Rhs transposed
        {MatMulKernelInfo(false, true, 0, 1, 1), false},  // M0 should be > 0
        {MatMulKernelInfo(false, true, 3, 11, 1), false}, // N0 not in {1, 2, 3, 4, 8, 16}
        {MatMulKernelInfo(false, true, 3, 7, 1), false},  // N0 not in {1, 2, 3, 4, 8, 16}
        {MatMulKernelInfo(false, true, 3, 3, 12), false}, // K0 not in {1, 2, 3, 4, 8, 16}
        {MatMulKernelInfo(false, true, 3, 3, 6), false},  // K0 not in {1, 2, 3, 4, 8, 16}
        {MatMulKernelInfo(false, true, 5, 1, 2), true},   {MatMulKernelInfo(false, true, 3, 3, 3), true},
        {MatMulKernelInfo(false, true, 2, 4, 8), true},

    };

    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{&cl_compile_ctx};
    GpuWorkloadSketch sketch{&context};

    // Set big enough shapes so that block sizes are not truncated. Also, set all dimensions equal
    // so that it doesn't fail for different NT/T configurations. We aim to test the block sizes here,
    // not the shapes themselves.
    const ITensorInfo *lhs_info = context.create_tensor_info(TensorInfo(TensorShape(100U, 100U), 1, DataType::F32));
    const ITensorInfo *rhs_info = context.create_tensor_info(TensorInfo(TensorShape(100U, 100U), 1, DataType::F32));

    for (auto &pair : supported_block_sizes)
    {
        MatMulAttributes matmul_attr{};
        matmul_attr.adj_lhs(pair.first.adj_lhs);
        matmul_attr.adj_rhs(pair.first.adj_rhs);

        GpuMatMulSettings matmul_settings{};
        matmul_settings.m0(pair.first.m0);
        matmul_settings.n0(pair.first.n0);
        matmul_settings.k0(pair.first.k0);

        Status status = GpuMatMul::validate_op(sketch, lhs_info, rhs_info, matmul_attr, matmul_settings);
        ARM_COMPUTE_EXPECT(bool(status) == pair.second, framework::LogLevel::ERRORS);
    }
}

TEST_CASE(ValidateInputShapes, framework::DatasetMode::ALL)
{
    // Create a sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{&cl_compile_ctx};
    GpuWorkloadSketch sketch{&context};

    // Configurations are assumed to be Nt/Nt, but will be transposed inside the test to test other configurations
    using ShapeConfigurationTuple                                   = std::tuple<TensorShape, TensorShape, bool>;
    const std::vector<ShapeConfigurationTuple> shape_configurations = {
        {TensorShape(5U, 1U), TensorShape(3U, 5U), true},
        {TensorShape(10U, 12U), TensorShape(3U, 10U), true},
        {TensorShape(8U, 4U), TensorShape(2U, 8U), true},
        {TensorShape(8U, 4U), TensorShape(2U, 5U), false}, // Mismatch in the K dimension
        {TensorShape(5U, 0U), TensorShape(2U, 5U), false}, // Invalid dimension
        {TensorShape(5U, 4U, 3U, 4U, 5U, 6U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U), true},
        {TensorShape(5U, 4U, 3U, 4U, 5U, 1U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U), false}, // no batch broadcasting
        {TensorShape(5U, 4U, 3U, 4U, 9U, 6U), TensorShape(2U, 5U, 3U, 4U, 5U, 6U),
         false}, // mismatch in batch dimension
    };

    for (auto &tuple : shape_configurations)
    {
        const bool expected = std::get<2>(tuple);

        for (bool adj_lhs : {false})
        {
            for (bool adj_rhs : {true})
            {
                TensorShape lhs_shape = std::get<0>(tuple);
                TensorShape rhs_shape = std::get<1>(tuple);

                if (adj_lhs)
                {
                    permute(lhs_shape, PermutationVector(1U, 0U));
                }

                if (adj_rhs)
                {
                    permute(rhs_shape, PermutationVector(1U, 0U));
                }

                const ITensorInfo *lhs_info = context.create_tensor_info(TensorInfo(lhs_shape, 1, DataType::F32));
                const ITensorInfo *rhs_info = context.create_tensor_info(TensorInfo(rhs_shape, 1, DataType::F32));

                MatMulAttributes matmul_attr{};
                matmul_attr.adj_lhs(adj_lhs);
                matmul_attr.adj_rhs(adj_rhs);

                GpuMatMulSettings matmul_settings{};
                matmul_settings.m0(1);
                matmul_settings.n0(1);
                matmul_settings.k0(1);

                Status status = GpuMatMul::validate_op(sketch, lhs_info, rhs_info, matmul_attr, matmul_settings);
                ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
            }
        }
    }
}

TEST_CASE(ValidateDataTypes, framework::DatasetMode::ALL)
{
    // Configurations are assumed to be Nt/Nt, but will be transposed inside the test to test other configurations
    using DataTypeConfigurationTuple = std::tuple<DataType, DataType, DataType, bool>;
    const std::vector<DataTypeConfigurationTuple> data_type_configurations = {
        {DataType::F32, DataType::F32, DataType::F32, true},
        {DataType::F16, DataType::F16, DataType::F16, true},
        {DataType::F16, DataType::F32, DataType::F32, false},                                  // no mixed precision
        {DataType::F64, DataType::F64, DataType::F64, false},                                  // no double precision
        {DataType::QASYMM8, DataType::QASYMM8, DataType::QASYMM8, false},                      // no quantized types
        {DataType::QASYMM8_SIGNED, DataType::QASYMM8_SIGNED, DataType::QASYMM8_SIGNED, false}, // no quantized types
        {DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8_PER_CHANNEL,
         false},                                                             // no quantized types
        {DataType::QASYMM16, DataType::QASYMM16, DataType::QASYMM16, false}, // no quantized types
        {DataType::QSYMM16, DataType::QSYMM16, DataType::QSYMM16, false},    // no quantized types
        {DataType::QSYMM8, DataType::QSYMM8, DataType::QSYMM8, false},       // no quantized types
        {DataType::S64, DataType::S64, DataType::S64, false},                // no integral types
        {DataType::S32, DataType::S32, DataType::S32, false},                // no integral types
        {DataType::S16, DataType::S16, DataType::S16, false},                // no integral types
        {DataType::S8, DataType::S8, DataType::S8, false},                   // no integral types
        {DataType::U64, DataType::U64, DataType::U64, false},                // no integral types
        {DataType::U32, DataType::U32, DataType::U32, false},                // no integral types
        {DataType::U16, DataType::U16, DataType::U16, false},                // no integral types
        {DataType::U8, DataType::U8, DataType::U8, false},                   // no integral types
    };
    // Create a sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{&cl_compile_ctx};
    GpuWorkloadSketch sketch{&context};

    const TensorShape shape = TensorShape(10U, 10U);
    MatMulAttributes  matmul_attr{};
    matmul_attr.adj_lhs(false);
    matmul_attr.adj_rhs(false);
    GpuMatMulSettings matmul_settings{};
    matmul_settings.m0(1);
    matmul_settings.n0(1);
    matmul_settings.k0(1);

    for (auto &tuple : data_type_configurations)
    {
        const bool expected = std::get<3>(tuple);

        const ITensorInfo *lhs_info = context.create_tensor_info(TensorInfo(shape, 1, std::get<0>(tuple)));
        const ITensorInfo *rhs_info = context.create_tensor_info(TensorInfo(shape, 1, std::get<1>(tuple)));

        Status status = GpuMatMul::validate_op(sketch, lhs_info, rhs_info, matmul_attr, matmul_settings);
        ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
    }
}

TEST_SUITE_END() // Validate

template <typename T>
using DynamicFusionGpuMatmulFixture = DynamicFusionGpuMatMulValidationFixture<CLTensor, CLAccessor, GpuMatMul, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunPrecommit,
                       DynamicFusionGpuMatmulFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(DFMatMulDataset(),
                               framework::dataset::make("TransposeA", {false}),
                               framework::dataset::make("TransposeB", {true}),
                               m0_values_lhs_nt_precommit,
                               n0_values_rhs_t_precommit,
                               k0_values_rhs_t_precommit,
                               framework::dataset::make("ExportRhsToCLImage", {false}),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunNightly,
                       DynamicFusionGpuMatmulFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(DFMatMulDataset(),
                               framework::dataset::make("TransposeA", {false}),
                               framework::dataset::make("TransposeB", {true}),
                               m0_values_lhs_nt_nightly,
                               n0_values_rhs_t_nightly,
                               k0_values_rhs_t_nightly,
                               framework::dataset::make("ExportRhsToCLImage", {false}),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunPrecommit,
                       DynamicFusionGpuMatmulFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(DFMatMulDataset(),
                               framework::dataset::make("TransposeA", {false}),
                               framework::dataset::make("TransposeB", {true}),
                               m0_values_lhs_nt_precommit,
                               n0_values_rhs_t_precommit,
                               k0_values_rhs_t_precommit,
                               framework::dataset::make("ExportRhsToCLImage", {false}),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunNightly,
                       DynamicFusionGpuMatmulFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(DFMatMulDataset(),
                               framework::dataset::make("TransposeA", {false}),
                               framework::dataset::make("TransposeB", {true}),
                               m0_values_lhs_nt_nightly,
                               n0_values_rhs_t_nightly,
                               k0_values_rhs_t_nightly,
                               framework::dataset::make("ExportRhsToCLImage", {false}),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0.f, abs_tolerance_f16);
}

TEST_SUITE_END() // FP16

TEST_SUITE_END() // Float
TEST_SUITE_END() // MatMul
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
