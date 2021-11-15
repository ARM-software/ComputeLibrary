/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/GEMMLowpFusedOffsetOutputDataset.h"
#include "tests/datasets/LargeGEMMLowpDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallGEMMLowpDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMLowpFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(GEMMLowp)
TEST_SUITE(MatrixMultiplyCore)
using NEGEMMLowpMatrixMultiplyCoreFixture = GEMMLowpMatrixMultiplyCoreValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, framework::dataset::concat(datasets::SmallGEMMLowpDataset(), datasets::LargeGEMMLowpDataset()),
               shape_a, shape_b, shape_c, a_offset, b_offset)
{
    // Create tensors
    Tensor a = create_tensor<Tensor>(shape_a, DataType::QASYMM8);
    Tensor b = create_tensor<Tensor>(shape_b, DataType::QASYMM8);
    Tensor c = create_tensor<Tensor>(shape_c, DataType::S32);

    a.info()->set_quantization_info(QuantizationInfo(1.0f / 255, a_offset));
    b.info()->set_quantization_info(QuantizationInfo(1.0f / 255, b_offset));

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEGEMMLowpMatrixMultiplyCore gemmlowp_mm;
    gemmlowp_mm.configure(&a, &b, nullptr, &c);

    // Validate padding is zero
    validate(a.info()->padding(), PaddingSize());
    validate(b.info()->padding(), PaddingSize());
    validate(c.info()->padding(), PaddingSize());
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Input not a multiple of 4
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::S32),                                 // Mismatching data type
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Invalid dimensions
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Invalid dimensions
                                             TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)),
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(8U, 11U), 1, DataType::S32),
                                            TensorInfo(TensorShape(64U, 32U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Expected", { true, false, false, false, true })),
    a_info, b_info, output_info, expected)
{
    // Lock tensors
    Status status =  NEGEMMLowpMatrixMultiplyCore::validate(&a_info.clone()->set_is_resizable(false),
                                                            &b_info.clone()->set_is_resizable(false),
                                                            nullptr,
                                                            &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

/** Test case for memory injection in @ref cpu::CpuGemmLowpMatrixMultiplyCore.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto gemm     = std::make_unique<cpu::CpuGemmLowpMatrixMultiplyCore>();
    auto a_info   = TensorInfo(TensorShape(32U, 72U), 1, DataType::QASYMM8);
    auto b_info   = TensorInfo(TensorShape(17U, 32U), 1, DataType::QASYMM8);
    auto dst_info = TensorInfo(TensorShape(17U, 72U), 1, DataType::S32);
    a_info.set_quantization_info(QuantizationInfo(1.0f / 255, -9));
    b_info.set_quantization_info(QuantizationInfo(1.0f / 255, 1));
    const auto gemm_info = GEMMInfo{};
    gemm->configure(&a_info, &b_info, nullptr, &dst_info, gemm_info);

    // telhs are newly created every call of this lambda function
    auto a   = create_tensor<Tensor>(a_info);
    auto b   = create_tensor<Tensor>(b_info);
    auto dst = create_tensor<Tensor>(dst_info);
    a.allocator()->allocate();
    b.allocator()->allocate();
    dst.allocator()->allocate();

    ITensorPack run_pack =
    {
        { TensorType::ACL_SRC_0, &a },
        { TensorType::ACL_SRC_1, &b },
        { TensorType::ACL_DST, &dst }
    };
    ITensorPack prep_pack =
    {
        { TensorType::ACL_SRC_1, &b },
    };

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(gemm->workspace(), mg, run_pack, prep_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(a), static_cast<uint8_t>(1));
        library->fill_tensor_value(Accessor(b), static_cast<uint8_t>(2));
        // This operator is configured once and captured by this lambda.
        gemm->prepare(prep_pack);
        gemm->run(run_pack);
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((uint8_t *)result_0.buffer())[i] == ((uint8_t *)result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

/** Test case for memory injection in @ref NEGEMMLowpMatrixMultiplyCore.
 *
 * Make sure @ref NEGEMMLowpMatrixMultiplyCore still works through injecting the memory at configure time using the old API.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MultipleExecutionWithConfigure, framework::DatasetMode::ALL)
{
    auto gemm     = std::make_unique<NEGEMMLowpMatrixMultiplyCore>();
    auto a_info   = TensorInfo(TensorShape(32U, 72U), 1, DataType::QASYMM8);
    auto b_info   = TensorInfo(TensorShape(17U, 32U), 1, DataType::QASYMM8);
    auto dst_info = TensorInfo(TensorShape(17U, 72U), 1, DataType::S32);
    a_info.set_quantization_info(QuantizationInfo(1.0f / 255, -9));
    b_info.set_quantization_info(QuantizationInfo(1.0f / 255, 1));
    const auto gemm_info = GEMMInfo{};
    auto       run_conv  = [&]()
    {
        auto a   = create_tensor<Tensor>(a_info);
        auto b   = create_tensor<Tensor>(b_info);
        auto dst = create_tensor<Tensor>(dst_info);
        gemm->configure(&a, &b, nullptr, &dst, gemm_info);
        a.allocator()->allocate();
        b.allocator()->allocate();
        dst.allocator()->allocate();
        library->fill_tensor_value(Accessor(a), static_cast<uint8_t>(1));
        library->fill_tensor_value(Accessor(b), static_cast<uint8_t>(2));
        gemm->run();
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((uint8_t *)result_0.buffer())[i] == ((uint8_t *)result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::ALL, datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::NIGHTLY, datasets::LargeGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

using NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture = GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;
TEST_SUITE(FusedOffsetOutput)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture, framework::DatasetMode::ALL, combine(datasets::SmallGEMMLowpFusedOffsetOutputUint8Dataset(),
                       framework::dataset::make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMLowpFusedOffsetOutputUint8Dataset(),
                       framework::dataset::make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FusedOffsetOutput
TEST_SUITE_END() // MatrixMultiplyCore
TEST_SUITE_END() // GEMMLowp
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
