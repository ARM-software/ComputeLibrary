/*
 * Copyright (c) 2017-2025 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuGEMMLowp.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/LargeGEMMLowpDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallGEMMLowpDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/fixtures/CpuGEMMLowpFixture.h"

/*

 * Tests for arm_compute::experimental::op::CpuGEMMLowp which is a shallow wrapper for
 * arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore Any future testing to the functionalities of arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore will
 * be tested in tests/validation/NEON/GEMMLowp.cpp given that op::CpuGEMMLowp remain a shallow wrapper.
*/

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuGEMMLowp)

using CpuGEMMLowpFixture = CpuGEMMLowpMatrixMultiplyCoreValidationFixture<Tensor, Accessor, arm_compute::experimental::op::CpuGEMMLowp>;
using CpuGEMMLowpStaticQuantFixture = CpuGEMMLowpStaticQuantMatrixMultiplyCoreValidationFixture<Tensor, Accessor, arm_compute::experimental::op::CpuGEMMLowp>;

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
    arm_compute::experimental::op::CpuGEMMLowp gemmlowp_mm;
    gemmlowp_mm.configure(a.info(), b.info(), nullptr, c.info());

    // Validate padding is zero
    validate(a.info()->padding(), PaddingSize());
    validate(b.info()->padding(), PaddingSize());
    validate(c.info()->padding(), PaddingSize());
}
// accumulation is not supported for Int8/UInt8 in aarch32
#ifdef __aarch64__
DATA_TEST_CASE(ValidateAccumulate, framework::DatasetMode::ALL, combine(
                                                                    zip(
                                                                     make("In0",{ TensorShape(21U, 1U) }),
                                                                     make("In1", { TensorShape(1U, 21U) }),
                                                                     make("Dst", { TensorShape(1U, 1U) }),
                                                                     make("a_offset", { -2 }),
                                                                     make("b_offset", { 13 })
                                                                    ),
                                                                    zip(
                                                                     make("OutputDataType", {  DataType::S32,  DataType::QASYMM8, DataType::QASYMM8_SIGNED}),
                                                                     make("Expected", { true, false, false })
                                                                    )),
               shape_a, shape_b, shape_dst, a_offset, b_offset, output_data_type, expected)
{
    DataType input_data_type = (output_data_type == DataType::S32 ? DataType::QASYMM8 : output_data_type);
    // Accumulation test for GEMM kernels
    TensorInfo a(shape_a, 1, input_data_type, QuantizationInfo(1.0f / 255, a_offset));
    TensorInfo b(shape_b, 1, input_data_type, QuantizationInfo(1.0f / 255, b_offset));
    TensorInfo dst(shape_dst, 1, output_data_type, QuantizationInfo());

    // Create and configure function
    GEMMInfo gemm_info = GEMMInfo();
    gemm_info.set_accumulate(true);

    if (is_data_type_quantized(output_data_type))
    {
        GEMMLowpOutputStageInfo gemmLowpOutputStageInfo = GEMMLowpOutputStageInfo();
        gemmLowpOutputStageInfo.type = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;

        gemm_info.set_gemmlowp_output_stage(gemmLowpOutputStageInfo);
    }

    arm_compute::experimental::op::CpuGEMMLowp gemmlowp_mm;
    Status status = gemmlowp_mm.validate(&a, &b, nullptr, &dst, gemm_info);

    ARM_COMPUTE_EXPECT((expected ==  bool(status)), framework::LogLevel::ERRORS);
}
#endif // __arch64__

// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Input not a multiple of 4
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::S32),                                 // Mismatching data type
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Invalid dimensions
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Invalid dimensions
                                             TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)),
                                             TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(1.f/255, 10)), // Invalid types
                                          }),
    make("InputBInfo",{ TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                          }),
    make("OutputInfo",{ TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(8U, 11U), 1, DataType::S32),
                                            TensorInfo(TensorShape(64U, 32U), 1, DataType::S32),
                                            TensorInfo(TensorShape(64U, 32U), 1, DataType::S32),
                                           }),
    make("Expected", { true, false, false, false, true, false })),
    a_info, b_info, output_info, expected)
{
    // Lock tensors
    Status status = arm_compute::experimental::op::CpuGEMMLowp::validate(&a_info.clone()->set_is_resizable(false),
                                                            &b_info.clone()->set_is_resizable(false),
                                                            nullptr,
                                                            &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on

/** Test case for memory injection in @ref arm_compute::experimental::op::CpuGEMMLowp.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto gemm     = std::make_unique<arm_compute::experimental::op::CpuGEMMLowp>();
    auto a_info   = TensorInfo(TensorShape(32U, 72U), 1, DataType::QASYMM8);
    auto b_info   = TensorInfo(TensorShape(17U, 32U), 1, DataType::QASYMM8);
    auto dst_info = TensorInfo(TensorShape(17U, 72U), 1, DataType::S32);
    a_info.set_quantization_info(QuantizationInfo(1.0f / 255, -9));
    b_info.set_quantization_info(QuantizationInfo(1.0f / 255, 1));
    const auto gemm_info = GEMMInfo{};
    gemm->configure(&a_info, &b_info, nullptr, &dst_info, gemm_info);

    // The LHS are newly created every call of this lambda function
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
    allocate_tensors(gemm->workspace(), ws);

    auto run_gemm = [&]() -> Tensor
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
    auto result_0 = run_gemm();
    auto result_1 = run_gemm();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((uint8_t *)result_0.buffer())[i] == ((uint8_t *)result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

FIXTURE_DATA_TEST_CASE(SmokeTest, CpuGEMMLowpFixture, framework::DatasetMode::ALL, datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_targets[0]), _references[0]);
}

#ifdef __aarch64__ // All the GeMM CPU assembly kernels for integer datatypes require aarch64
TEST_SUITE(Quantized)

DATA_TEST_CASE(ValidateQuantized, framework::DatasetMode::ALL, zip(
    make("InputAInfo", { TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(1.f/255, 10)),
                         TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)),
                                          }),
    make("InputBInfo",{ TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(1.f/256, 10)),
                        TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(1.f/256, 10)),
                                          }),
    make("OutputInfo",{ TensorInfo(TensorShape(64U, 32U), 1, DataType::QASYMM8_SIGNED),
                        TensorInfo(TensorShape(64U, 32U), 1, DataType::QASYMM8),
                                           }),
    make("Expected", { true, true })),
    a_info, b_info, output_info, expected)
{
    // Lock tensors
    Status status = arm_compute::experimental::op::CpuGEMMLowp::validate(&a_info.clone()->set_is_resizable(false),
                                                            &b_info.clone()->set_is_resizable(false),
                                                            nullptr,
                                                            &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(SmokeTestStaticQuant, CpuGEMMLowpStaticQuantFixture, framework::DatasetMode::ALL, combine(datasets::SmallGEMMLowpDataset(), make("DataType", DataType::QASYMM8), make("bool", false)/*is_multithreaded*/))
{
    // Validate output
    validate(Accessor(_targets[0]), _references[0]);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(SmokeTestStaticQuant, CpuGEMMLowpStaticQuantFixture, framework::DatasetMode::ALL, combine(datasets::SmallGEMMLowpDataset(), make("DataType", DataType::QASYMM8_SIGNED), make("bool", false)/*is_multithreaded*/))
{
    // Validate output
    validate(Accessor(_targets[0]), _references[0]);
}
TEST_SUITE_END() // QASYMM8_SIGNED

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuGEMMLowpStaticQuantFixture, framework::DatasetMode::ALL, combine(datasets::SmallGEMMLowpDataset(), make("DataType", DataType::QASYMM8_SIGNED), make("bool", true)/*is_multithreaded*/))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_targets[i]), _references[i]);
    }
}
TEST_SUITE_END() // ThreadSafety
#endif // ifndef BARE_METAL
TEST_SUITE_END() // Quantized
#endif // #ifdef __aarch64__
TEST_SUITE_END() // CpuGEMMLowp
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
