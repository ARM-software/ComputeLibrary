/*
 * Copyright (c) 2017-2026 Arm Limited.
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
#include "tests/datasets/GEMMLowpFusedOffsetOutputDataset.h"
#include "tests/datasets/LargeGEMMLowpDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallGEMMLowpDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/GEMMLowpFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
constexpr AbsoluteTolerance<float> tolerance_batched(1);
constexpr AbsoluteTolerance<float> tolerance_quant(1);
#ifdef __aarch64__
constexpr float large_test_tolerance_num = 0.00001f;
#endif // __aarch64__

#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<half_float::half> rel_tolerance_f16(half(
    0.2)); /**< Relative tolerance value for comparing reference's output against implementation's output for FP16 data types */
const AbsoluteTolerance<float>      abs_tolerance_f16(
         0.2f); /**< Absolute tolerance value for comparing reference's output against implementation's output for FP16 data types */
constexpr float tolerance_num = 0.07f; /**< Tolerance number for FP16 data types */
#endif                                 /* ARM_COMPUTE_ENABLE_FP16 */

} // namespace

const auto QuantizedActivationFunctionsDataset =
    make("ActivationInfo",
         {ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
          ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)});

TEST_SUITE(NEON)
TEST_SUITE(GEMMLowp)
TEST_SUITE(MatrixMultiplyCore)

using NEGEMMLowpMatrixMultiplyCoreFixture =
    GEMMLowpMatrixMultiplyCoreValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;
using NEGEMMLowpMatrixMultiplyCoreAccumulateFixture =
    GEMMLowpMatrixMultiplyAccumulateValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;
using NEGEMMLowpBatchedMatMulFixture =
    GEMMLowpMatrixMultiplyCoreValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore, false, false, true>;
using NEGEMMLowpMatrixMultiplyCoreDynamicQuantizationFixture =
    GEMMLowpMatrixMultiplyCoreDynamicQuantizationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;
using NEGEMMLowpDequantizedF32MatrixMultiplyValidationFixture =
    GEMMLowpDequantizedMatrixMultiplyValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore, float>;
using NEGEMMLowpDequantizedF16MatrixMultiplyValidationFixture =
    GEMMLowpDequantizedMatrixMultiplyValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore, half>;

DATA_TEST_CASE(Configuration,
               framework::DatasetMode::ALL,
               framework::dataset::concat(datasets::SmallGEMMLowpDataset(), datasets::LargeGEMMLowpDataset()),
               shape_a,
               shape_b,
               shape_c,
               a_offset,
               b_offset)
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
// accumulation is not supported for Int8/UInt8 in aarch32
#ifdef __aarch64__
DATA_TEST_CASE(ValidateAccumulate,
               framework::DatasetMode::ALL,
               combine(zip(make("In0", {TensorShape(21U, 1U)}),
                           make("In1", {TensorShape(1U, 21U)}),
                           make("Dst", {TensorShape(1U, 1U)}),
                           make("a_offset", {-2}),
                           make("a_offset", {13})),
                       zip(make("OutputDataType", {DataType::S32, DataType::QASYMM8, DataType::QASYMM8_SIGNED}),
                           make("Expected", {true, false, false}))),
               shape_a,
               shape_b,
               shape_dst,
               a_offset,
               b_offset,
               output_data_type,
               expected)
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
        gemmLowpOutputStageInfo.type                    = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;

        gemm_info.set_gemmlowp_output_stage(gemmLowpOutputStageInfo);
    }

    Status status = cpu::CpuGemmLowpMatrixMultiplyCore::validate(&a, &b, nullptr, &dst, gemm_info);

    ARM_COMPUTE_EXPECT((expected == bool(status)), framework::LogLevel::ERRORS);
}
#endif // __arch64__

// *INDENT-OFF*
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
    make("Expected", { true, false, false, false, true, false })
    ),
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

    ITensorPack run_pack  = {{TensorType::ACL_SRC_0, &a}, {TensorType::ACL_SRC_1, &b}, {TensorType::ACL_DST, &dst}};
    ITensorPack prep_pack = {
        {TensorType::ACL_SRC_1, &b},
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
    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((uint8_t *)result_0.buffer())[i] == ((uint8_t *)result_1.buffer())[i],
                           framework::LogLevel::ERRORS);
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
    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((uint8_t *)result_0.buffer())[i] == ((uint8_t *)result_1.buffer())[i],
                           framework::LogLevel::ERRORS);
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreFixture,
                       framework::DatasetMode::ALL,
                       datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMLowpMatrixMultiplyCoreFixture,
                       framework::DatasetMode::NIGHTLY,
                       datasets::LargeGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE(BatchedMatMul)
TEST_SUITE(QASYMM8)
using NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedUnsigned =
    GEMMLowpBatchedMatrixMultiplyCoreFusedOffsetOutputFixture<Tensor,
                                                              Accessor,
                                                              NEGEMMLowpMatrixMultiplyCore,
                                                              false,
                                                              false,
                                                              uint8_t,
                                                              uint8_t,
                                                              true>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedUnsigned,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpFusedBatchedMatMulDataset(),
                               make("DataType", {DataType::QASYMM8}),
                               make("reshape_b_only_on_first_run", {false})))
{
    validate(Accessor(_target), _reference, tolerance_batched);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
using NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedSigned =
    GEMMLowpBatchedMatrixMultiplyCoreFusedOffsetOutputFixture<Tensor,
                                                              Accessor,
                                                              NEGEMMLowpMatrixMultiplyCore,
                                                              false,
                                                              false,
                                                              int8_t,
                                                              int8_t,
                                                              true>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixtureBatchedSigned,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpFusedBatchedMatMulDataset(),
                               make("DataType", {DataType::QASYMM8_SIGNED}),
                               make("reshape_b_only_on_first_run", {false})))
{
    validate(Accessor(_target), _reference, tolerance_batched);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // BatchedMatMul

TEST_SUITE(FusedOffsetOutput)
using NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture =
    GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpFusedOffsetOutputUint8Dataset(),
                               make("DataType", {DataType::QASYMM8}),
                               make("reshape_b_only_on_first_run", {false})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quant);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMLowpFusedOffsetOutputUint8Dataset(),
                               make("DataType", {DataType::QASYMM8}),
                               make("reshape_b_only_on_first_run", {false})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // FusedOffsetOutput

// accumulation is not supported for Int8/UInt8 in aarch32
#ifdef __aarch64__
TEST_SUITE(ACCUMULATION)
TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreAccumulateFixture,
                       framework::DatasetMode::ALL,
                       datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMLowpMatrixMultiplyCoreAccumulateFixture,
                       framework::DatasetMode::NIGHTLY,
                       datasets::LargeGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32
TEST_SUITE_END() // ACCUMULATION
#endif           // __arch64__

TEST_SUITE(DynamicQuantization)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreDynamicQuantizationFixture,
                       framework::DatasetMode::ALL,
                       datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMLowpMatrixMultiplyCoreDynamicQuantizationFixture,
                       framework::DatasetMode::NIGHTLY,
                       datasets::LargeGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // DynamicQuantization

#ifdef __aarch64__
TEST_SUITE(UpdateStaticQuantInfoAfterConfigure)
TEST_SUITE(QASYMM8_SIGNED)
using NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureInt8Fixture =
    GEMMLowpGenericMatrixMultiplyCoreFusedOffsetOutputValidationFixture<Tensor,
                                                                        Accessor,
                                                                        NEGEMMLowpMatrixMultiplyCore,
                                                                        false,
                                                                        false,
                                                                        int8_t,
                                                                        int8_t,
                                                                        true>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureInt8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpFusedOffsetOutputUint8Dataset(),
                               make("DataTypeA", {DataType::QASYMM8_SIGNED}),
                               make("DataTypeB", {DataType::QASYMM8_SIGNED}),
                               make("reshape_b_only_on_first_run", {false}),
                               make("updated_sq_info_after_config", {true}),
                               QuantizedActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_batched);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureInt8Fixture,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMLowpFusedOffsetOutputUint8Dataset(),
                               make("DataTypeA", {DataType::QASYMM8_SIGNED}),
                               make("DataTypeB", {DataType::QASYMM8_SIGNED}),
                               make("reshape_b_only_on_first_run", {false}),
                               make("updated_sq_info_after_config", {true}),
                               QuantizedActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_batched, large_test_tolerance_num);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE(QASYMM8)
using NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureUInt8Fixture =
    GEMMLowpGenericMatrixMultiplyCoreFusedOffsetOutputValidationFixture<Tensor,
                                                                        Accessor,
                                                                        NEGEMMLowpMatrixMultiplyCore,
                                                                        false,
                                                                        false,
                                                                        uint8_t,
                                                                        uint8_t,
                                                                        true>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureUInt8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpFusedOffsetOutputUint8Dataset(),
                               make("DataTypeA", {DataType::QASYMM8}),
                               make("DataTypeB", {DataType::QASYMM8}),
                               make("reshape_b_only_on_first_run", {false}),
                               make("updated_sq_info_after_config", {true}),
                               QuantizedActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_batched);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureUInt8Fixture,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMLowpFusedOffsetOutputUint8Dataset(),
                               make("DataTypeA", {DataType::QASYMM8}),
                               make("DataTypeB", {DataType::QASYMM8}),
                               make("reshape_b_only_on_first_run", {false}),
                               make("updated_sq_info_after_config", {true}),
                               QuantizedActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_batched, large_test_tolerance_num);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(MixedQuantizedType)
using NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureInt8Fixture =
    GEMMLowpGenericMatrixMultiplyCoreFusedOffsetOutputValidationFixture<Tensor,
                                                                        Accessor,
                                                                        NEGEMMLowpMatrixMultiplyCore,
                                                                        false,
                                                                        false,
                                                                        uint8_t,
                                                                        int8_t,
                                                                        true>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpMatrixMultiplyCoreForUpdatedStaticQuantInfoAfterConfigureInt8Fixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpFusedOffsetOutputUint8Dataset(),
                               make("DataTypeA", {DataType::QASYMM8}),
                               make("DataTypeB", {DataType::QASYMM8_SIGNED}),
                               make("reshape_b_only_on_first_run", {false}),
                               make("updated_sq_info_after_config", {true}),
                               QuantizedActivationFunctionsDataset))
{
    validate(Accessor(_target), _reference, tolerance_batched);
}
TEST_SUITE_END() // MixedQuantizedType
TEST_SUITE_END() // UpdateStaticQuantInfoAfterConfigure

// Deqaunt tests involve returning FP32 from the MatrixMultiplyCore kernels and is only implemented in aarch64
TEST_SUITE(Dequant)
TEST_SUITE(FP32)

DATA_TEST_CASE(
    Validate,
    framework::DatasetMode::ALL,
    zip(make("InputAInfo",
             {
                 TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8, QuantizationInfo(1.f / 255, 10)),
                 TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(1.f / 255, 10)),
                 TensorInfo(TensorShape(16U, 32U),
                            1,
                            DataType::QASYMM8_SIGNED,
                            QuantizationInfo(1.f / 255, 10)), // Invalid types
             }),
        make("InputBInfo",
             {
                 TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(1.f / 256, 10)),
                 TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(1.f / 256, 10)),
                 TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8, QuantizationInfo(1.f / 256, 10)),
             }),
        make("OutputInfo",
             {
                 TensorInfo(TensorShape(64U, 32U), 1, DataType::F32),
                 TensorInfo(TensorShape(64U, 32U), 1, DataType::F32),
                 TensorInfo(TensorShape(64U, 32U), 1, DataType::F32),
             }),
        make("Expected", {true, true, false})),
    a_info,
    b_info,
    output_info,
    expected)
{
    // Lock tensors
    Status status = NEGEMMLowpMatrixMultiplyCore::validate(&a_info.clone()->set_is_resizable(false),
                                                           &b_info.clone()->set_is_resizable(false), nullptr,
                                                           &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

constexpr AbsoluteTolerance<float> tolerance_dequantized(0.01f);
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpDequantizedF32MatrixMultiplyValidationFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpDataset(),
                               make("DataTypeA", {DataType::QASYMM8_SIGNED, DataType::QASYMM8}),
                               make("DataTypeB", DataType::QASYMM8_SIGNED),
                               make("DataTypeB", DataType::F32),
                               make("accumulate", {true, false})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_dequantized);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMLowpDequantizedF32MatrixMultiplyValidationFixture,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMLowpDataset(),
                               make("DataTypeA", {DataType::QASYMM8_SIGNED, DataType::QASYMM8}),
                               make("DataTypeB", DataType::QASYMM8_SIGNED),
                               make("DataTypeB", DataType::F32),
                               make("accumulate", {false})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_dequantized, large_test_tolerance_num);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMLowpDequantizedF16MatrixMultiplyValidationFixture,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallGEMMLowpDataset(),
                               make("DataTypeA", DataType::QASYMM8_SIGNED),
                               make("DataTypeB", DataType::QASYMM8_SIGNED),
                               make("DataTypeC", DataType::F16),
                               make("accumulate", {true, false})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE_END() // Dequant
#endif           // __aarch64__

TEST_SUITE_END() // MatrixMultiplyCore
TEST_SUITE_END() // GEMMLowp
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
