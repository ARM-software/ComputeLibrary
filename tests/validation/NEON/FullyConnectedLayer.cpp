/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuFullyConnected.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/FullyConnectedLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FullyConnectedLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
constexpr RelativeTolerance<float> rel_tolerance_f32(0.01f);  /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<float> abs_tolerance_f32(0.001f); /**< Absolute tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const AbsoluteTolerance<float>            abs_tolerance_f16(0.3f);                   /**< Absolute tolerance value for comparing reference's output against implementation's output for DataType::F16 */
const RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2f)); /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                           tolerance_num_f16 = 0.07f;                 /**< Tolerance number for FP16 */
#endif                                                                               /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC*/

/** Tolerance for quantized asymmetric operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_signed(1);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
});

const auto FullyConnectedParameters = combine(framework::dataset::make("TransposeWeights", { false, true }), framework::dataset::make("ReshapeWeights", { false, true }));

const auto QuantizationData = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(1.f / 256.f, 10),
    QuantizationInfo(1.1f, 10),
});
const auto EmptyActivationFunctionDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
});
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH),
});

const auto ActivationFunctionsQuantizedDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.75f, 0.25f),
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(FullyConnectedLayer)

/** Test case for memory injection in @ref cpu::CpuFullyConnected.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto       fc          = std::make_unique<cpu::CpuFullyConnected>();
    const auto src_info    = TensorInfo(TensorShape(8U), 1, DataType::F32, DataLayout::NHWC);
    const auto weight_info = TensorInfo(TensorShape(8U, 4U), 1, DataType::F32, DataLayout::NHWC);
    const auto bias_info   = TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC);
    auto       dst_info    = TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC);
    const auto fc_info     = FullyConnectedLayerInfo{};
    fc->configure(&src_info, &weight_info, &bias_info, &dst_info, fc_info);

    // telhs are newly created every call of this lambda function
    auto src    = create_tensor<Tensor>(src_info);
    auto weight = create_tensor<Tensor>(weight_info);
    auto bias   = create_tensor<Tensor>(bias_info);
    src.allocator()->allocate();
    weight.allocator()->allocate();
    bias.allocator()->allocate();

    ITensorPack run_pack{ { TensorType::ACL_SRC_0, &src }, { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias } };
    ITensorPack prep_pack{ { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias } };

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(fc->workspace(), mg, run_pack, prep_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weight), 2.f);
        library->fill_tensor_value(Accessor(bias), 3.f);
        // This operator is configured once and captured by this lambda.
        fc->prepare(prep_pack);
        fc->run(run_pack);
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

/** Test case for memory injection in @ref NEFullyConnectedLayer.
 *
 * Make sure @ref NEFullyConnectedLayer still works through injecting the memory at configure time using the old API.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MultipleExecutionWithConfigure, framework::DatasetMode::ALL)
{
    auto       fc          = std::make_unique<NEFullyConnectedLayer>();
    const auto src_info    = TensorInfo(TensorShape(8U), 1, DataType::F32, DataLayout::NHWC);
    const auto weight_info = TensorInfo(TensorShape(8U, 4U), 1, DataType::F32, DataLayout::NHWC);
    const auto bias_info   = TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC);
    auto       dst_info    = TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC);
    const auto fc_info     = FullyConnectedLayerInfo{};
    auto       run_conv    = [&]()
    {
        auto src    = create_tensor<Tensor>(src_info);
        auto weight = create_tensor<Tensor>(weight_info);
        auto bias   = create_tensor<Tensor>(bias_info);
        auto dst    = create_tensor<Tensor>(dst_info);
        fc->configure(&src, &weight, &bias, &dst, fc_info);
        src.allocator()->allocate();
        weight.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();
        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weight), 2.f);
        library->fill_tensor_value(Accessor(bias), 3.f);
        fc->run();
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

/** Unit test for @ref cpu::CpuFullyConnected with quantized multipler > 1
 *
 * Tests output correctness.
 */
TEST_CASE(Quant8_Signed_Mult_gt_1, framework::DatasetMode::ALL)
{
    auto       fc          = std::make_unique<cpu::CpuFullyConnected>();
    const auto src_info    = TensorInfo(TensorShape(1U, 3U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(0.5f, -1));
    const auto weight_info = TensorInfo(TensorShape(1U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(0.5, -8));
    const auto bias_info   = TensorInfo(TensorShape(1U), 1, DataType::S32);
    auto       dst_info    = TensorInfo(TensorShape(1U, 3U), 1, DataType::QASYMM8_SIGNED, QuantizationInfo(0.1f, 0));
    const auto fc_info     = FullyConnectedLayerInfo{};
    fc->configure(&src_info, &weight_info, &bias_info, &dst_info, fc_info);

    // telhs are newly created every call of this lambda function
    auto src    = create_tensor<Tensor>(src_info);
    auto weight = create_tensor<Tensor>(weight_info);
    auto bias   = create_tensor<Tensor>(bias_info);
    auto dst    = create_tensor<Tensor>(dst_info);
    src.allocator()->allocate();
    weight.allocator()->allocate();
    bias.allocator()->allocate();
    dst.allocator()->allocate();

    ITensorPack run_pack{ { TensorType::ACL_SRC_0, &src }, { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias }, { TensorType::ACL_DST, &dst } };
    ITensorPack prep_pack{ { TensorType::ACL_SRC_1, &weight }, { TensorType::ACL_SRC_2, &bias } };

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(fc->workspace(), mg, run_pack, prep_pack);

    // Initialize input values
    const std::vector<int8_t>  src_values    = { 3, 63, 31 };
    const std::vector<int8_t>  weight_values = { -4 };
    const std::vector<int32_t> bias_values   = { 16 };
    const std::vector<int32_t> expected      = { 80, 127, 127 };
    library->fill_static_values(Accessor(src), src_values);
    library->fill_static_values(Accessor(weight), weight_values);
    library->fill_static_values(Accessor(bias), bias_values);

    // Run FC layer
    fc->prepare(prep_pack);
    fc->run(run_pack);

    auto dst_ptr = reinterpret_cast<int8_t *>(dst.buffer());
    for(size_t i = 0; i < dst.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(dst_ptr[i] == expected[i], framework::LogLevel::ERRORS);
    }
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),    // Mismatching data types
                                            TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),    // Invalid weights dimensions
                                            TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),    // Wrongly reshaped weights
                                            TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32),
                                          }),
    framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(315U, 271U), 1, DataType::F16),
                                             TensorInfo(TensorShape(192U, 192U), 1, DataType::F32),
                                             TensorInfo(TensorShape(192U, 192U), 1, DataType::F32),
                                             TensorInfo(TensorShape(217U, 315U), 1, DataType::F32),
                                             TensorInfo(TensorShape(217U, 315U), 1, DataType::F32),
                                             TensorInfo(TensorShape(192U, 192U), 1, DataType::F32),
                                          })),
    framework::dataset::make("BiasInfo",{ TensorInfo(TensorShape(271U), 1, DataType::F32),
                                          TensorInfo(TensorShape(192U), 1, DataType::F32),
                                          TensorInfo(TensorShape(192U), 1, DataType::F32),
                                          TensorInfo(TensorShape(271U), 1, DataType::F32),
                                          TensorInfo(TensorShape(271U), 1, DataType::F32),
                                          TensorInfo(TensorShape(192U), 1, DataType::F32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(192U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(192U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(271U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(192U, 4U), 1, DataType::F32),
                                           })),
    framework::dataset::make("TransposeWeights",{ true, true, false, true, true, true })),
    framework::dataset::make("ReshapedWeights",{ false, false, false, false, false , false})),
    framework::dataset::make("Expected", { false, true, true, false, false, true })),
    input_info, weights_info, bias_info, output_info, transpose_weights, reshaped_weights, expected)
{
    // Create Fully Connected layer info
    FullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights = transpose_weights;
    fc_info.are_weights_reshaped = reshaped_weights;

    Status status = NEFullyConnectedLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), fc_info);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEFullyConnectedLayerFixture = FullyConnectedLayerValidationFixture<Tensor, Accessor, NEFullyConnectedLayer, T>;
template <typename T>
using NEFullyConnectedLayerMixedDataLayoutFixture = FullyConnectedLayerValidationFixture<Tensor, Accessor, NEFullyConnectedLayer, T, true>;
template <typename T>
using NEFullyConnectedLayerDynamicWeightsFixture = FullyConnectedWithDynamicWeightsFixture<Tensor, Accessor, NEFullyConnectedLayer, T>;
template <typename T>
using NEFullyConnectedLayerDynamicBiasFixture = FullyConnectedWithDynamicBiasFixture<Tensor, Accessor, NEFullyConnectedLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFullyConnectedLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                                                                                                                        FullyConnectedParameters),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                EmptyActivationFunctionDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, NEFullyConnectedLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(
                           combine(datasets::FullyConnectedLayerWithActivationDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::F16)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFullyConnectedLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(),
                                                                                                                      FullyConnectedParameters),
                                                                                                                      framework::dataset::make("DataType", DataType::F16)),
                                                                                                              EmptyActivationFunctionDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunDynamicWeights, NEFullyConnectedLayerDynamicWeightsFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       framework::dataset::make("DataType", DataType::F16)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                       framework::dataset::make("WeightsReshaped", { false, true })))
{
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFullyConnectedLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 EmptyActivationFunctionDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, NEFullyConnectedLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(
                           framework::dataset::make("Input", TensorShape(9U, 5U, 7U)),
                           framework::dataset::make("Weights", TensorShape(315U, 271U))),
                       framework::dataset::make("Biases", TensorShape(271U))),
                       framework::dataset::make("Output", TensorShape(271U))),
                       FullyConnectedParameters),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, NEFullyConnectedLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(
                           combine(datasets::FullyConnectedLayerWithActivationDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::F32)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFullyConnectedLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                       framework::dataset::make("DataType", DataType::F32)),
                                                                                                               EmptyActivationFunctionDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunDynamicWeights, NEFullyConnectedLayerDynamicWeightsFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                       framework::dataset::make("WeightsReshaped", { false, true })))
{
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NEFullyConnectedLayerQuantizedFixture = FullyConnectedLayerValidationQuantizedFixture<Tensor, Accessor, NEFullyConnectedLayer, T>;
template <typename T>
using NEFullyConnectedLayerQuantizedMixedDataLayoutFixture = FullyConnectedLayerValidationQuantizedFixture<Tensor, Accessor, NEFullyConnectedLayer, T, true>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                           combine(datasets::SmallFullyConnectedLayerDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8)),
                       QuantizationData),
                       EmptyActivationFunctionDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, NEFullyConnectedLayerQuantizedMixedDataLayoutFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(
                                                                           framework::dataset::make("Input", TensorShape(9U, 5U, 7U)),
                                                                           framework::dataset::make("Weights", TensorShape(315U, 271U))),
                                                                       framework::dataset::make("Biases", TensorShape(271U))),
                                                               framework::dataset::make("Output", TensorShape(271U))),
                                                       FullyConnectedParameters),
                                               framework::dataset::make("DataType", DataType::QASYMM8)),
                                       QuantizationData),
                               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, NEFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                           combine(datasets::FullyConnectedLayerWithActivationDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8)),
                       QuantizationData),
                       ActivationFunctionsQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(
                           combine(datasets::LargeFullyConnectedLayerDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8)),
                       QuantizationData),
                       EmptyActivationFunctionDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunDynamicBias, NEFullyConnectedLayerDynamicBiasFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
}
FIXTURE_DATA_TEST_CASE(RunDynamicWeights, NEFullyConnectedLayerDynamicWeightsFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                       framework::dataset::make("WeightsReshaped", { false })))
{
}
TEST_SUITE_END()
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFullyConnectedLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                           combine(datasets::SmallFullyConnectedLayerDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                       QuantizationData),
                       EmptyActivationFunctionDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, NEFullyConnectedLayerQuantizedMixedDataLayoutFixture<int8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(combine(combine(
                                                                           framework::dataset::make("Input", TensorShape(9U, 5U, 7U)),
                                                                           framework::dataset::make("Weights", TensorShape(315U, 271U))),
                                                                       framework::dataset::make("Biases", TensorShape(271U))),
                                                               framework::dataset::make("Output", TensorShape(271U))),
                                                       FullyConnectedParameters),
                                               framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                       QuantizationData),
                               framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, NEFullyConnectedLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                           combine(datasets::FullyConnectedLayerWithActivationDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                       QuantizationData),
                       ActivationFunctionsQuantizedDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE(RunDynamicWeights, NEFullyConnectedLayerDynamicWeightsFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                       framework::dataset::make("ActivationInfo", ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))),
                       framework::dataset::make("WeightsReshaped", { false })))
{
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // FullyConnectedLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
