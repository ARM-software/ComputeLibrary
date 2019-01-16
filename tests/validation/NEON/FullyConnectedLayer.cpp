/*
 * Copyright (c) 2017-2019 ARM Limited.
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

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
});

const auto FullyConnectedParameters = combine(framework::dataset::make("TransposeWeights", { false, true }), framework::dataset::make("ReshapeWeights", { false, true }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(FullyConnectedLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                                                                           FullyConnectedParameters),
                                                                   CNNDataTypes),
               src_shape, weights_shape, bias_shape, dst_shape, transpose_weights, reshape_weights, data_type)
{
    const DataType         bias_data_type    = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;
    const QuantizationInfo quantization_info = is_data_type_quantized_asymmetric(data_type) ? QuantizationInfo(2.f / 255.f, 127) : QuantizationInfo();

    TensorShape ws(weights_shape);

    // Transpose weights if not done in the function
    if(!reshape_weights || !transpose_weights)
    {
        const size_t shape_x = ws.x();
        ws.set(0, ws.y());
        ws.set(1, shape_x);
    }

    // Create tensors
    Tensor src     = create_tensor<Tensor>(src_shape, data_type, 1, quantization_info);
    Tensor weights = create_tensor<Tensor>(ws, data_type, 1, quantization_info);
    Tensor bias    = create_tensor<Tensor>(bias_shape, bias_data_type, 1, quantization_info);
    Tensor dst     = create_tensor<Tensor>(dst_shape, data_type, 1, quantization_info);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create Fully Connected layer info
    FullyConnectedLayerInfo fc_info;
    fc_info.transpose_weights    = transpose_weights;
    fc_info.are_weights_reshaped = !reshape_weights;

    const QuantizationInfo src_quantization_info     = src.info()->quantization_info();
    const QuantizationInfo weights_quantization_info = weights.info()->quantization_info();

    // Create and configure function.
    NEFullyConnectedLayer fc;
    fc.configure(&src, &weights, &bias, &dst, fc_info);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(dst_shape);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate QuantizationInfo
    ARM_COMPUTE_EXPECT(src.info()->quantization_info() == src_quantization_info, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->quantization_info() == weights_quantization_info, framework::LogLevel::ERRORS);
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

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFullyConnectedLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                                                                                                                        FullyConnectedParameters),
                                                                                                                framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFullyConnectedLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeFullyConnectedLayerDataset(),
                                                                                                                      FullyConnectedParameters),
                                                                                                              framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFullyConnectedLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFullyConnectedLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NEFullyConnectedLayerQuantizedFixture = FullyConnectedLayerValidationQuantizedFixture<Tensor, Accessor, NEFullyConnectedLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(
                           combine(datasets::SmallFullyConnectedLayerDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255.f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(
                           combine(datasets::LargeFullyConnectedLayerDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 256.f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
