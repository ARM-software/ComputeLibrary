/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEWinogradLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeConvolutionLayerDataset.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"
#include "tests/validation/fixtures/WinogradLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const AbsoluteTolerance<float> tolerance_f16(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif                                               /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const AbsoluteTolerance<float> tolerance_q(1.0f);    /**< Tolerance value for comparing reference's output against implementation's output for fixed point data types */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
    DataType::QS8,
    DataType::QS16,
});
} // namespace

TEST_SUITE(NEON)

#if defined(__aarch64__)
TEST_SUITE(WinogradLayer)
template <typename T>
using NEWinogradLayerFixture = WinogradLayerValidationFixture<Tensor, Accessor, NEWinogradLayer, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEWinogradLayerFixture<float>, framework::DatasetMode::PRECOMMIT, datasets::SmallWinogradLayerDataset())
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END()
TEST_SUITE_END()
#endif /* __aarch64__ */

TEST_SUITE(ConvolutionLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallConvolutionLayerDataset(), datasets::LargeConvolutionLayerDataset()), CNNDataTypes),
               input_shape, weights_shape, bias_shape, output_shape, info, data_type)
{
    // Set fixed point position data type allowed
    int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    // Create tensors
    Tensor src     = create_tensor<Tensor>(input_shape, data_type, 1, fixed_point_position);
    Tensor weights = create_tensor<Tensor>(weights_shape, data_type, 1, fixed_point_position);
    Tensor bias    = create_tensor<Tensor>(bias_shape, data_type, 1, fixed_point_position);
    Tensor dst     = create_tensor<Tensor>(output_shape, data_type, 1, fixed_point_position);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEConvolutionLayer conv;
    conv.configure(&src, &weights, &bias, &dst, info);

    // Validate valid region
    const ValidRegion src_valid_region     = shape_to_valid_region(input_shape);
    const ValidRegion weights_valid_region = shape_to_valid_region(weights_shape);
    const ValidRegion bias_valid_region    = shape_to_valid_region(bias_shape);
    const ValidRegion dst_valid_region     = shape_to_valid_region(output_shape);

    validate(src.info()->valid_region(), src_valid_region);
    validate(weights.info()->valid_region(), weights_valid_region);
    validate(bias.info()->valid_region(), bias_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);
}

template <typename T>
using NEConvolutionLayerFixture = ConvolutionValidationFixture<Tensor, Accessor, NEConvolutionLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                     framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                             framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeConvolutionLayerDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                           framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                      framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                              framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                            framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NEConvolutionLayerFixedPointFixture = ConvolutionValidationFixedPointFixture<Tensor, Accessor, NEConvolutionLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
// We test for fixed point precision [4,6]
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true, false })),
                       framework::dataset::make("DataType", DataType::QS8)),
                       framework::dataset::make("FractionalBits", 4, 7)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                                       framework::dataset::make("DataType", DataType::QS8)),
                                                                                                                       framework::dataset::make("FractionalBits", 4, 7)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvolutionLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true, false })),
                       framework::dataset::make("DataType", DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeConvolutionLayerDataset(),
                                                                                                                        framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                                        framework::dataset::make("DataType", DataType::QS16)),
                                                                                                                        framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_q);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
