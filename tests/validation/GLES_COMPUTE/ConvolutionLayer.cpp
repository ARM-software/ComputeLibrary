/*
 * Copyright (c) 2017-2018 ARM Limited.
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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCConvolutionLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeConvolutionLayerDataset.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<half_float::half> tolerance_f16(half_float::half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                     tolerance_num = 0.07f;                /**< Tolerance number */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    // DataType::F32,
});
} // namespace

TEST_SUITE(GC)
TEST_SUITE(ConvolutionLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallConvolutionLayerDataset(), datasets::LargeConvolutionLayerDataset()), CNNDataTypes),
               input_shape, weights_shape, bias_shape, output_shape, info, data_type)
{
    // Set fixed point position data type allowed
    int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    auto bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

    // Create tensors
    GCTensor src     = create_tensor<GCTensor>(input_shape, data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));
    GCTensor weights = create_tensor<GCTensor>(weights_shape, data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));
    GCTensor bias    = create_tensor<GCTensor>(bias_shape, bias_data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));
    GCTensor dst     = create_tensor<GCTensor>(output_shape, data_type, 1, fixed_point_position, QuantizationInfo(2.f / 255.f, 127));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    const QuantizationInfo src_quantization_info     = src.info()->quantization_info();
    const QuantizationInfo weights_quantization_info = weights.info()->quantization_info();

    // Create and configure function
    GCConvolutionLayer conv;
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

    // Validate QuantizationInfo
    ARM_COMPUTE_EXPECT(src.info()->quantization_info() == src_quantization_info, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->quantization_info() == weights_quantization_info, framework::LogLevel::ERRORS);

    //Validate padding
    //TODO(COMPMID-415) Need to validate padding?
}

template <typename T>
using GCConvolutionLayerFixture = ConvolutionValidationFixture<GCTensor, GCAccessor, GCConvolutionLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, GCConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                     framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                             framework::dataset::make("DataType",
                                                                                                                     DataType::F16)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeConvolutionLayerDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                           framework::dataset::make("DataType",
                                                                                                                   DataType::F16)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
}
} // namespace validation
} // namespace test
} // namespace arm_compute
