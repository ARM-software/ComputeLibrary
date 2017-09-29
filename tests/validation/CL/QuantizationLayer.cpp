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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLQuantizationLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/QuantizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f32(1.0f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
const auto                         QuantizationShapes = concat(concat(concat(datasets::Small3DShapes(),
                                                                             datasets::Large3DShapes()),
                                                                      datasets::Small4DShapes()),
                                                               datasets::Large4DShapes());
} // namespace

TEST_SUITE(CL)
TEST_SUITE(QuantizationLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(QuantizationShapes, framework::dataset::make("DataType", DataType::F32)), shape, data_type)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::U8);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLQuantizationLayer quant_layer;
    quant_layer.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 4).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T>
using CLQuantizationLayerFixture = QuantizationValidationFixture<CLTensor, CLAccessor, CLQuantizationLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLQuantizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                                                                                                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLQuantizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                                                                                             framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
