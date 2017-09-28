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
#include "arm_compute/runtime/CL/functions/CLDequantizationLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DequantizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto DequantizationShapes = concat(concat(concat(datasets::Small3DShapes(),
                                                       datasets::Large3DShapes()),
                                                datasets::Small4DShapes()),
                                         datasets::Large4DShapes());
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DequantizationLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(DequantizationShapes, framework::dataset::make("DataType", DataType::U8)), shape, data_type)
{
    TensorShape shape_min_max = shape;
    shape_min_max.set(Window::DimX, 2);

    // Remove Y and Z dimensions and keep the batches
    shape_min_max.remove_dimension(1);
    shape_min_max.remove_dimension(1);

    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst     = create_tensor<CLTensor>(shape, DataType::F32);
    CLTensor min_max = create_tensor<CLTensor>(shape_min_max, DataType::F32);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(min_max.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLDequantizationLayer dequant_layer;
    dequant_layer.configure(&src, &dst, &min_max);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate valid region of min_max tensor
    const ValidRegion valid_region_min_max = shape_to_valid_region(shape_min_max);
    validate(min_max.info()->valid_region(), valid_region_min_max);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 4).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);

    // Validate padding of min_max tensor
    const PaddingSize padding_min_max = PaddingCalculator(shape_min_max.x(), 2).required_padding();
    validate(min_max.info()->padding(), padding_min_max);
}

template <typename T>
using CLDequantizationLayerFixture = DequantizationValidationFixture<CLTensor, CLAccessor, CLDequantizationLayer, T>;

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDequantizationLayerFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                                                                                                                   framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDequantizationLayerFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                                                                                                 framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
