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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLDequantizationLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DatatypeDataset.h"
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
TEST_SUITE(CL)
TEST_SUITE(DequantizationLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),      // Wrong input data type
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Wrong output data type
                                                       TensorInfo(TensorShape(16U, 16U, 2U, 5U), 1, DataType::QASYMM8),   // Missmatching shapes
                                                       TensorInfo(TensorShape(17U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Valid
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Valid
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(17U, 16U, 16U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { false, false, false, true, true})),
               input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLDequantizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration,
               framework::DatasetMode::ALL,
               combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::F16, DataType::F32 })),
               shape, data_type)
{
    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::QASYMM8, 1, QuantizationInfo(0.5f, -10));
    CLTensor dst = create_tensor<CLTensor>(shape, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLDequantizationLayer dequant_layer;
    dequant_layer.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    validate(src.info()->padding(), PaddingSize());
    validate(dst.info()->padding(), PaddingSize());
}

template <typename T>
using CLDequantizationLayerFixture = DequantizationValidationFixture<CLTensor, CLAccessor, CLDequantizationLayer, T>;

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDequantizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), datasets::QuantizedTypes()),
                                                                                                                framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDequantizationLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), datasets::QuantizedTypes()),
                                                                                                              framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDequantizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), datasets::QuantizedTypes()),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDequantizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), datasets::QuantizedTypes()),
                                                                                                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32

TEST_SUITE_END() // DequantizationLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
