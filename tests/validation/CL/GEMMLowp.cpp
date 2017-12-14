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
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
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
TEST_SUITE(CL)
TEST_SUITE(GEMMLowp)

TEST_SUITE(MatrixMultiplyCore)
using CLGEMMLowpMatrixMultiplyCoreFixture = GEMMLowpMatrixMultiplyCoreValidationFixture<CLTensor, CLAccessor, CLGEMMLowpMatrixMultiplyCore>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, framework::dataset::concat(datasets::SmallGEMMLowpDataset(), datasets::LargeGEMMLowpDataset()),
               shape_a, shape_b, shape_c, a_offset, b_offset)
{
    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_a, DataType::QASYMM8);
    CLTensor b = create_tensor<CLTensor>(shape_b, DataType::QASYMM8);
    CLTensor c = create_tensor<CLTensor>(shape_c, DataType::S32);

    a.info()->set_quantization_info(QuantizationInfo(1.0f / 255, a_offset));
    b.info()->set_quantization_info(QuantizationInfo(1.0f / 255, b_offset));

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMLowpMatrixMultiplyCore gemmlowp_mm;
    gemmlowp_mm.configure(&a, &b, &c);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::ALL, datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::NIGHTLY, datasets::LargeGEMMLowpDataset())
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // MatrixMultiplyCore

TEST_SUITE(OutputStage)
TEST_SUITE(QuantizeDownInt32ToUint8Scale)

const auto quantize_down_int32_to_uint8_scale_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1, 2) * framework::dataset::make("result_shift", 2,
                                                      3)
                                                      * framework::dataset::make("min", 0) * framework::dataset::make("max", 0) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_uint8_scale_relu_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1,
                                                           2)
                                                           * framework::dataset::make("result_shift", 2, 3) * framework::dataset::make("min", 0, 2) * framework::dataset::make("max", 171, 173) * framework::dataset::make("addBias", { false, true });

using CLGEMMLowpQuantizeDownInt32ToUint8ScaleFixture = GEMMLowpQuantizeDownInt32ToUint8ScaleValidationFixture<CLTensor, CLAccessor, CLGEMMLowpQuantizeDownInt32ToUint8Scale>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), quantize_down_int32_to_uint8_scale_cases),
               shape, result_offset, result_mult_int, result_shift, min, max, add_bias)
{
    TensorShape shape_bias(shape[0]);

    // Create tensors
    CLTensor in   = create_tensor<CLTensor>(shape, DataType::S32);
    CLTensor bias = create_tensor<CLTensor>(shape_bias, DataType::S32);
    CLTensor out  = create_tensor<CLTensor>(shape, DataType::QASYMM8);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMLowpQuantizeDownInt32ToUint8Scale output_stage;
    output_stage.configure(&in, add_bias ? &bias : nullptr, &out, result_offset, result_mult_int, result_shift, min, max);

    // Validate valid region input and output
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(in.info()->valid_region(), valid_region);
    validate(out.info()->valid_region(), valid_region);

    // Validate valid region bias
    if(add_bias)
    {
        const ValidRegion valid_region_bias = shape_to_valid_region(shape_bias);
        validate(bias.info()->valid_region(), valid_region_bias);
    }

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(in.info()->padding(), padding);
    validate(out.info()->padding(), padding);

    if(add_bias)
    {
        validate(bias.info()->padding(), padding);
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), quantize_down_int32_to_uint8_scale_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_relu_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(),
                       quantize_down_int32_to_uint8_scale_relu_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // BoundedReLu
TEST_SUITE_END() // QuantizeDownInt32ToUint8Scale

TEST_SUITE(QuantizeDownInt32ToUint8ScaleByFixedPoint)

const auto quantize_down_int32_to_uint8_scale_by_fixedpoint_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                    2)
                                                                    * framework::dataset::make("result_offset_after_shift", 2, 3) * framework::dataset::make("min", 0) * framework::dataset::make("max", 0) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_uint8_scale_by_fixedpoint_relu_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                         2)
                                                                         * framework::dataset::make("result_offset_after_shift", 2, 3) * framework::dataset::make("min", 0, 2) * framework::dataset::make("max", 171, 174) * framework::dataset::make("addBias", { false, true });

using CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture =
    GEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointValidationFixture<CLTensor, CLAccessor, CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                   quantize_down_int32_to_uint8_scale_by_fixedpoint_cases),
               shape, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max, add_bias)
{
    TensorShape shape_bias(shape[0]);

    // Create tensors
    CLTensor in   = create_tensor<CLTensor>(shape, DataType::S32);
    CLTensor bias = create_tensor<CLTensor>(shape_bias, DataType::S32);
    CLTensor out  = create_tensor<CLTensor>(shape, DataType::QASYMM8);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint output_stage;
    output_stage.configure(&in, add_bias ? &bias : nullptr, &out, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);

    // Validate valid region input and output
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(in.info()->valid_region(), valid_region);
    validate(out.info()->valid_region(), valid_region);

    // Validate valid region bias
    if(add_bias)
    {
        const ValidRegion valid_region_bias = shape_to_valid_region(shape_bias);
        validate(bias.info()->valid_region(), valid_region_bias);
    }

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(in.info()->padding(), padding);
    validate(out.info()->padding(), padding);

    if(add_bias)
    {
        validate(bias.info()->padding(), padding);
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_relu_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_relu_cases))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // BoundedReLu
TEST_SUITE_END() // QuantizeDownInt32ToUint8ScaleByFixedPoint

TEST_SUITE_END() // OutputStage
TEST_SUITE_END() // GEMMLowp
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute