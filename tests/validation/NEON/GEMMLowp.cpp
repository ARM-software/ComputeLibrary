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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpAssemblyMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/GEMMLowpFusedOffsetOutputDataset.h"
#include "tests/datasets/LargeGEMMLowpDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SmallGEMMLowpDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMLowpAssemblyFixture.h"
#include "tests/validation/fixtures/GEMMLowpFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto data_matrix_multiply = framework::dataset::make("M", 12, 20) * framework::dataset::make("N", 12, 20) * framework::dataset::make("K", 16);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ASSEMBLY_MATRIX_MULTIPLY)

using NEGEMMAssemblyFixture_S8 = GEMMLowpAssemblyFixture<Tensor, Accessor, NEGEMMLowpAssemblyMatrixMultiplyCore, int8_t>;
using NEGEMMAssemblyFixture_U8 = GEMMLowpAssemblyFixture<Tensor, Accessor, NEGEMMLowpAssemblyMatrixMultiplyCore, uint8_t>;

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMAssemblyFixture_S8, framework::DatasetMode::PRECOMMIT, data_matrix_multiply)
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMAssemblyFixture_U8, framework::DatasetMode::PRECOMMIT, data_matrix_multiply)
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(GEMMLowp)
TEST_SUITE(MatrixMultiplyCore)
using NEGEMMLowpMatrixMultiplyCoreFixture = GEMMLowpMatrixMultiplyCoreValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;

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
    NEGEMMLowpMatrixMultiplyCore gemmlowp_mm;
    gemmlowp_mm.configure(&a, &b, nullptr, &c);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Input not a multiple of 4
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::S32),                                 // Mismatching data type
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Invalid dimensions
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)), // Invalid dimensions
                                             TensorInfo(TensorShape(16U, 32U), 1, DataType::QASYMM8, QuantizationInfo(1.f/255, 10)),
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(33U, 21U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                            TensorInfo(TensorShape(64U, 16U), 1, DataType::QASYMM8, QuantizationInfo(1.f/256, 10)),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(33U, 13U), 1, DataType::S32),
                                            TensorInfo(TensorShape(8U, 11U), 1, DataType::S32),
                                            TensorInfo(TensorShape(64U, 32U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Expected", { false, false, false, false, true })),
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

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::ALL, datasets::SmallGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpMatrixMultiplyCoreFixture, framework::DatasetMode::NIGHTLY, datasets::LargeGEMMLowpDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

using NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture = GEMMLowpMatrixMultiplyCoreFusedOffsetOutputValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;
TEST_SUITE(FusedOffsetOutput)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture, framework::DatasetMode::ALL, datasets::SmallGEMMLowpFusedOffsetOutputDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture, framework::DatasetMode::NIGHTLY, datasets::LargeGEMMLowpFusedOffsetOutputDataset())
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FusedOffsetOutput
TEST_SUITE_END() // MatrixMultiplyCore

TEST_SUITE(OutputStage)

TEST_SUITE(QuantizeDownInt32ToUint8Scale)

const auto quantize_down_int32_to_uint8_scale_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1, 2) * framework::dataset::make("result_shift", 2,
                                                      3)
                                                      * framework::dataset::make("min", 0) * framework::dataset::make("max", 0) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_uint8_scale_relu_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1,
                                                           2)
                                                           * framework::dataset::make("result_shift", 2, 3) * framework::dataset::make("min", 0, 2) * framework::dataset::make("max", 171, 174) * framework::dataset::make("addBias", { false, true });

using NEGEMMLowpQuantizeDownInt32ToUint8ScaleFixture = GEMMLowpQuantizeDownInt32ToUint8ScaleValidationFixture<Tensor, Accessor, NEGEMMLowpQuantizeDownInt32ToUint8Scale>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::S32), // Input not a multiple of 16
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::S32), // Invalid min and max
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::S32), // Wrong output data type
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(21U), 1, DataType::S32),
                                            TensorInfo(TensorShape(21U), 1, DataType::S32),
                                            TensorInfo(TensorShape(20U), 1, DataType::S32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Min",{        0,
                                            8,
                                            13,
                                           })),
    framework::dataset::make("Max",{        205,
                                            300,
                                            180,
                                           })),
    framework::dataset::make("Expected", { true, false, false })),
    a_info, b_info, output_info, min, max, expected)
{
    // Lock tensors
    Status status =  NEGEMMLowpQuantizeDownInt32ToUint8Scale::validate(&a_info.clone()->set_is_resizable(false),
                                                                     &b_info.clone()->set_is_resizable(false),
                                                                     &output_info.clone()->set_is_resizable(false),
                                                                     min,
                                                                     max);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_cases),
               shape, result_offset, result_mult_int, result_shift, min, max, add_bias)
{
    TensorShape shape_bias(shape[0]);

    // Create tensors
    Tensor in   = create_tensor<Tensor>(shape, DataType::S32);
    Tensor bias = create_tensor<Tensor>(shape_bias, DataType::S32);
    Tensor out  = create_tensor<Tensor>(shape, DataType::QASYMM8);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEGEMMLowpQuantizeDownInt32ToUint8Scale output_stage;
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
    const PaddingSize padding(0);
    validate(in.info()->padding(), padding);
    validate(out.info()->padding(), padding);

    if(add_bias)
    {
        validate(bias.info()->padding(), padding);
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), quantize_down_int32_to_uint8_scale_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpQuantizeDownInt32ToUint8ScaleFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), quantize_down_int32_to_uint8_scale_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
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

using NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture =
    GEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointValidationFixture<Tensor, Accessor, NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::S32), // Input not a multiple of 16
                                             TensorInfo(TensorShape(21U, 13U), 1, DataType::S32), // Invalid min and max
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::S32), // Wrong output data type
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(21U), 1, DataType::S32),
                                            TensorInfo(TensorShape(21U), 1, DataType::S32),
                                            TensorInfo(TensorShape(20U), 1, DataType::S32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Min",{        0,
                                            8,
                                            13,
                                           })),
    framework::dataset::make("Max",{        205,
                                            300,
                                            180,
                                           })),
    framework::dataset::make("Expected", { true, false, false })),
    a_info, b_info, output_info, min, max, expected)
{
    // Lock tensors
    Status status =  NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(&a_info.clone()->set_is_resizable(false),
                                                                                 &b_info.clone()->set_is_resizable(false),
                                                                                 &output_info.clone()->set_is_resizable(false),
                                                                                 min,
                                                                                 max);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                                                                   quantize_down_int32_to_uint8_scale_by_fixedpoint_cases),
               shape, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max, add_bias)
{
    TensorShape shape_bias(shape[0]);

    // Create tensors
    Tensor in   = create_tensor<Tensor>(shape, DataType::S32);
    Tensor bias = create_tensor<Tensor>(shape_bias, DataType::S32);
    Tensor out  = create_tensor<Tensor>(shape, DataType::QASYMM8);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint output_stage;
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
    const PaddingSize padding(0);
    validate(in.info()->padding(), padding);
    validate(out.info()->padding(), padding);

    if(add_bias)
    {
        validate(bias.info()->padding(), padding);
    }
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(),
                       quantize_down_int32_to_uint8_scale_by_fixedpoint_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // BoundedReLu

TEST_SUITE_END() // QuantizeDownInt32ToUint8ScaleByFixedPoint
TEST_SUITE_END() // OutputStage

TEST_SUITE_END() // GEMMLowp
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
