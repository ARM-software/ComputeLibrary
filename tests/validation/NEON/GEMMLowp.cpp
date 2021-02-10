/*
 * Copyright (c) 2017-2021 Arm Limited.
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
TEST_SUITE(NEON)
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

    // Validate padding is zero
    validate(a.info()->padding(), PaddingSize());
    validate(b.info()->padding(), PaddingSize());
    validate(c.info()->padding(), PaddingSize());
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
    framework::dataset::make("Expected", { true, false, false, false, true })),
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
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture, framework::DatasetMode::ALL, combine(datasets::SmallGEMMLowpFusedOffsetOutputUint8Dataset(),
                       framework::dataset::make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpMatrixMultiplyCoreFusedOffsetOutputFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMLowpFusedOffsetOutputUint8Dataset(),
                       framework::dataset::make("DataType", { DataType::QASYMM8 })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FusedOffsetOutput
TEST_SUITE_END() // MatrixMultiplyCore

TEST_SUITE(OutputStage)

TEST_SUITE(QuantizeDownInt32Scale)

TEST_SUITE(QASYMM8)

const auto quantize_down_int32_to_uint8_scale_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1, 2) * framework::dataset::make("result_shift", 2,
                                                      3)
                                                      * framework::dataset::make("min", 0) * framework::dataset::make("max", 255) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_uint8_scale_relu_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1,
                                                           2)
                                                           * framework::dataset::make("result_shift", 2, 3) * framework::dataset::make("min", 0, 2) * framework::dataset::make("max", 171, 174) * framework::dataset::make("addBias", { false, true });

using NEGEMMLowpQuantizeDownInt32ScaleFixture = GEMMLowpQuantizeDownInt32ToUint8ScaleValidationFixture<Tensor, Accessor, NEGEMMLowpOutputStage>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::S32), // Input not a multiple of 16
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::S32), // Wrong output data type
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(21U), 1, DataType::S32),
                                            TensorInfo(TensorShape(20U), 1, DataType::S32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Min",{        0,
                                            13,
                                           })),
    framework::dataset::make("Max",{        205,
                                            180,
                                           })),
    framework::dataset::make("Expected", { true, false })),
    a_info, b_info, output_info, min, max, expected)
{

    GEMMLowpOutputStageInfo output_stage = GEMMLowpOutputStageInfo();
    output_stage.type        = GEMMLowpOutputStageType::QUANTIZE_DOWN;
    output_stage.gemmlowp_min_bound        = min;
    output_stage.gemmlowp_max_bound        = max;
    output_stage.output_data_type = DataType::QASYMM8;

    // Lock tensors
    Status status =  NEGEMMLowpOutputStage::validate(&a_info.clone()->set_is_resizable(false),
                                                                     &b_info.clone()->set_is_resizable(false),
                                                                     &output_info.clone()->set_is_resizable(false),
                                                                     output_stage);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_CASE(NoPaddingAdded, framework::DatasetMode::PRECOMMIT)
{
    Tensor input1 = create_tensor<Tensor>(TensorShape(21U, 13U), DataType::S32);
    Tensor input2 = create_tensor<Tensor>(TensorShape(21U, 1U), DataType::S32);
    Tensor output = create_tensor<Tensor>(TensorShape(21U, 13U), DataType::QASYMM8);

    GEMMLowpOutputStageInfo output_stage = GEMMLowpOutputStageInfo();
    output_stage.type                    = GEMMLowpOutputStageType::QUANTIZE_DOWN;
    output_stage.gemmlowp_min_bound      = 0;
    output_stage.gemmlowp_max_bound      = 205;
    output_stage.output_data_type        = DataType::QASYMM8;

    NEGEMMLowpOutputStage f;
    f.configure(&input1, &input2, &output, output_stage);

    // Validate padding is zero
    validate(input1.info()->padding(), PaddingSize());
    validate(input2.info()->padding(), PaddingSize());
    validate(output.info()->padding(), PaddingSize());
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_uint8_scale_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END() // BoundedReLu

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)

const auto quantize_down_int32_to_int8_scale_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1, 2) * framework::dataset::make("result_shift", 2,
                                                     3)
                                                     * framework::dataset::make("min", 0) * framework::dataset::make("max", 0) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_int8_scale_relu_cases = framework::dataset::make("result_offset", -2, 1) * framework::dataset::make("result_mult_int", 1,
                                                          2)
                                                          * framework::dataset::make("result_shift", 2, 3) * framework::dataset::make("min", -100, -98) * framework::dataset::make("max", 71, 74) * framework::dataset::make("addBias", { false, true });

using NEGEMMLowpQuantizeDownInt32ScaleFixture = GEMMLowpQuantizeDownInt32ToInt8ScaleValidationFixture<Tensor, Accessor, NEGEMMLowpOutputStage>;

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
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8_SIGNED),
                                            TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8_SIGNED),
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Min",{        -10,
                                            -200,
                                            -113,
                                           })),
    framework::dataset::make("Max",{        105,
                                            300,
                                            -18,
                                           })),
    framework::dataset::make("Expected", { true, false, false })),
    a_info, b_info, output_info, min, max, expected)
{
    GEMMLowpOutputStageInfo output_stage = GEMMLowpOutputStageInfo();
    output_stage.type        = GEMMLowpOutputStageType::QUANTIZE_DOWN;
    output_stage.gemmlowp_min_bound        = min;
    output_stage.gemmlowp_max_bound        = max;
    output_stage.output_data_type = DataType::QASYMM8_SIGNED;

    // Lock tensors
    Status status =  NEGEMMLowpOutputStage::validate(&a_info.clone()->set_is_resizable(false),
                                                                     &b_info.clone()->set_is_resizable(false),
                                                                     &output_info.clone()->set_is_resizable(false),
                                                                     output_stage);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_int8_scale_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ScaleFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), quantize_down_int32_to_int8_scale_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END() // BoundedReLu

TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // QuantizeDownInt32Scale

TEST_SUITE(QuantizeDownInt32ToUint8ScaleByFixedPoint)

const auto quantize_down_int32_to_uint8_scale_by_fixedpoint_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                    2)
                                                                    * framework::dataset::make("result_offset_after_shift", 2, 3) * framework::dataset::make("min", 0) * framework::dataset::make("max", 255) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_uint8_scale_by_fixedpoint_relu_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                         2)
                                                                         * framework::dataset::make("result_offset_after_shift", 2, 3) * framework::dataset::make("min", 0, 2) * framework::dataset::make("max", 171, 174) * framework::dataset::make("addBias", { false, true });

using NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointFixture =
    GEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointValidationFixture<Tensor, Accessor, NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint>;

using NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointFixture =
    GEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointValidationFixture<Tensor, Accessor, NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::S32), // Input not a multiple of 16
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::S32), // Wrong output data type
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(21U), 1, DataType::S32),
                                            TensorInfo(TensorShape(20U), 1, DataType::S32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Min",{        0,
                                            13,
                                           })),
    framework::dataset::make("Max",{        205,
                                            180,
                                           })),
    framework::dataset::make("Expected", { true, false })),
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

TEST_SUITE(QuantizeDownInt32ToInt8ScaleByFixedPoint)

const auto quantize_down_int32_to_int8_scale_by_fixedpoint_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                   2)
                                                                   * framework::dataset::make("result_offset_after_shift", 2, 3) * framework::dataset::make("min", -128) * framework::dataset::make("max", 128) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_int8_scale_by_fixedpoint_relu_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                        2)
                                                                        * framework::dataset::make("result_offset_after_shift", 2, 3) * framework::dataset::make("min", -2, 0) * framework::dataset::make("max", 1, 3) * framework::dataset::make("addBias", { false, true });

using NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointFixture =
    GEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointValidationFixture<Tensor, Accessor, NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
        framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::F32), // Invalid input data type
                                                 TensorInfo(TensorShape(20U, 13U), 1, DataType::S32), // Wrong output data type
                                                 TensorInfo(TensorShape(21U, 13U), 1, DataType::S32),
        }),
        framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(21U), 1, DataType::S32),
                                                TensorInfo(TensorShape(20U), 1, DataType::S32),
                                                TensorInfo(TensorShape(21U), 1, DataType::S32),
        })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8_SIGNED),
                                                TensorInfo(TensorShape(20U, 13U), 1, DataType::S32),
                                                TensorInfo(TensorShape(21U, 13U), 1, DataType::QASYMM8_SIGNED),
        })),
        framework::dataset::make("Min",{ -110,
                                         -113,
                                         -113,
        })),
        framework::dataset::make("Max",{ 87,
                                         97,
                                         97,
        })),
        framework::dataset::make("Expected", { false, false, true })),
               a_info, b_info, output_info, min, max, expected)
{
    // Lock tensors
    Status status =  NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::validate(&a_info.clone()->set_is_resizable(false),
                                                                                  &b_info.clone()->set_is_resizable(false),
                                                                                  &output_info.clone()->set_is_resizable(false),
                                                                                  min,
                                                                                  max);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_int8_scale_by_fixedpoint_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE(BoundedReLu)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_int8_scale_by_fixedpoint_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // BoundedReLu
TEST_SUITE_END() // QuantizeDownInt32ToInt8ScaleByFixedPoint

TEST_SUITE(QuantizeDownInt32ToInt16ScaleByFixedPoint)

const auto quantize_down_int32_to_int16_scale_by_fixedpoint_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                    2)
                                                                    * framework::dataset::make("min", -32768) * framework::dataset::make("max", 32767) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_int16_scale_by_fixedpoint_relu_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600, 254601602) * framework::dataset::make("result_shift", 1,
                                                                         2)
                                                                         * framework::dataset::make("min", -2, 0) * framework::dataset::make("max", 1, 3) * framework::dataset::make("addBias", { false, true });
const auto quantize_down_int32_to_int16_scale_by_fixedpoint_multgreat1_cases = framework::dataset::make("result_fixedpoint_multiplier", 1073741823,
                                                                                                        1073741825)
                                                                               * framework::dataset::make("result_shift", -3,
                                                                                                          -2)
                                                                               * framework::dataset::make("min", -32768) * framework::dataset::make("max", 32767) * framework::dataset::make("addBias", { false, true });

const auto quantize_down_int32_to_int16_scale_by_fixedpoint_multgreat1_relu_cases = framework::dataset::make("result_fixedpoint_multiplier", 254601600,
                                                                                                             254601602)
                                                                                    * framework::dataset::make("result_shift", -3,
                                                                                                               -1)
                                                                                    * framework::dataset::make("min", -2, 0) * framework::dataset::make("max", 1, 3) * framework::dataset::make("addBias", { false, true });

using NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointFixture =
    GEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointValidationFixture<Tensor, Accessor, NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
    framework::dataset::make("InputAInfo", { TensorInfo(TensorShape(21U, 13U), 1, DataType::S32), // Input not a multiple of 16
                                             TensorInfo(TensorShape(20U, 13U), 1, DataType::S32), // Wrong output data type
                                          }),
    framework::dataset::make("InputBInfo",{ TensorInfo(TensorShape(21U), 1, DataType::S32),
                                            TensorInfo(TensorShape(20U), 1, DataType::S32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(21U, 13U), 1, DataType::QSYMM16),
                                            TensorInfo(TensorShape(20U, 13U), 1, DataType::S32),
                                           })),
    framework::dataset::make("Min",{        -205,
                                            -180,
                                           })),
    framework::dataset::make("Max",{        205,
                                            180,
                                           })),
    framework::dataset::make("Expected", { true, false })),
    a_info, b_info, output_info, min, max, expected)
{
    // Lock tensors
    Status status =  NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::validate(&a_info.clone()->set_is_resizable(false),
                                                                                 &b_info.clone()->set_is_resizable(false),
                                                                                 &output_info.clone()->set_is_resizable(false),
                                                                                 min,
                                                                                 max);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(NoRelu)
TEST_SUITE(MultSmallerEq1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_int16_scale_by_fixedpoint_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // MultSmallerEq1
TEST_SUITE(MultGreater1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_int16_scale_by_fixedpoint_multgreat1_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // MultGreater1
TEST_SUITE_END() // NoRelu
TEST_SUITE(BoundedReLu)
TEST_SUITE(MultSmallerEq1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_int16_scale_by_fixedpoint_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // MultSmallerEq1
TEST_SUITE(MultGreater1)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointFixture, framework::DatasetMode::ALL, combine(datasets::SmallShapes(),
                       quantize_down_int32_to_int16_scale_by_fixedpoint_multgreat1_relu_cases))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // MultGreater1
TEST_SUITE_END() // BoundedReLu
TEST_SUITE_END() // QuantizeDownInt32ToInt16ScaleByFixedPoint
TEST_SUITE_END() // OutputStage
TEST_SUITE_END() // GEMMLowp
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
