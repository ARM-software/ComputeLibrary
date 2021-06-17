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
TEST_SUITE_END() // GEMMLowp
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
