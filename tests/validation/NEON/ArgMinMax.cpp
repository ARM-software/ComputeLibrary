/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/runtime/NEON/functions/NEArgMinMaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SplitDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ArgMinMaxFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto OpsDataset   = framework::dataset::make("Operation", { ReductionOperation::ARG_IDX_MIN, ReductionOperation::ARG_IDX_MAX });
const auto AxisDataset  = framework::dataset::make("Axis", { 0, 1, 2, 3 });
const auto QInfoDataset = framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) });

const auto ArgMinMaxSmallDatasetAxis0 = framework::dataset::make("Shape",
{
    TensorShape{ 1U, 5U },
    TensorShape{ 2U, 3U },
    TensorShape{ 1U },
    TensorShape{ 3U },
    TensorShape{ 2U },
    TensorShape{ 5U },
    TensorShape{ 17U },
    TensorShape{ 15U, 2U },
});
using ArgMinMaxSmallDataset = datasets::Small4DShapes;
using ArgMinMaxLargeDataset = datasets::Large4DShapes;
}

TEST_SUITE(NEON)
TEST_SUITE(ArgMinMax)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid axis
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid output shape
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32) // Invalid operation
        }),
        framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(27U, 3U, 1U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::S32),
                                                 TensorInfo(TensorShape(32U, 16U, 1U, 2U), 1, DataType::F32)
        })),
        framework::dataset::make("Axis", { 4, 0, 2, 0 })),
        framework::dataset::make("Operation", { ReductionOperation::ARG_IDX_MAX, ReductionOperation::ARG_IDX_MAX, ReductionOperation::ARG_IDX_MAX, ReductionOperation::MEAN_SUM })),
        framework::dataset::make("Expected", { false, false, true, false })),
        input_info, output_info, axis, operation, expected)
{
    const Status status = NEArgMinMaxLayer::validate(&input_info.clone()->set_is_resizable(false), axis, &output_info.clone()->set_is_resizable(false), operation);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T1, typename T2>
using NEArgMinMaxValidationFixture = ArgMinMaxValidationFixture<Tensor, Accessor, NEArgMinMaxLayer, T1, T2>;

using NEArgMinMaxValidationFixture_S32_S32 = NEArgMinMaxValidationFixture<int32_t, int32_t>;
using NEArgMinMaxValidationFixture_F16_S32 = NEArgMinMaxValidationFixture<half, int32_t>;
using NEArgMinMaxValidationFixture_F32_S32 = NEArgMinMaxValidationFixture<float, int32_t>;
using NEArgMinMaxValidationFixture_F32_S64 = NEArgMinMaxValidationFixture<float, int64_t>;

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmallAxis0,
                       NEArgMinMaxValidationFixture_S32_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDatasetAxis0,
                                                       framework::dataset::make("DataTypeIn", DataType::S32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       framework::dataset::make("Axis", { 0 })),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArgMinMaxValidationFixture_S32_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset(),
                                                       framework::dataset::make("DataTypeIn", DataType::S32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArgMinMaxValidationFixture_S32_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(ArgMinMaxLargeDataset(),
                                                       framework::dataset::make("DataTypeIn", DataType::S32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArgMinMaxValidationFixture_F16_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset(),
                                                       framework::dataset::make("DataTypeIn", DataType::F16)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArgMinMaxValidationFixture_F16_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(ArgMinMaxLargeDataset(),
                                                       framework::dataset::make("DataTypeIn", DataType::F16)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArgMinMaxValidationFixture_F32_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset(),
                                                       framework::dataset::make("DataTypeIn", DataType::F32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall_F32_S64,
                       NEArgMinMaxValidationFixture_F32_S64,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset(),
                                                       framework::dataset::make("DataTypeIn", DataType::F32)),
                                               framework::dataset::make("DataTypeOut", DataType::S64)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArgMinMaxValidationFixture_F32_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(ArgMinMaxLargeDataset(),
                                                       framework::dataset::make("DataTypeIn", DataType::F32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T1, typename T2>
using NEArgMinMaxQuantizedValidationFixture = ArgMinMaxValidationQuantizedFixture<Tensor, Accessor, NEArgMinMaxLayer, T1, T2>;

using NEArgMinMaxQuantizedValidationFixture_U8_S32 = NEArgMinMaxQuantizedValidationFixture<uint8_t, int32_t>;
using NEArgMinMaxQuantizedValidationFixture_S8_S32 = NEArgMinMaxQuantizedValidationFixture<int8_t, int32_t>;

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArgMinMaxQuantizedValidationFixture_U8_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(ArgMinMaxSmallDataset(),
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArgMinMaxQuantizedValidationFixture_U8_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(ArgMinMaxLargeDataset(),
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEArgMinMaxQuantizedValidationFixture_S8_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(ArgMinMaxSmallDataset(),
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8_SIGNED)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEArgMinMaxQuantizedValidationFixture_S8_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(ArgMinMaxLargeDataset(),
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8_SIGNED)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // ArgMinMax
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
