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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLArgMinMaxLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
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
const auto ArgMinMaxSmallDataset = framework::dataset::make("Shape",
{
    TensorShape{ 1U, 7U, 1U, 3U },
    TensorShape{ 3U, 1U, 3U, 2U },
    TensorShape{ 2U, 1U, 3U, 2U },
    TensorShape{ 149U, 5U, 1U, 2U },
    TensorShape{ 166U, 5U, 1U, 2U },
    TensorShape{ 322U, 5U, 1U, 2U },
    TensorShape{ 128U, 5U, 21U, 3U },
    TensorShape{ 2560, 2U, 2U, 2U },
});

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

const auto OpsDataset   = framework::dataset::make("Operation", { ReductionOperation::ARG_IDX_MIN, ReductionOperation::ARG_IDX_MAX });
const auto AxisDataset  = framework::dataset::make("Axis", { 0, 1, 2, 3 });
const auto QInfoDataset = framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) });

const auto ArgMinMaxLargeDataset = framework::dataset::make("Shape",
{ TensorShape{ 517U, 123U, 13U, 2U } });
} // namespace
TEST_SUITE(CL)
TEST_SUITE(ArgMinMax)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid axis
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid output shape
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32), // Invalid operation
                                                TensorInfo(TensorShape(32U, 16U, 16U, 2U), 1, DataType::F32) // Not allowed keeping the dimension 
        }),
        framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(27U, 3U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(27U, 3U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::S32),
                                                 TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(32U, 16U, 1U, 2U), 1, DataType::U32)
        })),
        framework::dataset::make("Axis", { 4, 0, 2, 0, 2 })),
        framework::dataset::make("Operation", { ReductionOperation::ARG_IDX_MAX, ReductionOperation::ARG_IDX_MAX, ReductionOperation::ARG_IDX_MAX, ReductionOperation::MEAN_SUM, ReductionOperation::ARG_IDX_MAX })),
        framework::dataset::make("Expected", { false, false, true, false, false })),
        input_info, output_info, axis, operation, expected)
{
    const Status status = CLArgMinMaxLayer::validate(&input_info.clone()->set_is_resizable(false), axis, &output_info.clone()->set_is_resizable(false), operation);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T1, typename T2>
using CLArgMinMaxValidationFixture = ArgMinMaxValidationFixture<CLTensor, CLAccessor, CLArgMinMaxLayer, T1, T2>;

using CLArgMinMaxValidationFixture_S32_S32 = CLArgMinMaxValidationFixture<int32_t, int32_t>;
using CLArgMinMaxValidationFixture_F16_S32 = CLArgMinMaxValidationFixture<half, int32_t>;
using CLArgMinMaxValidationFixture_F32_S32 = CLArgMinMaxValidationFixture<float, int32_t>;
using CLArgMinMaxValidationFixture_F32_S64 = CLArgMinMaxValidationFixture<float, int64_t>;

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmallAxis0,
                       CLArgMinMaxValidationFixture_S32_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDatasetAxis0,
                                                       framework::dataset::make("DataTypeIn", DataType::S32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       framework::dataset::make("Axis", { 0 })),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLArgMinMaxValidationFixture_S32_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset,
                                                       framework::dataset::make("DataTypeIn", DataType::S32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLArgMinMaxValidationFixture_S32_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(ArgMinMaxLargeDataset,
                                                       framework::dataset::make("DataTypeIn", DataType::S32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // S32

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLArgMinMaxValidationFixture_F16_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset,
                                                       framework::dataset::make("DataTypeIn", DataType::F16)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLArgMinMaxValidationFixture_F16_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(ArgMinMaxLargeDataset,
                                                       framework::dataset::make("DataTypeIn", DataType::F16)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLArgMinMaxValidationFixture_F32_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset,
                                                       framework::dataset::make("DataTypeIn", DataType::F32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall_F32_S64,
                       CLArgMinMaxValidationFixture_F32_S64,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(ArgMinMaxSmallDataset,
                                                       framework::dataset::make("DataTypeIn", DataType::F32)),
                                               framework::dataset::make("DataTypeOut", DataType::S64)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLArgMinMaxValidationFixture_F32_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(ArgMinMaxLargeDataset,
                                                       framework::dataset::make("DataTypeIn", DataType::F32)),
                                               framework::dataset::make("DataTypeOut", DataType::S32)),
                                       AxisDataset),
                               OpsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T1, typename T2>
using CLArgMinMaxQuantizedValidationFixture = ArgMinMaxValidationQuantizedFixture<CLTensor, CLAccessor, CLArgMinMaxLayer, T1, T2>;

using CLArgMinMaxQuantizedValidationFixture_U8_S32 = CLArgMinMaxQuantizedValidationFixture<uint8_t, int32_t>;
using CLArgMinMaxQuantizedValidationFixture_S8_S32 = CLArgMinMaxQuantizedValidationFixture<int8_t, int32_t>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLArgMinMaxQuantizedValidationFixture_U8_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(ArgMinMaxSmallDataset,
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLArgMinMaxQuantizedValidationFixture_U8_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(ArgMinMaxLargeDataset,
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLArgMinMaxQuantizedValidationFixture_S8_S32,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(ArgMinMaxSmallDataset,
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8_SIGNED)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLArgMinMaxQuantizedValidationFixture_S8_S32,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(ArgMinMaxLargeDataset,
                                                               framework::dataset::make("DataTypeIn", DataType::QASYMM8_SIGNED)),
                                                       framework::dataset::make("DataTypeOut", DataType::S32)),
                                               AxisDataset),
                                       OpsDataset),
                               QInfoDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ArgMinMax
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
