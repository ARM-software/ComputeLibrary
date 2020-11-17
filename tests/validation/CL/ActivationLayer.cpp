/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/RuntimeContext.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ActivationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_qsymm16(1.f);

/** Define tolerance of the activation layer.
 *
 * @param[in] activation The activation function used.
 * @param[in] data_type  Data type.
 *
 * @return Tolerance depending on the activation function.
 */
AbsoluteTolerance<float> tolerance(ActivationLayerInfo::ActivationFunction activation, DataType data_type)
{
    constexpr float epsilon = 1e-6f;

    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LINEAR:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.2f : epsilon);
        case ActivationLayerInfo::ActivationFunction::SQUARE:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.1f : epsilon);
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.001f : epsilon);
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.00001f : epsilon);
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
        case ActivationLayerInfo::ActivationFunction::ELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.01f : 0.00001f);
        case ActivationLayerInfo::ActivationFunction::TANH:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.001f : 0.00001f);
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
            return AbsoluteTolerance<float>(data_type == DataType::F16 ? 0.01f : epsilon);
        default:
            return AbsoluteTolerance<float>(epsilon);
    }
}

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32
});

/** Input data sets. */
const auto ActivationDataset = combine(combine(framework::dataset::make("InPlace", { false, true }), datasets::ActivationFunctions()), framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));

} // namespace

TEST_SUITE(CL)
TEST_SUITE(ActivationLayer)
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data types
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8), // Invalid quantization info
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QSYMM16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QSYMM16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QSYMM16), // Invalid activation function for QSYMM16
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QSYMM16, QuantizationInfo(1.f / 32768.f, 0)),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QSYMM16, QuantizationInfo(1.f / 32768.f, 0)),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QSYMM16, QuantizationInfo(1.f / 32768.f, 0)),
                                                     })),
               framework::dataset::make("ActivationInfo", { ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SQRT),
                                                          })),
               framework::dataset::make("Expected", { false, true, true, true, false, false, true, true, false })),
               input_info, output_info, act_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLActivationLayer::validate(&input_info.clone()->set_is_resizable(false), (output_info.total_size() == 0) ? nullptr : &output_info.clone()->set_is_resizable(false), act_info)) == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

/** [CLActivationLayerFixture snippet] **/
template <typename T>
using CLActivationLayerFixture = ActivationValidationFixture<CLTensor, CLAccessor, CLActivationLayer, T>;
/** [CLActivationLayerFixture snippet] **/

TEST_SUITE(Float)
TEST_SUITE(FP16)
/** [CLActivationLayer Test snippet] **/
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ActivationDataset),
                                                                                                      framework::dataset::make("DataType",
                                                                                                              DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
/** [CLActivationLayer Test snippet] **/
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ActivationDataset), framework::dataset::make("DataType",
                                                                                                       DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLActivationLayerQuantizedFixture = ActivationValidationQuantizedFixture<CLTensor, CLAccessor, CLActivationLayer, T>;

const auto QuantizedActivationDataset8 = combine(combine(framework::dataset::make("InPlace", { false }),
                                                         concat(datasets::ActivationFunctionsQuantized(), framework::dataset::make("ActivationFunction", ActivationLayerInfo::ActivationFunction::HARD_SWISH))),
                                                 framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));

const auto QuantizedActivationDataset16 = combine(combine(framework::dataset::make("InPlace", { false }),
                                                          datasets::ActivationFunctionsQuantized()),
                                                  framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), QuantizedActivationDataset8),
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::QASYMM8)),
                                                                                                                  framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 128.0f) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), QuantizedActivationDataset8),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::QASYMM8_SIGNED)),
                                                                                                                 framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 10.0f) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance(_function, _data_type));
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLActivationLayerQuantizedFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallShapes(), QuantizedActivationDataset16),
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::QSYMM16)),
                                                                                                                  framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 32768.f, 0) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // ActivationLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
