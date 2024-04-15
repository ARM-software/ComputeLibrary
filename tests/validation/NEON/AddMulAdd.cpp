/*
 * Copyright (c) 2023 Arm Limited.
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
#ifdef __aarch64__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEAddMulAdd.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/AddMulAddFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f);     /**< Tolerance for floating point tests */
const AbsoluteTolerance<half>      tolerance_fp16(half(0.1f)); /**< Tolerance for 16-bit floating point tests */
constexpr AbsoluteTolerance<float> tolerance_quant(1);         /**< Tolerance for quantized tests */

const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),

    // Boundaries are aligned with Quantized Data ranges -- DOUBLE check before changing
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, -2.f)
});

// QASYMM8 test quantizations
const auto qasymm8_input1_qinfo_set = framework::dataset::make("Input1QInfo", { QuantizationInfo(0.1, 10) }); // Representable Range: [-1, 24.5]
const auto qasymm8_input2_qinfo_set = framework::dataset::make("Input2QInfo", { QuantizationInfo(0.2, 60) }); // Representable Range: [-12, 39]
const auto qasymm8_bn_mul_qinfo_set = framework::dataset::make("BnMulInfo", { QuantizationInfo(0.001, 55) }); // Representable Range: [-0.11, 0.2]
const auto qasymm8_bn_add_qinfo_set = framework::dataset::make("BnAddInfo", { QuantizationInfo(0.02, 20) });  // Representable Range: [-0.4, 4.7]

// Representable Range: [-9.36, 51.84], Expected F32 range: [-13, 63.5], leaving some space for saturation
const auto qasymm8_add_output_qinfo_set = framework::dataset::make("AddOutputInfo", { QuantizationInfo(0.24, 39) });

// Representable Range: [-4.8, 10.5], Expected FP32 range: [-6.985, 12.7], leaving some space for saturation
// This range also makes sense with the activation boundaries above, i.e. [-2, 8] for LU_BOUNDED_RELU and [0, 6] for BOUNDED_RELU
const auto qasymm8_final_output_qinfo_set = framework::dataset::make("FinalOutputInfo", { QuantizationInfo(0.06, 80) });

// QASYMM8_SIGNED test quantizations
const auto qasymm8_signed_input1_qinfo_set = framework::dataset::make("Input1QInfo", { QuantizationInfo(0.1, 10) });  // Representable Range: [-13.8, 11.7]
const auto qasymm8_signed_input2_qinfo_set = framework::dataset::make("Input2QInfo", { QuantizationInfo(0.2, -60) }); // Representable Range: [-13.6, 39.4]
const auto qasymm8_signed_bn_mul_qinfo_set = framework::dataset::make("BnMulInfo", { QuantizationInfo(0.001, 55) });  // Representable Range: [-0.183, 0.072]
const auto qasymm8_signed_bn_add_qinfo_set = framework::dataset::make("BnAddInfo", { QuantizationInfo(0.4, -120) });  // Representable Range: [-0.32, 9.08]

// Representable Range: [-21.36, 39.84], Expected F32 range: [-27.4, 51.1], leaving some space for saturation
const auto qasymm8_signed_add_output_qinfo_set = framework::dataset::make("AddOutputInfo", { QuantizationInfo(0.24, -39) });

// Representable Range: [-4.8, 10.5], Expected FP32 range: [-9.6713, 14.0942], leaving some space for saturation
// This range also makes sense with the activation boundaries above, i.e. [-2, 8] for LU_BOUNDED_RELU and [0, 6] for BOUNDED_RELU
const auto qasymm8_signed_final_output_qinfo_set = framework::dataset::make("FinalOutputInfo", { QuantizationInfo(0.06, -48) });

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(AddMulAdd)

template <typename T>
using NEAddMulAddFloatFixture = AddMulAddFloatValidationFixture<Tensor, Accessor, NEAddMulAdd, T, true>;

template <typename T>
using NEAddMulAddFloatFixtureWoIntermOut = AddMulAddFloatValidationFixture<Tensor, Accessor, NEAddMulAdd, T, false>;

TEST_SUITE(Float)

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEAddMulAddFloatFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                            ActivationFunctionsDataset))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference); // Arithmetic Addition has more strict tolerance
    validate(Accessor(_target), _reference, tolerance_fp32);
}

// This test is to stress the case when there is no intermediate output required (i.e. nullptr)
FIXTURE_DATA_TEST_CASE(RunSmallWithoutIntermOutput, NEAddMulAddFloatFixtureWoIntermOut<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("ActivationInfo", { ActivationLayerInfo() })))
{
    // Validate outputs
    validate(Accessor(_target), _reference, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEAddMulAddFloatFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)),
                                                                                                          ActivationFunctionsDataset))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference); // Arithmetic Addition has more strict tolerance
    validate(Accessor(_target), _reference, tolerance_fp32);
}

TEST_SUITE_END() // F32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEAddMulAddFloatFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                           ActivationFunctionsDataset))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference); // Arithmetic Addition has more strict tolerance
    validate(Accessor(_target), _reference, tolerance_fp16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEAddMulAddFloatFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(),
                                                                                                                 framework::dataset::make("DataType", DataType::F16)),
                                                                                                         ActivationFunctionsDataset))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference); // Arithmetic Addition has more strict tolerance
    validate(Accessor(_target), _reference, tolerance_fp16);
}
TEST_SUITE_END() // F16
#endif           // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE_END() // Float

template <typename T>
using NEAddMulQuantizedFixture = AddMulAddQuantizedValidationFixture<Tensor, Accessor, NEAddMulAdd, T, true>;

template <typename T>
using NEAddMulAddQuantizedFixtureWoIntermOut = AddMulAddQuantizedValidationFixture<Tensor, Accessor, NEAddMulAdd, T, false>;

TEST_SUITE(Quantized)

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEAddMulQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                       ActivationFunctionsDataset),
                                                                                                                       qasymm8_input1_qinfo_set),
                                                                                                                       qasymm8_input2_qinfo_set),
                                                                                                                       qasymm8_bn_mul_qinfo_set),
                                                                                                                       qasymm8_bn_add_qinfo_set),
                                                                                                                       qasymm8_add_output_qinfo_set),
                                                                                                               qasymm8_final_output_qinfo_set))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference, tolerance_quant);
    validate(Accessor(_target), _reference, tolerance_quant);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEAddMulQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                     framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                     ActivationFunctionsDataset),
                                                                                                                     qasymm8_input1_qinfo_set),
                                                                                                                     qasymm8_input2_qinfo_set),
                                                                                                                     qasymm8_bn_mul_qinfo_set),
                                                                                                                     qasymm8_bn_add_qinfo_set),
                                                                                                                     qasymm8_add_output_qinfo_set),
                                                                                                             qasymm8_final_output_qinfo_set))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference, tolerance_quant);
    validate(Accessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEAddMulQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                      ActivationFunctionsDataset),
                                                                                                                      qasymm8_signed_input1_qinfo_set),
                                                                                                                      qasymm8_signed_input2_qinfo_set),
                                                                                                                      qasymm8_signed_bn_mul_qinfo_set),
                                                                                                                      qasymm8_signed_bn_add_qinfo_set),
                                                                                                                      qasymm8_signed_add_output_qinfo_set),
                                                                                                              qasymm8_signed_final_output_qinfo_set))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference, tolerance_quant);
    validate(Accessor(_target), _reference, tolerance_quant);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEAddMulQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                    ActivationFunctionsDataset),
                                                                                                                    qasymm8_signed_input1_qinfo_set),
                                                                                                                    qasymm8_signed_input2_qinfo_set),
                                                                                                                    qasymm8_signed_bn_mul_qinfo_set),
                                                                                                                    qasymm8_signed_bn_add_qinfo_set),
                                                                                                                    qasymm8_signed_add_output_qinfo_set),
                                                                                                            qasymm8_signed_final_output_qinfo_set))
{
    // Validate outputs
    validate(Accessor(_interm_target), _interm_reference, tolerance_quant);
    validate(Accessor(_target), _reference, tolerance_quant);
}
TEST_SUITE_END() // QASYMM8_SIGNED

TEST_SUITE_END() // Quantized

TEST_SUITE_END() // AddMulAdd
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // __aarch64__
