/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPReluLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ElementwiseOperationsFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_fp32(0.000001f);

/** Input data sets **/
const auto PReluLayerQASYMM8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::QASYMM8)),
                                              framework::dataset::make("DataType",
                                                                       DataType::QASYMM8));
const auto PReluLayerFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                           framework::dataset::make("DataType", DataType::F32));

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
RelativeTolerance<float> tolerance_fp16(0.001f);

const auto PReluLayerFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                           framework::dataset::make("DataType", DataType::F16));

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(PReluLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),      // Window shrink
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, true, false, false, false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEPReluLayer::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEPReluLayerFixture = PReluLayerValidationFixture<Tensor, Accessor, NEPReluLayer, T>;

template <typename T>
using NEPReluLayerQuantizedFixture = PReluLayerValidationQuantizedFixture<Tensor, Accessor, NEPReluLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, datasets::SmallShapes(),
               shape)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, DataType::QASYMM8);
    Tensor ref_src2 = create_tensor<Tensor>(shape, DataType::QASYMM8);
    Tensor dst      = create_tensor<Tensor>(shape, DataType::QASYMM8);

    // Create and Configure function
    NEPReluLayer prelu;
    prelu.configure(&ref_src1, &ref_src2, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEPReluLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                     PReluLayerQASYMM8Dataset),
                                                                                                                     framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                                                                                                     framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                             framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255.f, 5) }))

                      )
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPReluLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeShapes(),
                                                                                                                 PReluLayerQASYMM8Dataset),
                                                                                                                 framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                                                                                                 framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                 framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255.f, 5) }))

                      )
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPReluLayerFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), PReluLayerFP16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16, 0.01);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPReluLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), PReluLayerFP16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16, 0.01);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, datasets::SmallShapes(),
               shape)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, DataType::F32);
    Tensor ref_src2 = create_tensor<Tensor>(shape, DataType::F32);
    Tensor dst      = create_tensor<Tensor>(shape, DataType::F32);

    // Create and Configure function
    NEPReluLayer prelu;
    prelu.configure(&ref_src1, &ref_src2, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEPReluLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), PReluLayerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPReluLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), PReluLayerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}

template <typename T>
using NEPReluLayerBroadcastFixture = PReluLayerBroadcastValidationFixture<Tensor, Accessor, NEPReluLayer, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NEPReluLayerBroadcastFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapesBroadcast(),
                                                                                                                    PReluLayerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast, NEPReluLayerBroadcastFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapesBroadcast(),
                                                                                                                        PReluLayerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // PReluLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
