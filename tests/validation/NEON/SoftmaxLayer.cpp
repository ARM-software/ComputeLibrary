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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SoftmaxLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
constexpr AbsoluteTolerance<float> tolerance_f32(0.000001f);
RelativeTolerance<half>            tolerance_f16(half(0.2));

/** Tolerance for quantized operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(SoftmaxLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(concat(datasets::Small2DShapes(), datasets::Medium2DShapes()), CNNDataTypes), shape, data_type)
{
    const QuantizationInfo quantization_info = is_data_type_quantized_asymmetric(data_type) ? QuantizationInfo(1.f / 255.f, 0) : QuantizationInfo();

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type, 1, quantization_info);
    Tensor dst = create_tensor<Tensor>(shape, data_type, 1, QuantizationInfo(1.f / 256.f, 0));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NESoftmaxLayer smx_layer;
    smx_layer.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // NESoftmaxLayer configures the paddings only in the 2D case
    if(shape.num_dimensions() <= 2)
    {
        // Validate padding
        const int         step    = 16 / data_size_from_type(data_type);
        const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
        validate(src.info()->padding(), padding);
        validate(dst.info()->padding(), PaddingSize());
    }
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8, // Invalid output quantization info
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Window shrink
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),// Invalid input dimensionality
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,  //Invalid axis value
                                                                  QuantizationInfo(1.f/256, 12)),
                                                      }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 0)),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 0)),
                                                     })),
               framework::dataset::make("beta", { 1.0,
                                                  2.0,
                                                  1.0,
                                                  2.0,
                                                  1.0,
                                                  2.0,
                                                  1.0,
                                                  2.0,
                                                  1.0,
                                                })),
               framework::dataset::make("axis", { 1,
                                                  1,
                                                  1,
                                                  1,
                                                  1,
                                                  1,
                                                  1,
                                                  0,
                                                })),
               framework::dataset::make("Expected", { false, false, false, false, false, true, true, false })),
               input_info, output_info, beta, axis, expected)
{
    ARM_COMPUTE_EXPECT(bool(NESoftmaxLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), beta, axis)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NESoftmaxLayerFixture = SoftmaxValidationFixture<Tensor, Accessor, NESoftmaxLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NESoftmaxLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                 framework::dataset::make("DataType", DataType::F16)),
                                                                                                                 framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                         framework::dataset::make("Axis", { 1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NESoftmaxLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                           framework::dataset::make("Axis", { 1, 2, 3 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::F16)),
                                                                                                               framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                       framework::dataset::make("Axis", { 1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() //FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall2D, NESoftmaxLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                            framework::dataset::make("Axis", { 1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NESoftmaxLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                            framework::dataset::make("Axis", { 1, 2, 3 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                                                                                framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                        framework::dataset::make("Axis", { 1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() //FP32
TEST_SUITE_END() //Float

template <typename T>
using NESoftmaxLayerQuantizedFixture = SoftmaxValidationQuantizedFixture<Tensor, Accessor, NESoftmaxLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall2D, NESoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                 framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                 combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                         framework::dataset::make("Beta", { 1.0f, 2.f }))),
                                                                                                                 framework::dataset::make("Axis", { 1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NESoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(datasets::Small4DShapes(),
                                                                                                                 framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                 combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                         framework::dataset::make("Beta", { 1.0f, 2.f }))),
                                                                                                                 framework::dataset::make("Axis", { 1, 2, 3 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                   combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f }))),
                                                                                                                   framework::dataset::make("Axis", { 1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() //QASYMM8
TEST_SUITE_END() //Quantized

TEST_SUITE_END() //SoftmaxLayer
TEST_SUITE_END() //NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
