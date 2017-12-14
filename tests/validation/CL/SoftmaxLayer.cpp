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
#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "tests/CL/CLAccessor.h"
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
RelativeTolerance<half>  tolerance_f16(half(0.2));
RelativeTolerance<float> tolerance_f32(0.001f);

/** Tolerance for fixed point operations */
constexpr AbsoluteTolerance<int16_t> tolerance_fixed_point(2);

/** Tolerance for quantized operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::QASYMM8,
    DataType::F16,
    DataType::F32,
    DataType::QS8,
    DataType::QS16,
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(SoftmaxLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(concat(datasets::SoftmaxLayerSmallShapes(), datasets::SoftmaxLayerLargeShapes()), CNNDataTypes), shape, data_type)
{
    // Set fixed point position and quantization info if is allowed
    const int              fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;
    const QuantizationInfo quantization_info    = is_data_type_quantized_asymmetric(data_type) ? QuantizationInfo(1.f / 255.f, 0) : QuantizationInfo();

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, data_type, 1, fixed_point_position, quantization_info);
    CLTensor dst = create_tensor<CLTensor>(shape, data_type, 1, fixed_point_position, QuantizationInfo(1.f / 256.f, 0));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLSoftmaxLayer smx_layer;
    smx_layer.configure(&src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Get reduction kernel info
    CLLogits1DMaxShiftExpSumKernel::ParallelReductionInfo reduction_info = CLLogits1DMaxShiftExpSumKernel::is_parallel_reduction(shape.x());

    // Validate src padding
    // Legacy path used only by quantized asymmetric data type
    if(is_data_type_quantized_asymmetric(data_type))
    {
        const PaddingSize padding_src = PaddingCalculator(shape.x(), 16).required_padding();
        validate(src.info()->padding(), padding_src);
    }
    else
    {
        const PaddingSize padding_src = PaddingCalculator(shape.x(), std::get<1>(reduction_info)).required_padding();
        validate(src.info()->padding(), padding_src);
    }

    // Validate dst padding
    const PaddingSize padding_dst = PaddingCalculator(shape.x(), 16).required_padding();
    validate(dst.info()->padding(), padding_dst);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching data types
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 2), // Mismatching fixed point
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8, // Invalid output quantization info
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Window shrink
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 12)),
                                                      }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 12)),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QS8, 3),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8,
                                                                  QuantizationInfo(1.f/256, 0)),
                                                     })),
               framework::dataset::make("Expected", { false, false, false, false, false, true, true, true })),
               input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLSoftmaxLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLSoftmaxLayerFixture = SoftmaxValidationFixture<CLTensor, CLAccessor, CLSoftmaxLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                           framework::dataset::make("DataType", DataType::F16)),
                                                                                                   framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                               framework::dataset::make("DataType", DataType::F16)),
                                                                                                       framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                            framework::dataset::make("DataType", DataType::F32)),
                                                                                                    framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                framework::dataset::make("DataType", DataType::F32)),
                                                                                                        framework::dataset::make("Beta", { 1.0f, 2.0f })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLSoftmaxLayerFixedPointFixture = SoftmaxValidationFixedPointFixture<CLTensor, CLAccessor, CLSoftmaxLayer, T>;

TEST_SUITE(FixedPoint)
TEST_SUITE(QS8)
// Testing for fixed point position [1,6) as reciprocal limits the maximum fixed point position to 5
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(datasets::SoftmaxLayerSmallShapes(), framework::dataset::make("DataType",
                                                                                                                       DataType::QS8)),
                                                                                                               framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fixed_point);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(), framework::dataset::make("DataType",
                                                                                                                   DataType::QS8)),
                                                                                                                   framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fixed_point);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14) as reciprocal limits the maximum fixed point position to 14
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::QS16)),
                                                                                                                framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fixed_point);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::QS16)),
                                                                                                                    framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fixed_point);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLSoftmaxLayerQuantizedFixture = SoftmaxValidationQuantizedFixture<CLTensor, CLAccessor, CLSoftmaxLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                               combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                       framework::dataset::make("Beta", { 1.0f, 2.f }))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLSoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                   combine(framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
                                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f }))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
