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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/FullyConnectedLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FullyConnectedLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
RelativeTolerance<float>            tolerance_f32(0.05f);
RelativeTolerance<half_float::half> tolerance_f16(half(0.2));
constexpr float                     tolerance_num = 0.07f; /**< Tolerance number */

/** Tolerance for fixed point operations */
constexpr AbsoluteTolerance<float> tolerance_fixed_point(1.f);
/** Tolerance for quantized asymmetric operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
    DataType::QS8,
    DataType::QS16,
    DataType::QASYMM8,
});

const auto FullyConnectedParameters = combine(framework::dataset::make("TransposeWeights", { false, true }), framework::dataset::make("ReshapeWeights", { false, true }));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(FullyConnectedLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallFullyConnectedLayerDataset(), datasets::LargeFullyConnectedLayerDataset()),
                                                                           FullyConnectedParameters),
                                                                   CNNDataTypes),
               src_shape, weights_shape, bias_shape, dst_shape, transpose_weights, reshape_weights, data_type)
{
    // Set fixed point position data type allowed
    const int              fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;
    const DataType         bias_data_type       = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;
    const QuantizationInfo quantization_info    = is_data_type_quantized_asymmetric(data_type) ? QuantizationInfo(2.f / 255.f, 127) : QuantizationInfo();

    TensorShape ws(weights_shape);

    // Transpose weights if not done in the function
    if(!reshape_weights || !transpose_weights)
    {
        const size_t shape_x = ws.x();
        ws.set(0, ws.y());
        ws.set(1, shape_x);
    }

    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(src_shape, data_type, 1, fixed_point_position, quantization_info);
    CLTensor weights = create_tensor<CLTensor>(ws, data_type, 1, fixed_point_position, quantization_info);
    CLTensor bias    = create_tensor<CLTensor>(bias_shape, bias_data_type, 1, fixed_point_position, quantization_info);
    CLTensor dst     = create_tensor<CLTensor>(dst_shape, data_type, 1, fixed_point_position, quantization_info);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    const QuantizationInfo src_quantization_info     = src.info()->quantization_info();
    const QuantizationInfo weights_quantization_info = weights.info()->quantization_info();

    // Create and configure function.
    CLFullyConnectedLayer fc;
    fc.configure(&src, &weights, &bias, &dst, transpose_weights, !reshape_weights);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(dst_shape);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate QuantizationInfo
    ARM_COMPUTE_EXPECT(src.info()->quantization_info() == src_quantization_info, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->quantization_info() == weights_quantization_info, framework::LogLevel::ERRORS);
}

template <typename T>
using CLFullyConnectedLayerFixture = FullyConnectedLayerValidationFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T, false>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                                                                                                                        FullyConnectedParameters),
                                                                                                                framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeFullyConnectedLayerDataset(),
                                                                                                                      FullyConnectedParameters),
                                                                                                              framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLFullyConnectedLayerFixedPointFixture = FullyConnectedLayerValidationFixedPointFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T, false>;

TEST_SUITE(FixedPoint)
TEST_SUITE(QS8)
// Testing for fixed point position [1,6) as reciprocal limits the maximum fixed point position to 5
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       FullyConnectedParameters),
                       framework::dataset::make("DataType",
                                                DataType::QS8)),
                       framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fixed_point);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(),
                       FullyConnectedParameters),
                       framework::dataset::make("DataType",
                                                DataType::QS8)),
                       framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fixed_point);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14) as reciprocal limits the maximum fixed point position to 14
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       FullyConnectedParameters),
                       framework::dataset::make("DataType",
                                                DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fixed_point);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(),
                       FullyConnectedParameters),
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
using CLFullyConnectedLayerQuantizedFixture = FullyConnectedLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLFullyConnectedLayer, T, false>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(
                           combine(datasets::SmallFullyConnectedLayerDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255.f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFullyConnectedLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(
                           combine(datasets::LargeFullyConnectedLayerDataset(),
                                   FullyConnectedParameters),
                           framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 256.f, 10) })))
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
