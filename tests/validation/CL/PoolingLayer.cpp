/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/PoolingLayerDataset.h"
#include "tests/datasets/PoolingTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PoolingLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;
namespace
{
/** Input data sets for floating-point data types */
const auto PoolingLayerDatasetFP = combine(datasets::PoolingTypes(), make("PoolingSize", { Size2D(2, 2), Size2D(3, 3), Size2D(5, 7) }),
                                                   make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) }),
                                           make("ExcludePadding", { true, false }));

const auto PoolingLayerDatasetFPSmall = combine(datasets::PoolingTypes(), make("PoolingSize", { Size2D(2, 2), Size2D(3, 3) }),
                                                        make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0) }),
                                                make("ExcludePadding", { true, false }));

/** Input data sets for asymmetric data type */
const auto PoolingLayerDatasetQASYMM8 = combine(concat(combine(make("PoolingType",
{
    PoolingType::MAX, PoolingType::AVG,
}),
make("PoolingSize", { Size2D(2, 2), Size2D(3, 3) }),
make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) })),
combine(make("PoolingType", { PoolingType::AVG }), make("PoolingSize", { Size2D(5, 7) }), make("PadStride", { PadStrideInfo(2, 1, 0, 0) }))),
make("ExcludePadding", { true }));

const auto PoolingLayerDatasetQASYMM8Small = combine(make("PoolingType",
{
    PoolingType::MAX, PoolingType::AVG,
}),
make("PoolingSize", { Size2D(2, 2), Size2D(5, 7) }),
make("PadStride", { PadStrideInfo(1, 2, 1, 1) }),
make("ExcludePadding", { true }));

const auto PoolingLayerDatasetFPIndicesSmall = combine(make("PoolingType",
{ PoolingType::MAX }),
make("PoolingSize", { Size2D(2, 2) }),
make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 0, 0) }),
make("ExcludePadding", { true, false }));

constexpr AbsoluteTolerance<float>   tolerance_f32(0.001f);  /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float>   tolerance_f16(0.01f);   /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);   /**< Tolerance value for comparing reference's output against implementation's output for 8-bit asymmetric type */
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_s(1); /**< Tolerance value for comparing reference's output against implementation's output for 8-bit signed asymmetric type */
const auto                           pool_data_layout_dataset = make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC });

const auto pool_fp_mixed_precision_dataset = make("FpMixedPrecision", { true, false });

void RoundToNearestIntegerPoolTestBody(const DataLayout layout, const TensorShape &shape,
    const TensorShape &output_shape)
{
    const auto pool_info = PoolingLayerInfo(PoolingType::AVG,
        Size2D(3,1), layout, PadStrideInfo(), true /* exclude padding */);

    const auto dtype = DataType::QASYMM8_SIGNED;
    const auto qinfo = QuantizationInfo(1.f, 0);

    CLTensor input = create_tensor<CLTensor>(shape, dtype, 1, qinfo, layout);
    CLTensor output = create_tensor<CLTensor>(output_shape, dtype, 1, qinfo, layout);

    CLPoolingLayer pool;
    pool.configure(&input, &output, pool_info);

    input.allocator()->allocate();
    output.allocator()->allocate();

    std::vector<int8_t> values = {-10, -10, -9};
    std::vector<int8_t> refs = {-10};

    ARM_COMPUTE_EXPECT(values.size() == shape.total_size(), framework::LogLevel::ERRORS);

    library->fill_static_values(CLAccessor(input), values);

    pool.run();

    output.map(true);
    for(unsigned int i = 0; i < refs.size(); ++i)
    {
        const int8_t ref = refs[i];
        const int8_t target = reinterpret_cast<int8_t *>(output.buffer())[i];

        ARM_COMPUTE_EXPECT(ref == target, framework::LogLevel::ERRORS);
    }

    output.unmap();
}

} // namespace

TEST_SUITE(CL)
TEST_SUITE(PoolingLayer)

TEST_CASE(RoundToNearestIntegerNHWC, framework::DatasetMode::ALL)
{
    const auto layout = DataLayout::NHWC;
    const auto shape = TensorShape(1U,3U,1U);
    const auto output_shape = TensorShape(1U,1U,1U);

    RoundToNearestIntegerPoolTestBody(layout, shape, output_shape);
}

TEST_CASE(RoundToNearestIntegerNCHW, framework::DatasetMode::ALL)
{
    const auto layout = DataLayout::NCHW;
    const auto shape = TensorShape(3U,1U,1U);
    const auto output_shape = TensorShape(1U,1U,1U);

    RoundToNearestIntegerPoolTestBody(layout, shape, output_shape);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data type
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Invalid pad/size combination
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Invalid pad/size combination
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8), // Invalid parameters
                                                       TensorInfo(TensorShape(15U, 13U, 5U), 1, DataType::F32),     // Non-rectangular Global Pooling
                                                       TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32),     // Invalid output Global Pooling
                                                       TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(1U, 16U, 1U),  1, DataType::F32),
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(25U, 16U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(2U, 2U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(12U, 12U, 5U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(1U, 15U, 1U), 1, DataType::F32),
                                                     }),
               make("PoolInfo",  { PoolingLayerInfo(PoolingType::AVG, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NCHW, PadStrideInfo(1, 1, 2, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 2)),
                                                       PoolingLayerInfo(PoolingType::L2, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, DataLayout::NCHW),
                                                       PoolingLayerInfo(PoolingType::MAX, DataLayout::NCHW),
                                                       PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NHWC, PadStrideInfo(), false),
                                                       PoolingLayerInfo(PoolingType::AVG, DataLayout::NCHW),
                                                       PoolingLayerInfo(PoolingType::MAX, 2, DataLayout::NHWC, PadStrideInfo(1, 1, 0, 0), false),
                                                      }),
               make("Expected", { false, false, false, false, true, false, true, true , false})),
               input_info, output_info, pool_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPoolingLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info)) == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

template <typename T>
using CLPoolingLayerFixture = PoolingLayerValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;
template <typename T>
using CLPoolingLayerMixedDataLayoutFixture = PoolingLayerValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T, true>;

template <typename T>
using CLSpecialPoolingLayerFixture = SpecialPoolingLayerValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

template <typename T>
using CLMixedPrecesionPoolingLayerFixture = PoolingLayerValidationMixedPrecisionFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

template <typename T>
using CLPoolingLayerIndicesFixture = PoolingLayerIndicesValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSpecial, CLSpecialPoolingLayerFixture<float>, framework::DatasetMode::ALL, datasets::PoolingLayerDatasetSpecial() * make("DataType", DataType::F32))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(), PoolingLayerDatasetFPSmall,
                                                                                                                  make("DataType",
                                                                                                                          DataType::F32),
                                                                                                          pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLPoolingLayerMixedDataLayoutFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                       datasets::PoolingTypes(),
                                                       make("PoolingSize", { Size2D(2, 2) }),
                                               make("PadStride", { PadStrideInfo(2, 1, 0, 0) }),
                                       make("ExcludePadding", { false }),
                               make("DataType", DataType::F32),
                       pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), PoolingLayerDatasetFP,
                                                                                                                make("DataType",
                                                                                                                        DataType::F32),
                                                                                                        pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallIndices, CLPoolingLayerIndicesFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                                                                                                                        PoolingLayerDatasetFPIndicesSmall,
                                                                                                                                make("DataType",
                                                                                                                                        DataType::F32),
                                                                                                                        pool_data_layout_dataset,make("UseKernelIndices", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
    validate(CLAccessor(_target_indices), _ref_indices);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<float>, framework::DatasetMode::ALL,
                       combine(make("InputShape", { TensorShape(27U, 13U, 2U),
                                                                             TensorShape(27U, 13U, 2U, 4U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size2D(27, 13) }),
                                    make("PadStride", PadStrideInfo(1, 1, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F32),
                                    make("DataLayout", DataLayout::NHWC)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(make("InputShape", { TensorShape(79U, 37U, 11U),
                                                                             TensorShape(79U, 37U, 11U, 4U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size2D(79, 37) }),
                                    make("PadStride", PadStrideInfo(1, 1, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F32),
                                    make("DataLayout", DataLayout::NHWC)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // GlobalPooling

TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMixedPrecesionPoolingLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                                                                                                                       PoolingLayerDatasetFPSmall,
                                                                                                                               make("DataType", DataType::F16),
                                                                                                                       pool_data_layout_dataset,
                                                                                                                       pool_fp_mixed_precision_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLMixedPrecesionPoolingLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), PoolingLayerDatasetFP,
                                                                                                                     make("DataType", DataType::F16),
                                                                                                                     pool_data_layout_dataset,
                                                                                                                     pool_fp_mixed_precision_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmallIndices, CLPoolingLayerIndicesFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                                                                                                                       PoolingLayerDatasetFPIndicesSmall,
                                                                                                                               make("DataType",
                                                                                                                                       DataType::F16),
                                                                                                                       pool_data_layout_dataset, make("UseKernelIndices", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
    validate(CLAccessor(_target_indices), _ref_indices);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<half>, framework::DatasetMode::ALL,
                       combine(make("InputShape", { TensorShape(27U, 13U, 2U),
                                                                             TensorShape(27U, 13U, 2U, 4U)
                                                                            }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size2D(27, 13) }),
                                    make("PadStride", PadStrideInfo(1, 1, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F16),
                                    make("DataLayout", DataLayout::NHWC)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(make("InputShape", { TensorShape(79U, 37U, 11U),
                                                                             TensorShape(79U, 37U, 11U, 4U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size2D(79, 37) }),
                                    make("PadStride", PadStrideInfo(1, 1, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F16),
                                    make("DataLayout", DataLayout::NHWC)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // GlobalPooling

TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)

template <typename T>
using CLPoolingLayerQuantizedFixture = PoolingLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;
template <typename T>
using CLPoolingLayerQuantizedMixedDataLayoutFixture = PoolingLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLPoolingLayer, T, true>;

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                                                                                                                     PoolingLayerDatasetQASYMM8Small,
                                                                                                                             make("DataType", DataType::QASYMM8),
                                                                                                                     pool_data_layout_dataset,
                                                                                                                     make("InputQuantInfo", { QuantizationInfo(1.f / 255.f, 10), QuantizationInfo(1.f / 255.f, 10) }),
                                                                                                                     make("OutputQuantInfo", { QuantizationInfo(1.f / 255.f, 5), QuantizationInfo(1.f / 255.f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLPoolingLayerQuantizedMixedDataLayoutFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                       make("PoolingType", { PoolingType::MAX, PoolingType::AVG }),
                                                       make("PoolingSize", { Size2D(2, 2) }),
                                               make("PadStride", { PadStrideInfo(1, 2, 1, 1) }),
                                       make("ExcludePadding", { true }),
                               make("DataType", DataType::QASYMM8),
                       make("DataLayout", { DataLayout::NHWC, DataLayout::NCHW }),
                       make("InputQuantInfo", { QuantizationInfo(1.f / 255.f, 10) }),
                       make("OutputQuantInfo", { QuantizationInfo(1.f / 255.f, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                                                                                                                    PoolingLayerDatasetQASYMM8Small,
                                                                                                                            make("DataType", DataType::QASYMM8_SIGNED),
                                                                                                                    pool_data_layout_dataset,
                                                                                                                    make("InputQuantInfo", { QuantizationInfo(1.f / 127.f, -10), QuantizationInfo(1.f / 127.f, -10) }),
                                                                                                                    make("OutputQuantInfo", { QuantizationInfo(1.f / 127.f, -5), QuantizationInfo(1.f / 127.f, -10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_s);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLPoolingLayerQuantizedMixedDataLayoutFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallNoneUnitShapes(),
                       make("PoolingType", { PoolingType::MAX, PoolingType::AVG }),
                                                       make("PoolingSize", { Size2D(2, 2) }),
                                               make("PadStride", { PadStrideInfo(1, 2, 1, 1) }),
                                       make("ExcludePadding", { true }),
                               make("DataType", DataType::QASYMM8_SIGNED),
                       make("DataLayout", { DataLayout::NHWC, DataLayout::NCHW }),
                       make("InputQuantInfo", { QuantizationInfo(1.f / 127.f, -10) }),
                       make("OutputQuantInfo", { QuantizationInfo(1.f / 127.f, -10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_s);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // PoolingLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
