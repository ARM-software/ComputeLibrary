/*
 * Copyright (c) 2022, 2024-2025 Arm Limited.
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

#include "arm_compute/core/TensorShape.h"
#include "tests/framework/datasets/Datasets.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLPooling3dLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/Pooling3dLayerDataset.h"
#include "tests/datasets/PoolingTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/Pooling3dLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::ContainerDataset;
using framework::dataset::make;
namespace
{
/** Input data sets for floating-point data types */
const auto Pooling3dLayerDatasetFP = combine(datasets::PoolingTypes(), make("PoolingSize", { Size3D(2, 3, 2) }),
                                                             make("Stride", { Size3D(1, 1, 1), Size3D(2, 1, 1), Size3D(1, 2, 1), Size3D(2, 2, 1) }),
                                                     make("Padding", { Padding3D(0, 1, 0), Padding3D(1, 1, 1) }),
                                             make("ExcludePadding", { true, false }));

const auto Pooling3dLayerDatasetFPSmall = combine(datasets::PoolingTypes(), make("PoolingSize", { Size3D(2, 2, 2), Size3D(3, 3, 3) }),
                                                                  make("Stride", { Size3D(2, 2, 2), Size3D(2, 1, 1) }),
                                                          make("Padding", { Padding3D(0, 0, 0), Padding3D(1, 1, 1), Padding3D(1, 0, 0) }),
                                                  make("ExcludePadding", { true, false }));

const auto Pooling3DLayerDatasetQuantized = combine(make("PoolingType", { PoolingType::MAX, PoolingType::AVG }),
                                                                            make("PoolingSize", { Size3D(2, 3, 2) }),
                                                                    make("Stride", { Size3D(1, 1, 1), Size3D(2, 1, 1), Size3D(1, 2, 1), Size3D(1, 1, 2), Size3D(2, 2, 1)}),
                                                            make("Padding", { Padding3D(0, 0, 0), Padding3D(1, 1, 1), Padding3D(1, 0, 0) }),
                                                    make("ExcludePadding", { true }));

using ShapeDataset = ContainerDataset<std::vector<TensorShape>>;

constexpr AbsoluteTolerance<float>   tolerance_f32(0.001f);       /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float>   tolerance_f16(0.1f);         /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_signed(1); /**< Tolerance value for comparing reference's output against implementation's output for QASYMM8_SIGNED integer datatype*/
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);        /**< Tolerance value for comparing reference's output against implementation's output for 8-bit asymmetric type */

} // namespace

TEST_SUITE(CL)
TEST_SUITE(Pooling3dLayer)

TEST_CASE(RoundToNearestInteger, framework::DatasetMode::ALL)
{
    const auto pool_info = Pooling3dLayerInfo(PoolingType::AVG,
        Size3D(3,1,1), Size3D(1,1,1), Padding3D(), true /* exclude padding */);

    const auto shape = TensorShape(1U,3U,1U,1U);
    const auto output_shape = TensorShape(1U,1U,1U,1U);

    const auto dtype = DataType::QASYMM8_SIGNED;
    const auto layout = DataLayout::NDHWC;
    const auto qinfo = QuantizationInfo(1.f, 0);

    CLTensor input = create_tensor<CLTensor>(shape, dtype, 1, qinfo, layout);
    CLTensor output = create_tensor<CLTensor>(output_shape, dtype, 1, qinfo, layout);

    CLPooling3dLayer pool;
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

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(make("InputInfo", { TensorInfo(TensorShape(2U, 27U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Mismatching data type
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC), // Invalid pad/size combination
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC), // Invalid pad/size combination
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Invalid output shape
                                                       TensorInfo(TensorShape(5U, 13U, 15U, 2U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Global Pooling
                                                       TensorInfo(TensorShape(13U,13U, 5U, 1U, 2U), 1, DataType::F32, DataLayout::NDHWC),  // Invalid output Global Pooling
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 4U), 1, DataType::F32, DataLayout::NDHWC), // Invalid data type
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 5U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(1U, 16U,  1U, 3U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(2U, 25U, 11U, 3U, 3U), 1, DataType::F16, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(2U, 30U, 11U, 3U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(2U, 25U, 16U, 3U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 3U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U,  1U,  1U, 1U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Global pooling applied
                                                       TensorInfo(TensorShape(5U,  2U,  2U, 2U, 2U), 1, DataType::F32, DataLayout::NDHWC), // Invalid output Global Pooling
                                                       TensorInfo(TensorShape(5U, 12U, 12U, 3U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 12U, 12U, 3U, 4U), 1, DataType::QASYMM8, DataLayout::NDHWC), // Invalid data type
                                                       TensorInfo(TensorShape(5U,  1U, 1U, 1U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(1U, 15U, 1U, 2U, 4U), 1, DataType::F32, DataLayout::NDHWC), // Output width larger than input
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 3U),  1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 2U),  1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 3U),  1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 3U),  1, DataType::F32, DataLayout::NDHWC),
                                                     }),
               make("PoolInfo",  { Pooling3dLayerInfo(PoolingType::AVG, 3, Size3D(1, 1, 1), Padding3D(0, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(1, 1, 1), Padding3D(2, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(1, 1, 1), Padding3D(0, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::L2,  3, Size3D(1, 1, 1), Padding3D(0, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::AVG),
                                                       Pooling3dLayerInfo(PoolingType::MAX),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(), Padding3D(), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(1U, 1U, 1U), Padding3D(), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG),
                                                       Pooling3dLayerInfo(PoolingType::MAX, 2, Size3D(1, 1, 2), Padding3D(0, 0, 0), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(2U, 2U, 2U), Padding3D(), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 1, Size3D(2U, 2U, 2U), Padding3D(2, 2, 2), true), // Pool size is smaller than the padding size with padding excluded
                                                       Pooling3dLayerInfo(PoolingType::AVG, 1, Size3D(2U, 2U, 2U), Padding3D(2, 2, 2), false), // Pool size is smaller than the padding size with padding included
                                                       Pooling3dLayerInfo(PoolingType::AVG, 3, Size3D(2U, 2U, 2U), Padding3D(2,1,2,2,1,2), false, false, DimensionRoundingType::CEIL), // CEIL with asymmetric Padding
                                                      }),
               make("Expected", { false, false, false, false, true, false, false, false, true , false, true, false, false, false})),
               input_info, output_info, pool_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPooling3dLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info)) == expected, framework::LogLevel::ERRORS);
}


template <typename T>
using CLPooling3dLayerFixture = Pooling3dLayerValidationFixture<CLTensor, CLAccessor, CLPooling3dLayer, T>;

template <typename T>
using CLSpecialPooling3dLayerFixture = SpecialPooling3dLayerValidationFixture<CLTensor, CLAccessor, CLPooling3dLayer, T>;

template <typename T>
using CLPooling3dLayerGlobalFixture = Pooling3dLayerGlobalValidationFixture<CLTensor, CLAccessor, CLPooling3dLayer, T>;

template <typename T>
using CLPooling3dLayerQuantizedFixture = Pooling3dLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLPooling3dLayer, T>;

// clang-format on
// *INDENT-ON*
TEST_SUITE(QUANTIZED)

TEST_SUITE(QASYMM8)
// Small Dataset Quantized Dataset
FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5dShapes(),
                                                                                                                       Pooling3DLayerDatasetQuantized,
                                                                                                                               make("DataType", DataType::QASYMM8),
                                                                                                                       make("InputQuantInfo", { QuantizationInfo(1.f / 127.f, 10), QuantizationInfo(1.f / 127.f, 10) }),
                                                                                                                       make("OutputQuantInfo", { QuantizationInfo(1.f / 127.f, 5), QuantizationInfo(1.f / 127.f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

// Large Dataset Quantized Dataset
FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large5dShapes(),
                                                                                                                       Pooling3DLayerDatasetQuantized,
                                                                                                                               make("DataType", DataType::QASYMM8),
                                                                                                                       make("InputQuantInfo", { QuantizationInfo(1.f / 127.f, 10), QuantizationInfo(1.f / 127.f, 10) }),
                                                                                                                       make("OutputQuantInfo", { QuantizationInfo(1.f / 127.f, 5), QuantizationInfo(1.f / 127.f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()

TEST_SUITE(QASYMM8_SIGNED)

// Large Dataset Quantized Dataset Signed
FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5dShapes(),
                                                                                                                      Pooling3DLayerDatasetQuantized,
                                                                                                                              make("DataType", DataType::QASYMM8_SIGNED),
                                                                                                                      make("InputQuantInfo", { QuantizationInfo(1.f / 127.f, -10), QuantizationInfo(1.f / 127.f, -10) }),
                                                                                                                      make("OutputQuantInfo", { QuantizationInfo(1.f / 127.f, -5), QuantizationInfo(1.f / 127.f, -10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_signed);
}

// Large Dataset Quantized pooling test
FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerQuantizedFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large5dShapes(),
                                                                                                                    Pooling3DLayerDatasetQuantized,
                                                                                                                            make("DataType", DataType::QASYMM8_SIGNED),
                                                                                                                    make("InputQuantInfo", { QuantizationInfo(1.f / 127.f, -10), QuantizationInfo(1.f / 127.f, -10) }),
                                                                                                                    make("OutputQuantInfo", { QuantizationInfo(1.f / 127.f, -5), QuantizationInfo(1.f / 127.f, -10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8_signed);
}

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSpecial, CLSpecialPooling3dLayerFixture<float>, framework::DatasetMode::ALL, datasets::Pooling3dLayerDatasetSpecial() * make("DataType", DataType::F32))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5dShapes(), Pooling3dLayerDatasetFPSmall,
                                                                                                            make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::Large5dShapes(), Pooling3dLayerDatasetFP,
                                                                                                          make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<float>, framework::DatasetMode::ALL,
                       combine(make("InputShape", { TensorShape(3U, 27U, 13U, 4U),
                                                                             TensorShape(4U, 27U, 13U, 4U, 2U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size3D(27, 13, 4) }),
                                    make("Strides",  Size3D(1, 1, 1)),
                                    make("Paddings", Padding3D(0, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallGlobal, CLPooling3dLayerGlobalFixture<float>, framework::DatasetMode::ALL,
                       combine(make("InputShape", { TensorShape(27U, 13U, 4U, 3U),
                                                                             TensorShape(27U, 13U, 4U, 4U, 2U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(make("InputShape", { TensorShape(4U, 79U, 37U, 11U),
                                                                             TensorShape(4U, 79U, 37U, 11U, 2U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size3D(79, 37, 11) }),
                                    make("Strides",  Size3D(1, 1, 1)),
                                    make("Paddings", Padding3D(0, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5x5Shapes(), Pooling3dLayerDatasetFPSmall,
                                                                                                           make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::Large5dShapes(), Pooling3dLayerDatasetFP,
                                                                                                         make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<half>, framework::DatasetMode::ALL,
                       combine(make("InputShape", { TensorShape(3U, 27U, 13U, 4U),
                                                                             TensorShape(4U, 27U, 13U, 4U, 2U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size3D(27, 13, 4) }),
                                    make("Strides",  Size3D(1, 1, 1)),
                                    make("Paddings", Padding3D(0, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunSmallGlobal, CLPooling3dLayerGlobalFixture<half>, framework::DatasetMode::ALL,
                       combine(make("InputShape", { TensorShape(27U, 13U, 4U, 3U),
                                                                             TensorShape(27U, 13U, 4U, 4U, 2U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(make("InputShape", { TensorShape(4U, 79U, 37U, 11U),
                                                                             TensorShape(4U, 79U, 37U, 11U, 2U)
                                                                           }),
                                    make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX }),
                                    make("PoolingSize", { Size3D(79, 37, 11) }),
                                    make("Strides",  Size3D(1, 1, 1)),
                                    make("Paddings", Padding3D(0, 0, 0)),
                                    make("ExcludePadding", false),
                                    make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // Pooling3dLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
