/*
 * Copyright (c) 2017-2020, 2022-2023 Arm Limited.
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
#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/cpu/kernels/CpuSoftmaxKernel.h"
#include "tests/NEON/Accessor.h"
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
using framework::dataset::make;
namespace
{
/** Tolerance for float operations */
constexpr AbsoluteTolerance<float> tolerance_f32(0.000001f);
RelativeTolerance<half>            tolerance_f16(half(0.2));

/** Tolerance for quantized operations */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_signed(1);

/** CNN data types */
const auto CNNDataTypes = make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(SoftmaxLayer)
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching data types
                        TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching shapes
                        TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8, // Invalid output quantization info
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,  //Invalid axis high
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,  //Invalid axis low
                                    QuantizationInfo(1.f/256, 12)),
                        }),
    make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U), 1, DataType::F16),
                        TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                        TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 0)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 0)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 0)),
                        }),
    make("beta", { 1.0,
                   2.0,
                   1.0,
                   2.0,
                   1.0,
                   1.0,
                   2.0,
                   1.0,
                }),
    make("axis", { 0,
                   0,
                   0,
                   1,
                   0,
                   -1,
                   2,
                   -3,
                }),
    make("Expected", { false, false, false, true, true, true, false, false })),
    input_info, output_info, beta, axis, expected)
{
    ARM_COMPUTE_EXPECT(bool(NESoftmaxLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), beta, axis)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NESoftmaxLayerFixture = SoftmaxValidationFixture<Tensor, Accessor, NESoftmaxLayer, T>;

DATA_TEST_CASE(KernelSelection, framework::DatasetMode::ALL,
    concat(concat(
        combine(
            make("CpuExt", std::string("NEON")),
            make("DataType", { DataType::F32,
                            DataType::F16,
                            DataType::QASYMM8,
                            DataType::QASYMM8_SIGNED})
        ),
        combine(
            make("CpuExt", std::string("SVE")),
            make("DataType", { DataType::F32,
                            DataType::F16}))
        ),
        combine(
            make("CpuExt", std::string("SVE2")),
            make("DataType", { DataType::QASYMM8,
                            DataType::QASYMM8_SIGNED}))
        ),
        cpu_ext, data_type)
{
    using namespace cpu::kernels;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.sve  = (cpu_ext == "SVE");
    cpu_isa.sve2 = (cpu_ext == "SVE2");
    cpu_isa.fp16 = (data_type == DataType::F16);

    const auto *selected_impl = CpuSoftmaxKernel::get_implementation(
        SoftmaxKernelDataTypeISASelectorData{ data_type, cpu_isa, false /* is_log */ }, cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl);

    std::string expected = "neon_" + cpu_impl_dt(data_type) + "_softmax";
    std::string actual   = selected_impl->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NESoftmaxLayerFixture<half>, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::Small4DShapes(),
        make("DataType", DataType::F16),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, 1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NESoftmaxLayerFixture<half>, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::Small4DShapes(),
        make("DataType", DataType::F16),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, 2, -1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY,
    combine(
        datasets::SoftmaxLayerLargeShapes(),
        make("DataType", DataType::F16),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() //FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall2D, NESoftmaxLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::F32),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NESoftmaxLayerFixture<float>, framework::DatasetMode::PRECOMMIT,
    combine(datasets::Small4DShapes(),
        make("DataType", DataType::F32),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -2, 3 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY,
    combine(datasets::SoftmaxLayerLargeShapes(),
        make("DataType", DataType::F32),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0 })))
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
FIXTURE_DATA_TEST_CASE(RunSmall2D, NESoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::QASYMM8),
        combine(
            make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
            make("Beta", { 1.0f, 2.f })
        ),
        make("Axis", { 0, -1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NESoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
    combine(
        datasets::Small4DShapes(),
        make("DataType", DataType::QASYMM8),
        combine(
            make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
            make("Beta", { 1.0f, 2.f })),
        make("Axis", { 0, 1, -2 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NESoftmaxLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
    combine(
        datasets::SoftmaxLayerLargeShapes(),
        make("DataType", DataType::QASYMM8),
        combine(
            make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
            make("Beta", { 1.0f, 2.0f })
        ),
        make("Axis", { 0 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() //QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall2D, NESoftmaxLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::QASYMM8_SIGNED),
        combine(
            make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
            make("Beta", { 1.0f, 2.f })
        ),
        make("Axis", { 0, -1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
FIXTURE_DATA_TEST_CASE(RunSmall4D, NESoftmaxLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
    combine(
        datasets::Small4DShapes(),
        make("DataType", DataType::QASYMM8_SIGNED),
        combine(
            make("QuantizationInfo", { QuantizationInfo(0.5f, -10) }),
            make("Beta", { 1.0f, 2.f })
        ),
        make("Axis", { 0, 1, -1 })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}
TEST_SUITE_END() //QASYMM8_SIGNED

TEST_SUITE_END() //Quantized

TEST_SUITE_END() //SoftmaxLayer
TEST_SUITE_END() //NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
