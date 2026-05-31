/*
 * Copyright (c) 2017-2021, 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/NormalizationTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/NormalizationLayerFixture.h"
#include "tests/validation/Validation.h"

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
#ifdef ARM_COMPUTE_ENABLE_FP16
constexpr AbsoluteTolerance<float> tolerance_f16(0.1f);
#endif /* ARM_COMPUTE_ENABLE_FP16 */
constexpr AbsoluteTolerance<float> tolerance_f32(0.00001f);

/** Input data set. */
const auto NormalizationDataset     = combine(datasets::SmallShapes(),
                                              datasets::NormalizationTypes(),
                                              make("NormalizationSize", 3, 9, 2),
                                              make("Beta", {0.5f, 1.f, 2.f}),
                                              make("IsScaled", {true}));
const auto NormalizationDatasetFP32 = combine(datasets::NormalizationTypes(),
                                              make("NormalizationSize", 3, 9, 2),
                                              make("Beta", {0.5f, 1.f, 2.f}),
                                              make("IsScaled", {true, false}));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(NormalizationLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching data type input/output
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching shapes
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Even normalization
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Non implemented IN_MAP_2D
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                          }),
    make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                          }),
    make("NormInfo",  { NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                            NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                            NormalizationLayerInfo(NormType::IN_MAP_1D, 4),
                                            NormalizationLayerInfo(NormType::IN_MAP_2D, 5),
                                            NormalizationLayerInfo(NormType::CROSS_MAP, 1),
                                           }),
    make("Expected", { false, false, false, true, true })
    ),
    input_info, output_info, norm_info, expected)
{
    bool is_valid = bool(NENormalizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), norm_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NENormalizationLayerFixture = NormalizationValidationFixture<Tensor, Accessor, NENormalizationLayer, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NENormalizationLayerFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(NormalizationDataset,
                               make("DataType", DataType::F16),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NENormalizationLayerFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallShapes(),
                               NormalizationDatasetFP32,
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NENormalizationLayerFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeShapes(),
                               NormalizationDatasetFP32,
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // NormalizationLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
