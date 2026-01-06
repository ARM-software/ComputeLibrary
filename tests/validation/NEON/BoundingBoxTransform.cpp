/*
 * Copyright (c) 2019-2021, 2024-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEBoundingBoxTransform.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/BoundingBoxTransformFixture.h"
#include "tests/validation/Validation.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
RelativeTolerance<float> relative_tolerance_f32(0.01f);
AbsoluteTolerance<float> absolute_tolerance_f32(0.001f);
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<half>  relative_tolerance_f16(half(0.2));
AbsoluteTolerance<float> absolute_tolerance_f16(half(0.02f));
#endif // ARM_COMPUTE_ENABLE_FP16

constexpr AbsoluteTolerance<uint16_t> tolerance_qasymm16(1);

// *INDENT-OFF*
// clang-format off
const auto BboxInfoDataset = make("BboxInfo", { BoundingBoxTransformInfo(20U, 20U, 2U, true),
                                                                    BoundingBoxTransformInfo(128U, 128U, 4U, true),
                                                                    BoundingBoxTransformInfo(800U, 600U, 1U, false),
                                                                    BoundingBoxTransformInfo(800U, 600U, 2U, true, { { 1.0, 0.5, 1.5, 2.0 } }),
                                                                    BoundingBoxTransformInfo(800U, 600U, 4U, false, { { 1.0, 0.5, 1.5, 2.0 } }),
                                                                    BoundingBoxTransformInfo(800U, 600U, 4U, false, { { 1.0, 0.5, 1.5, 2.0 } }, true)
                                                                  });

const auto DeltaDataset = make("DeltasShape", { TensorShape(36U, 1U),
                                                                    TensorShape(36U, 2U),
                                                                    TensorShape(36U, 2U),
                                                                    TensorShape(40U, 1U),
                                                                    TensorShape(40U, 20U),
                                                                    TensorShape(40U, 100U),
                                                                    TensorShape(40U, 200U)
                                                                  });
// clang-format on
// *INDENT-ON*
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(BBoxTransform)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("BoxesInfo", { TensorInfo(TensorShape(4U, 128U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 128U), 1, DataType::F32), // Wrong number of box fields
                                                       TensorInfo(TensorShape(4U, 128U), 1, DataType::F16), // Wrong data type
                                                       TensorInfo(TensorShape(4U, 128U), 1, DataType::F32), // Wrong number of classes
                                                       TensorInfo(TensorShape(4U, 128U), 1, DataType::F32),  // Deltas and predicted boxes have different dimensions
                                                       TensorInfo(TensorShape(4U, 128U), 1, DataType::F32)}),
               // Scaling is zero
               make("PredBoxesInfo",{ TensorInfo(TensorShape(128U, 128U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(128U, 128U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(127U, 128U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(128U, 100U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(128U, 100U), 1, DataType::F32),
                                                          TensorInfo(TensorShape(128U, 128U), 1, DataType::F32)}),
               make("DeltasInfo", { TensorInfo(TensorShape(128U, 128U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(128U, 128U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(127U, 128U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(128U, 100U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(128U, 128U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(128U, 128U), 1, DataType::F32)}),
               make("BoundingBoxTransofmInfo", { BoundingBoxTransformInfo(800.f, 600.f, 1.f),
                                                                     BoundingBoxTransformInfo(800.f, 600.f, 1.f),
                                                                     BoundingBoxTransformInfo(800.f, 600.f, 1.f),
                                                                     BoundingBoxTransformInfo(800.f, 600.f, 1.f),
                                                                     BoundingBoxTransformInfo(800.f, 600.f, 0.f)}),
               make("Expected", { true, false, false, false, false, false})
               ),
               boxes_info, pred_boxes_info, deltas_info, bbox_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEBoundingBoxTransform::validate(&boxes_info.clone()->set_is_resizable(true), &pred_boxes_info.clone()->set_is_resizable(true), &deltas_info.clone()->set_is_resizable(true), bbox_info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEBoundingBoxTransformFixture = BoundingBoxTransformFixture<Tensor, Accessor, NEBoundingBoxTransform, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(BoundingBox,
                       NEBoundingBoxTransformFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(DeltaDataset, BboxInfoDataset, make("DataType", {DataType::F32})))
{
    // Validate output
    validate(Accessor(_target), _reference, relative_tolerance_f32, 0.f, absolute_tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(BoundingBox,
                       NEBoundingBoxTransformFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(DeltaDataset, BboxInfoDataset, make("DataType", {DataType::F16})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, relative_tolerance_f16, 0.03f, absolute_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM16)
template <typename T>
using NEBoundingBoxTransformQuantizedFixture =
    BoundingBoxTransformQuantizedFixture<Tensor, Accessor, NEBoundingBoxTransform, T>;

FIXTURE_DATA_TEST_CASE(BoundingBox,
                       NEBoundingBoxTransformQuantizedFixture<uint16_t>,
                       framework::DatasetMode::ALL,
                       combine(DeltaDataset,
                               BboxInfoDataset,
                               make("DataType", {DataType::QASYMM16}),
                               make("DeltasQuantInfo", {QuantizationInfo(0.125f, 0)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm16);
}
TEST_SUITE_END() // QASYMM16
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // BBoxTransform
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
