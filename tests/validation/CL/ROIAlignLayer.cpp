/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLROIAlignLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/datasets/ROIDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ROIAlignLayerFixture.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> relative_tolerance_f32(0.01f);
AbsoluteTolerance<float> absolute_tolerance_f32(0.001f);

RelativeTolerance<float> relative_tolerance_f16(0.01f);
AbsoluteTolerance<float> absolute_tolerance_f16(0.001f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(RoiAlign)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(250U, 128U, 3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(250U, 128U, 3U), 1, DataType::F32), // Mismatching data type input/rois
                                                       TensorInfo(TensorShape(250U, 128U, 3U), 1, DataType::F32), // Mismatching data type input/output
                                                       TensorInfo(TensorShape(250U, 128U, 2U), 1, DataType::F32), // Mismatching depth size input/output
                                                       TensorInfo(TensorShape(250U, 128U, 3U), 1, DataType::F32), // Mismatching number of rois and output batch size
                                                       TensorInfo(TensorShape(250U, 128U, 3U), 1, DataType::F32), // Invalid number of values per ROIS
                                                       TensorInfo(TensorShape(250U, 128U, 3U), 1, DataType::F32), // Mismatching height and width input/output

                                                     }),
               framework::dataset::make("RoisInfo", { TensorInfo(TensorShape(5, 4U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(5, 4U), 1, DataType::F16),
                                                      TensorInfo(TensorShape(5, 4U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(5, 4U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(5, 10U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(4, 4U), 1, DataType::F32),
                                                      TensorInfo(TensorShape(5, 4U), 1, DataType::F32),
                                                    })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(7U, 7U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(7U, 7U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(7U, 7U, 3U, 4U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(7U, 7U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(7U, 7U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(7U, 7U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(5U, 5U, 3U, 4U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("PoolInfo", { ROIPoolingLayerInfo(7U, 7U, 1./8),
                                                      ROIPoolingLayerInfo(7U, 7U, 1./8),
                                                      ROIPoolingLayerInfo(7U, 7U, 1./8),
                                                      ROIPoolingLayerInfo(7U, 7U, 1./8),
                                                      ROIPoolingLayerInfo(7U, 7U, 1./8),
                                                      ROIPoolingLayerInfo(7U, 7U, 1./8),
                                                      ROIPoolingLayerInfo(7U, 7U, 1./8),
                                                      })),
               framework::dataset::make("Expected", { true, false, false, false, false, false, false })),
               input_info, rois_info, output_info, pool_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLROIAlignLayer::validate(&input_info.clone()->set_is_resizable(true), &rois_info.clone()->set_is_resizable(true), &output_info.clone()->set_is_resizable(true), pool_info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLROIAlignLayerFixture = ROIAlignLayerFixture<CLTensor, CLAccessor, CLROIAlignLayer, T>;

TEST_SUITE(Float)
FIXTURE_DATA_TEST_CASE(SmallROIAlignLayerFloat, CLROIAlignLayerFixture<float>, framework::DatasetMode::ALL,
                       framework::dataset::combine(framework::dataset::combine(datasets::SmallROIDataset(),
                                                                               framework::dataset::make("DataType", { DataType::F32 })),
                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, relative_tolerance_f32, .02f, absolute_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(SmallROIAlignLayerHalf, CLROIAlignLayerFixture<half>, framework::DatasetMode::ALL,
                       framework::dataset::combine(framework::dataset::combine(datasets::SmallROIDataset(),
                                                                               framework::dataset::make("DataType", { DataType::F16 })),
                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, relative_tolerance_f16, .02f, absolute_tolerance_f16);
}
TEST_SUITE_END() // Float

TEST_SUITE_END() // RoiAlign
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
