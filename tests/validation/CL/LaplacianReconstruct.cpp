/*
 * Copyright (c) 2018 Arm Limited.
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
#include "arm_compute/runtime/CL/CLPyramid.h"
#include "arm_compute/runtime/CL/functions/CLLaplacianPyramid.h"
#include "arm_compute/runtime/CL/functions/CLLaplacianReconstruct.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/LaplacianReconstructFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto small_laplacian_reconstruct_levels = framework::dataset::make("NumLevels", 2, 3);
const auto large_laplacian_reconstruct_levels = framework::dataset::make("NumLevels", 2, 5);

const auto formats = combine(framework::dataset::make("FormatIn", Format::S16), framework::dataset::make("FormatOut", Format::U8));

template <typename T>
void validate_laplacian_reconstruct(CLTensor &target, const SimpleTensor<T> &reference, BorderMode border_mode, size_t num_levels)
{
    const unsigned int filter_size = 5;
    const unsigned int border_size(filter_size / 2);

    BorderSize border(std::pow(border_size, num_levels));

    // Validate output
    ValidRegion valid_region = shape_to_valid_region(reference.shape(), border_mode == BorderMode::UNDEFINED, border);
    validate(CLAccessor(target), reference, valid_region);
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(LaplacianReconstruct)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(
                                                           concat(datasets::Medium2DShapes(), datasets::Large2DShapes()),
                                                           datasets::BorderModes()),
                                                           large_laplacian_reconstruct_levels),
                                                           shape, border_mode, num_levels)
{
    // Create pyramid info
    PyramidInfo pyramid_info(num_levels, SCALE_PYRAMID_HALF, shape, Format::S16);
    CLPyramid   dst_pyramid{};
    dst_pyramid.init(pyramid_info);

    // Create Tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::U8);

    // The first two dimensions of the output tensor must match the first two
    // dimensions of the tensor in the last level of the pyramid
    TensorShape dst_shape(shape);
    dst_shape.set(0, dst_pyramid.get_pyramid_level(num_levels - 1)->info()->dimension(0));
    dst_shape.set(1, dst_pyramid.get_pyramid_level(num_levels - 1)->info()->dimension(1));
    CLTensor dst = create_tensor<CLTensor>(dst_shape, DataType::S16);

    // The dimensions of the reconstruct are the same as the src shape
    CLTensor rec_dst = create_tensor<CLTensor>(shape, DataType::U8);

    // Create and configure pyramid function
    CLLaplacianPyramid laplacian_pyramid;
    laplacian_pyramid.configure(&src, &dst_pyramid, &dst, border_mode, 0);

    // Create and configure reconstruct function
    CLLaplacianReconstruct laplacian_reconstruct;
    laplacian_reconstruct.configure(&dst_pyramid, &dst, &rec_dst, border_mode, 0);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    for(size_t level = 0; level < pyramid_info.num_levels(); ++level)
    {
        ARM_COMPUTE_EXPECT(dst_pyramid.get_pyramid_level(level)->info()->is_resizable(), framework::LogLevel::ERRORS);
    }

    ARM_COMPUTE_EXPECT(rec_dst.info()->is_resizable(), framework::LogLevel::ERRORS);
}

using CLLaplacianReconstructFixture = LaplacianReconstructValidationFixture<CLTensor, CLAccessor, CLLaplacianReconstruct, CLLaplacianPyramid, int16_t, uint8_t, CLPyramid>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLLaplacianReconstructFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(
                       datasets::Medium2DShapes(),
                       datasets::BorderModes()),
                       small_laplacian_reconstruct_levels),
                       formats))
{
    validate_laplacian_reconstruct(_target, _reference, _border_mode, _pyramid_levels);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLLaplacianReconstructFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(
                       datasets::Large2DShapes(),
                       datasets::BorderModes()),
                       large_laplacian_reconstruct_levels),
                       formats))
{
    validate_laplacian_reconstruct(_target, _reference, _border_mode, _pyramid_levels);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
