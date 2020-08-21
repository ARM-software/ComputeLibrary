/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NECannyEdge.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/ArrayAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ImageFileDatasets.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CannyEdgeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/* Allowed ratio of mismatches between target and reference (1.0 = 100%) */
const float allowed_mismatch_ratio = 0.1f;

const auto data = combine(framework::dataset::make("GradientSize", { 3, 5, 7 }),
                          combine(framework::dataset::make("Normalization", { MagnitudeType::L1NORM, MagnitudeType::L2NORM }), datasets::BorderModes()));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(CannyEdge)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::Small2DShapes(), datasets::Large2DShapes()), data), framework::dataset::make("Format", Format::U8)),
               shape, gradient_size, normalization, border_mode, format)
{
    CannyEdgeParameters params = canny_edge_parameters();
    // Convert normalisation type to integer
    const auto norm_type = static_cast<int>(normalization) + 1;

    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type_from_format(format));
    Tensor dst = create_tensor<Tensor>(shape, data_type_from_format(format));
    src.info()->set_format(format);
    dst.info()->set_format(format);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create Canny edge configure function
    NECannyEdge canny_edge;
    canny_edge.configure(&src, &dst, params.upper_thresh, params.lower_thresh, gradient_size, norm_type, border_mode, params.constant_border_value);

    // Validate valid region
    validate(src.info()->valid_region(), shape_to_valid_region(shape, (BorderMode::UNDEFINED == border_mode)));
    validate(dst.info()->valid_region(), shape_to_valid_region(shape, (BorderMode::UNDEFINED == border_mode)));

    // Validate padding
    PaddingCalculator calculator(shape.x(), 8);
    calculator.set_border_mode(border_mode);
    calculator.set_border_size(gradient_size / 2);
    calculator.set_access_offset(-gradient_size / 2);
    calculator.set_accessed_elements(16);

    validate(src.info()->padding(), calculator.required_padding());
    validate(dst.info()->padding(), PaddingSize{ 1 });
}

template <typename T>
using NECannyEdgeFixture = CannyEdgeValidationFixture<Tensor, Accessor, KeyPointArray, NECannyEdge, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NECannyEdgeFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallImageFiles(), data), framework::dataset::make("Format", Format::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, AbsoluteTolerance<uint8_t>(0), allowed_mismatch_ratio);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NECannyEdgeFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeImageFiles(), data), framework::dataset::make("Format", Format::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference, AbsoluteTolerance<uint8_t>(0), allowed_mismatch_ratio);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
