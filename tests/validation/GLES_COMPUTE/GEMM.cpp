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
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCGEMM.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/GLES_COMPUTE/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeGEMMDataset.h"
#include "tests/datasets/SmallGEMMDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F32,
});
} // namespace

TEST_SUITE(GC)
TEST_SUITE(GEMM)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallGEMMDataset(), datasets::LargeGEMMDataset()), CNNDataTypes),
               shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type)
{
    // Set fixed point position data type allowed
    const int fixed_point_position = is_data_type_fixed_point(data_type) ? 3 : 0;

    // Create tensors
    GCTensor a   = create_tensor<GCTensor>(shape_a, data_type, 1, fixed_point_position);
    GCTensor b   = create_tensor<GCTensor>(shape_b, data_type, 1, fixed_point_position);
    GCTensor c   = create_tensor<GCTensor>(shape_c, data_type, 1, fixed_point_position);
    GCTensor dst = create_tensor<GCTensor>(output_shape, data_type, 1, fixed_point_position);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    GCGEMM gemm;
    gemm.configure(&a, &b, &c, &dst, alpha, beta);
}

template <typename T>
using GCGEMMFixture = GEMMValidationFixture<GCTensor, GCAccessor, GCGEMM, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, GCGEMMFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallGEMMDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCGEMMFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
