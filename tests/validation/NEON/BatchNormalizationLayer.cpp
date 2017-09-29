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
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/RandomBatchNormalizationLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BatchNormalizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f32(0.00001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
#ifdef ARM_COMPUTE_ENABLE_FP16
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif                                                   /* ARM_COMPUTE_ENABLE_FP16 */
constexpr AbsoluteTolerance<float> tolerance_qs8(3.0f);  /**< Tolerance value for comparing reference's output against implementation's output for DataType::QS8 */
constexpr AbsoluteTolerance<float> tolerance_qs16(6.0f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::QS16 */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(BatchNormalizationLayer)

template <typename T>
using NEBatchNormalizationLayerFixture = BatchNormalizationLayerValidationFixture<Tensor, Accessor, NEBatchNormalizationLayer, T>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::RandomBatchNormalizationLayerDataset(), framework::dataset::make("DataType", { DataType::QS8, DataType::QS16, DataType::F32 })),
               shape0, shape1, epsilon, dt)
{
    // Set fixed point position data type allowed
    int fixed_point_position = (arm_compute::is_data_type_fixed_point(dt)) ? 3 : 0;

    // Create tensors
    Tensor src   = create_tensor<Tensor>(shape0, dt, 1, fixed_point_position);
    Tensor dst   = create_tensor<Tensor>(shape0, dt, 1, fixed_point_position);
    Tensor mean  = create_tensor<Tensor>(shape1, dt, 1, fixed_point_position);
    Tensor var   = create_tensor<Tensor>(shape1, dt, 1, fixed_point_position);
    Tensor beta  = create_tensor<Tensor>(shape1, dt, 1, fixed_point_position);
    Tensor gamma = create_tensor<Tensor>(shape1, dt, 1, fixed_point_position);

    // Create and Configure function
    NEBatchNormalizationLayer norm;
    norm.configure(&src, &dst, &mean, &var, &beta, &gamma, epsilon);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape0);
    validate(dst.info()->valid_region(), valid_region);
}

TEST_SUITE(Float)
FIXTURE_DATA_TEST_CASE(Random, NEBatchNormalizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::RandomBatchNormalizationLayerDataset(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32, 0);
}
TEST_SUITE_END()

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(Float16)
FIXTURE_DATA_TEST_CASE(Random, NEBatchNormalizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::RandomBatchNormalizationLayerDataset(),
                                                                                                                  framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16, 0);
}
TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(Quantized)
template <typename T>
using NEBatchNormalizationLayerFixedPointFixture = BatchNormalizationLayerValidationFixedPointFixture<Tensor, Accessor, NEBatchNormalizationLayer, T>;

TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(Random, NEBatchNormalizationLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::RandomBatchNormalizationLayerDataset(),
                       framework::dataset::make("DataType", DataType::QS8)),
                       framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs8, 0);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(Random, NEBatchNormalizationLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::RandomBatchNormalizationLayerDataset(),
                       framework::dataset::make("DataType", DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs16, 0);
}
TEST_SUITE_END()

TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
