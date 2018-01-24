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
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCNormalizePlanarYUVLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/RandomNormalizePlanarYUVLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/NormalizePlanarYUVLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f16(0.5f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
} // namespace

TEST_SUITE(GC)
TEST_SUITE(NormalizePlanarYUVLayer)

template <typename T>
using GCNormalizePlanarYUVLayerFixture = NormalizePlanarYUVLayerValidationFixture<GCTensor, GCAccessor, GCNormalizePlanarYUVLayer, T>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::RandomNormalizePlanarYUVLayerDataset(), framework::dataset::make("DataType", { DataType::F16 })),
               shape0, shape1, dt)
{
    // Create tensors
    GCTensor src  = create_tensor<GCTensor>(shape0, dt, 1);
    GCTensor dst  = create_tensor<GCTensor>(shape0, dt, 1);
    GCTensor mean = create_tensor<GCTensor>(shape1, dt, 1);
    GCTensor sd   = create_tensor<GCTensor>(shape1, dt, 1);

    // Create and Configure function
    GCNormalizePlanarYUVLayer norm;
    norm.configure(&src, &dst, &mean, &sd);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape0);
    validate(dst.info()->valid_region(), valid_region);
}

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Random, GCNormalizePlanarYUVLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::RandomNormalizePlanarYUVLayerDataset(),
                                                                                                                  framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16, 0);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
