/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/CPP/functions/CPPPermute.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PermuteFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto PermuteVectors = framework::dataset::make("PermutationVector",
{
    PermutationVector(2U, 0U, 1U),
    PermutationVector(1U, 2U, 0U),
    PermutationVector(0U, 1U, 2U),
    PermutationVector(0U, 2U, 1U),
    PermutationVector(1U, 0U, 2U),
    PermutationVector(2U, 1U, 0U),
});
const auto PermuteInputLayout     = framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC });
const auto PermuteParametersSmall = concat(concat(datasets::Small2DShapes(), datasets::Small3DShapes()), datasets::Small4DShapes()) * PermuteInputLayout * PermuteVectors;
const auto PermuteParametersLarge = datasets::Large4DShapes() * PermuteInputLayout * PermuteVectors;

} // namespace
TEST_SUITE(CPP)
TEST_SUITE(Permute)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::Small4DShapes(), framework::dataset::make("DataType", { DataType::S8, DataType::U8, DataType::S16, DataType::U16, DataType::U32, DataType::S32, DataType::F16, DataType::F32 })),
               shape, data_type)
{
    // Define permutation vector
    const PermutationVector perm(2U, 0U, 1U);

    // Permute shapes
    TensorShape output_shape = shape;
    permute(output_shape, perm);

    // Create tensors
    Tensor ref_src = create_tensor<Tensor>(shape, data_type);
    Tensor dst     = create_tensor<Tensor>(output_shape, data_type);

    // Create and Configure function
    CPPPermute perm_func;
    perm_func.configure(&ref_src, &dst, perm);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(output_shape);
    validate(dst.info()->valid_region(), valid_region);
}

template <typename T>
using CPPPermuteFixture = PermuteValidationFixture<Tensor, Accessor, CPPPermute, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CPPPermuteFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CPPPermuteFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END()

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall, CPPPermuteFixture<uint16_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CPPPermuteFixture<uint16_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmall, CPPPermuteFixture<uint32_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CPPPermuteFixture<uint32_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
