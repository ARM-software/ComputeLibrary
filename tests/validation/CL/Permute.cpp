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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLPermute.h"
#include "tests/CL/CLAccessor.h"
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
const auto PermuteVectors3 = framework::dataset::make("PermutationVector",
{
    PermutationVector(2U, 0U, 1U),
    PermutationVector(1U, 2U, 0U),
    PermutationVector(0U, 1U, 2U),
    PermutationVector(0U, 2U, 1U),
    PermutationVector(1U, 0U, 2U),
    PermutationVector(2U, 1U, 0U),
});
const auto PermuteVectors4 = framework::dataset::make("PermutationVector",
{
    PermutationVector(3U, 2U, 0U, 1U),
    PermutationVector(3U, 2U, 1U, 0U),
    PermutationVector(2U, 3U, 1U, 0U),
    PermutationVector(1U, 3U, 2U, 0U),
    PermutationVector(3U, 1U, 2U, 0U),
    PermutationVector(3U, 0U, 2U, 1U),
    PermutationVector(0U, 3U, 2U, 1U)
});
const auto PermuteVectors         = concat(PermuteVectors3, PermuteVectors4);
const auto PermuteInputLayout     = framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC });
const auto PermuteParametersSmall = concat(concat(datasets::Small2DShapes(), datasets::Small3DShapes()), datasets::Small4DShapes()) * PermuteInputLayout * PermuteVectors;
const auto PermuteParametersLarge = datasets::Large4DShapes() * PermuteInputLayout * PermuteVectors;
} // namespace
TEST_SUITE(CL)
TEST_SUITE(Permute)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo",{
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // valid
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // permutation not supported
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // permutation not supported
                TensorInfo(TensorShape(1U, 7U), 1, DataType::U8),              // invalid input size
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // valid
                TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32),  // valid
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // permutation not supported
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::S16),     // valid
                TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32),  // permutation not supported
                TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32),  // valid
                TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32)   // permutation not supported

        }),
        framework::dataset::make("OutputInfo", {
                TensorInfo(TensorShape(5U, 7U, 7U, 3U), 1, DataType::U16),
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),
                TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),
                TensorInfo(TensorShape(5U, 7U), 1, DataType::U8),
                TensorInfo(TensorShape(5U, 7U, 7U, 3U), 1, DataType::U16),
                TensorInfo(TensorShape(13U, 37U, 27U, 2U), 1, DataType::F32),
                TensorInfo(TensorShape(5U, 7U, 7U, 3U), 1, DataType::U16),
                TensorInfo(TensorShape(3U, 5U, 7U, 7U), 1, DataType::S16),
                TensorInfo(TensorShape(13U, 37U, 27U, 2U), 1, DataType::F32),
                TensorInfo(TensorShape(37U, 2U, 13U, 27U), 1, DataType::F32),
                TensorInfo(TensorShape(37U, 2U, 13U, 27U), 1, DataType::F32)

        })),
        framework::dataset::make("PermutationVector", {
                PermutationVector(2U, 1U, 0U),
                PermutationVector(2U, 2U, 1U),
                PermutationVector(1U, 1U, 1U),
                PermutationVector(2U, 0U, 1U),
                PermutationVector(2U, 0U, 1U),
                PermutationVector(1U, 2U, 0U),
                PermutationVector(3U, 2U, 0U, 1U),
                PermutationVector(3U, 2U, 0U, 1U),
                PermutationVector(2U, 3U, 1U, 0U),
                PermutationVector(2U, 3U, 1U, 0U),
                PermutationVector(0U, 0U, 0U, 1000U)
        })),
        framework::dataset::make("Expected", { true, false, false, false, true, true, false, true, false, true, false })),
        input_info, output_info, perm_vect, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPermute::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), perm_vect)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::Small4DShapes(), framework::dataset::make("DataType", { DataType::S8, DataType::U8, DataType::S16, DataType::U16, DataType::U32, DataType::S32, DataType::F16, DataType::F32 })),
               shape, data_type)
{
    // Define permutation vector
    const PermutationVector perm(2U, 0U, 1U);

    // Permute shapes
    TensorShape output_shape = shape;
    permute(output_shape, perm);

    // Create tensors
    CLTensor ref_src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst     = create_tensor<CLTensor>(output_shape, data_type);

    // Create and Configure function
    CLPermute perm_func;
    perm_func.configure(&ref_src, &dst, perm);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(output_shape);
    validate(dst.info()->valid_region(), valid_region);
}

#ifndef DOXYGEN_SKIP_THIS

template <typename T>
using CLPermuteFixture = PermuteValidationFixture<CLTensor, CLAccessor, CLPermute, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPermuteFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U8))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLPermuteFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U8))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPermuteFixture<uint16_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPermuteFixture<uint16_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPermuteFixture<uint32_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPermuteFixture<uint32_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U32

#endif /* DOXYGEN_SKIP_THIS */

TEST_SUITE_END() // Permute
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
