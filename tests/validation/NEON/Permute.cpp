/*
 * Copyright (c) 2018-2020, 2024 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Helpers.h"
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
const auto PermuteVectors2 = framework::dataset::make("PermutationVector",
{
    PermutationVector(0U, 1U),
    PermutationVector(1U, 0U),
});
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
const auto PermuteVectors         = concat(concat(PermuteVectors2, PermuteVectors3), PermuteVectors4);
const auto PermuteParametersSmall = concat(concat(datasets::Small2DShapes(), datasets::Small3DShapes()), datasets::Small4DShapes()) * PermuteVectors;
const auto PermuteParametersLarge = datasets::Large4DShapes() * PermuteVectors;
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Permute)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // permutation not supported
                                                                                        TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // permutation not supported
                                                                                        TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // permutation not supported
                                                                                        TensorInfo(TensorShape(1U, 7U), 1, DataType::U8),              // invalid input size
                                                                                        TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // valid
                                                                                        TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32),  // valid
                                                                                        TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::U16),     // permutation not supported
                                                                                        TensorInfo(TensorShape(7U, 7U, 5U, 3U), 1, DataType::S16),     // permutation not supported
                                                                                        TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32),  // permutation not supported
                                                                                        TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32),  // permutation not supported
                                                                                        TensorInfo(TensorShape(27U, 13U, 37U, 2U), 1, DataType::F32)  // permutation not supported

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
                                                framework::dataset::make("Expected", { true, false, false, false, true, true, false,true, false, true, false })),
                                            input_info, output_info, perm_vect, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEPermute::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), perm_vect)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEPermuteFixture = PermuteValidationFixture<Tensor, Accessor, NEPermute, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPermuteFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPermuteFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPermuteFixture<uint16_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPermuteFixture<uint16_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(U32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPermuteFixture<uint32_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPermuteFixture<uint32_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPermuteFixture<float16_t>, framework::DatasetMode::PRECOMMIT,
                       PermuteParametersSmall * framework::dataset::make("DataType", DataType::F16))
{
    if (cpu_supports_dtypes({DataType::F16}))
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPermuteFixture<float16_t>, framework::DatasetMode::NIGHTLY,
                       PermuteParametersLarge * framework::dataset::make("DataType", DataType::F16))
{
    if (cpu_supports_dtypes({DataType::F16}))
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
