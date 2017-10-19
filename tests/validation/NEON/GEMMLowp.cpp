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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowp.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMInterleaveBlockedFixture.h"
#include "tests/validation/fixtures/GEMMLowpFixture.h"

#include "arm_compute/core/NEON/kernels/NEGEMMInterleaveBlockedKernel.h"
#include "tests/NEON/Helper.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */

const auto data_mnk     = framework::dataset::make("M", 12, 20) * framework::dataset::make("N", 12, 20) * framework::dataset::make("K", 12, 15);
const auto data_offsets = framework::dataset::make("a", -3, 3) * framework::dataset::make("b", -1, 2) * framework::dataset::make("c", 1, 3) * framework::dataset::make("cm", 0,
                          3)
                          * framework::dataset::make("shift", 0, 4);

const auto data_int_blk = framework::dataset::make("M", 8, 12) * framework::dataset::make("N", 8, 12) * framework::dataset::make("by", 8, 13) * framework::dataset::make("block", 4, 9);

const auto data_int_blk_tr = framework::dataset::make("M", 8, 17) * framework::dataset::make("N", 8, 14) * framework::dataset::make("by", 12) * framework::dataset::make("block", 4);

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(GEMMLowp)

TEST_SUITE(U8)

TEST_SUITE(INTERLEAVE_BLOCKED)

using NEInterleaveBlocked            = NESynthetizeFunction<NEGEMMInterleaveBlockedKernel>;
using NEGEMMInterleaveBlockedFixture = GEMMInterleaveBlockedValidationFixture<Tensor, Accessor, NEInterleaveBlocked>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMInterleaveBlockedFixture, framework::DatasetMode::PRECOMMIT, data_int_blk)
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END()

TEST_SUITE(INTERLEAVE_BLOCKED_TRANSPOSED)
using NEInterleaveBlockedTransposed            = NESynthetizeFunction<NEGEMMInterleaveBlockedKernel>;
using NEGEMMInterleaveBlockedTransposedFixture = GEMMInterleaveBlockedValidationFixture<Tensor, Accessor, NEInterleaveBlockedTransposed, true>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMInterleaveBlockedTransposedFixture, framework::DatasetMode::PRECOMMIT, data_int_blk_tr)
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}

TEST_SUITE_END()

using NEGEMMLowpOffsetFixture = GEMMLowpOffsetValidationFixture<Tensor, Accessor, NEGEMMLowp>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpOffsetFixture, framework::DatasetMode::PRECOMMIT, data_mnk *data_offsets)
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END()

//FIXME: This is in the process of being updated, for more info please refer to COMPMID-624.
#if 0  // defined(__aarch64__)
TEST_SUITE(U32)
using NEGEMMLowpFixture = GEMMLowpValidationFixture<Tensor, Accessor, NEGEMMLowp>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpFixture, framework::DatasetMode::PRECOMMIT, framework::dataset::make("M", 12, 20) * framework::dataset::make("N", 12, 20) * framework::dataset::make("K",
                       16))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END()
#endif // defined(__aarch64__)
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
