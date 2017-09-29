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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowp.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/GEMMLowpFixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

#include "arm_compute/core/NEON/kernels/NEGEMMInterleaveBlockedKernel.h"
#include "tests/NEON/Helper.h"

namespace arm_compute
{
namespace test
{
const auto data_int_blk = framework::dataset::make("M", 800) * framework::dataset::make("N", 800) * framework::dataset::make("by", 8, 13) * framework::dataset::make("block", 4, 9);

TEST_SUITE(NEON)

TEST_SUITE(INTERLEAVE_BLOCKED)
using NEInterleaveBlocked            = NESynthetizeFunction<NEGEMMInterleaveBlockedKernel>;
using NEGEMMInterleaveBlockedFixture = GEMMInterleaveBlockedFixture<Tensor, NEInterleaveBlocked, Accessor>;
REGISTER_FIXTURE_DATA_TEST_CASE(InterleaveBlocked, NEGEMMInterleaveBlockedFixture, framework::DatasetMode::ALL, data_int_blk);
TEST_SUITE_END()

#if 0  //FIXME: enable when we update NEGEMMLowp interface to work without offsets
TEST_SUITE(U32)
using NEGEMMLowpFixture = GEMMLowpFixture<Tensor, NEGEMMLowp, Accessor>;
REGISTER_FIXTURE_DATA_TEST_CASE(GEMMLowp, NEGEMMLowpFixture, framework::DatasetMode::ALL, framework::dataset::make("M", 100, 120) * framework::dataset::make("N", 100,
                                110)
                                * framework::dataset::make("K", 16, 20));

TEST_SUITE_END()
#endif // defined(__aarch64__)

TEST_SUITE_END()

} // namespace test
} // namespace arm_compute
