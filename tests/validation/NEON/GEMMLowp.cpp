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
#include "arm_compute/core/NEON/kernels/NEGEMMInterleaveBlockedKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowp.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/datasets/LargeGEMMLowpDataset.h"
#include "tests/datasets/SmallGEMMLowpDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMInterleaveBlockedFixture.h"
#include "tests/validation/fixtures/GEMMLowpFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto data_int_blk         = framework::dataset::make("M", 8, 12) * framework::dataset::make("N", 8, 12) * framework::dataset::make("by", 8, 13) * framework::dataset::make("block", 4, 9);
const auto data_int_blk_tr      = framework::dataset::make("M", 8, 17) * framework::dataset::make("N", 8, 14) * framework::dataset::make("by", 12) * framework::dataset::make("block", 4);
const auto data_matrix_multiply = framework::dataset::make("M", 12, 20) * framework::dataset::make("N", 12, 20) * framework::dataset::make("K", 16);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(GEMMLowp)

TEST_SUITE(S8)

TEST_SUITE(INTERLEAVE_BLOCKED)

using NEInterleaveBlocked            = NESynthetizeFunction<NEGEMMInterleaveBlockedKernel>;
using NEGEMMInterleaveBlockedFixture = GEMMInterleaveBlockedValidationFixture<Tensor, Accessor, NEInterleaveBlocked>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMInterleaveBlockedFixture, framework::DatasetMode::PRECOMMIT, data_int_blk)
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(INTERLEAVE_BLOCKED_TRANSPOSED)
using NEInterleaveBlockedTransposed            = NESynthetizeFunction<NEGEMMInterleaveBlockedKernel>;
using NEGEMMInterleaveBlockedTransposedFixture = GEMMInterleaveBlockedValidationFixture<Tensor, Accessor, NEInterleaveBlockedTransposed, true>;
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMInterleaveBlockedTransposedFixture, framework::DatasetMode::PRECOMMIT, data_int_blk_tr)
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END()

using NEGEMMLowpOffsetFixture = GEMMLowpOffsetValidationFixture<Tensor, Accessor, NEGEMMLowp>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallGEMMLowpDataset(), datasets::LargeGEMMLowpDataset()), framework::dataset::make("DataType",
                                                                   DataType::S8)),
               shape_a, shape_b, shape_c, a_offset, b_offset, c_offset, c_mult_int, out_shift, data_type)
{
    // Create tensors
    Tensor a = create_tensor<Tensor>(shape_a, data_type);
    Tensor b = create_tensor<Tensor>(shape_b, data_type);
    Tensor c = create_tensor<Tensor>(shape_c, data_type);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEGEMMLowp gemmlowp;
    gemmlowp.configure(&a, &b, &c, a_offset, b_offset, c_offset, c_mult_int, out_shift);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMLowpOffsetFixture, framework::DatasetMode::ALL, combine(datasets::SmallGEMMLowpDataset(), framework::dataset::make("DataType", DataType::S8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMLowpOffsetFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeGEMMLowpDataset(), framework::dataset::make("DataType", DataType::S8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S32)
using NEGEMMLowpMatrixMultiplyFixture = GEMMLowpMatrixMultiplyValidationFixture<Tensor, Accessor, NEGEMMLowpMatrixMultiplyCore>;
FIXTURE_DATA_TEST_CASE(MatrixMultiply, NEGEMMLowpMatrixMultiplyFixture, framework::DatasetMode::PRECOMMIT, data_matrix_multiply)
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
