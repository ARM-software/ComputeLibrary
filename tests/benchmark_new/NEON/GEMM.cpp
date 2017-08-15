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
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/TypePrinter.h"
#include "tests/datasets_new/MatrixMultiplyGEMMDataset.h"
#include "tests/datasets_new/system_tests/googlenet/inceptionv1/GoogLeNetInceptionV1GEMMDataset.h"
#include "tests/fixtures_new/GEMMFixture.h"

namespace arm_compute
{
namespace test
{
namespace
{
const auto data_types = framework::dataset::make("DataType",
{
#if ARM_COMPUTE_ENABLE_FP16
    DataType::F16,
#endif /* ARM_COMPUTE_ENABLE_FP16 */
    DataType::F32,
    DataType::QS8
});
} // namespace

using NEGEMMFixture = GEMMFixture<Tensor, NEGEMM>;

TEST_SUITE(NEON)

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1GEMM, NEGEMMFixture, framework::DatasetMode::ALL, framework::dataset::combine(datasets::GoogLeNetInceptionV1GEMMDataset(), data_types));
REGISTER_FIXTURE_DATA_TEST_CASE(MatrixMultiplyGEMM, NEGEMMFixture, framework::DatasetMode::ALL, framework::dataset::combine(datasets::MatrixMultiplyGEMMDataset(), data_types));

TEST_SUITE_END()
} // namespace test
} // namespace arm_compute
