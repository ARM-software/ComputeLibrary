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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/TypePrinter.h"
#include "tests/datasets_new/GoogLeNetGEMMDataset.h"
#include "tests/fixtures_new/GEMMFixture.h"

namespace arm_compute
{
namespace test
{
namespace
{
auto data_types = framework::dataset::make("DataType",
{
#if ARM_COMPUTE_ENABLE_FP16
    DataType::FP16,
#endif /* ARM_COMPUTE_ENABLE_FP16 */
    DataType::F32
});
} // namespace

using CLGEMMFixture = GEMMFixture<CLTensor, CLGEMM>;

TEST_SUITE(CL)

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetGEMM, CLGEMMFixture, framework::DatasetMode::ALL, framework::dataset::combine(datasets::GoogLeNetGEMMDataset(), std::move(data_types)));

TEST_SUITE_END()
} // namespace test
} // namespace arm_compute
