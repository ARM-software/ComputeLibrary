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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCGEMM.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/benchmark/fixtures/GEMMFixture.h"
#include "tests/datasets/GoogleNetGEMMDataset.h"
#include "tests/datasets/MatrixMultiplyGEMMDataset.h"
#include "tests/datasets/system_tests/googlenet/inceptionv1/GoogLeNetInceptionV1GEMMDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
namespace
{
const auto data_types          = framework::dataset::make("DataType", { DataType::F32 });
const auto reshape_b_only_once = framework::dataset::make("ReshapeBOnlyOnce", { false });
} // namespace

using GCGEMMFixture = GEMMFixture<GCTensor, GCGEMM, GCAccessor>;

TEST_SUITE(GC)

REGISTER_FIXTURE_DATA_TEST_CASE(GoogLeNetInceptionV1GEMM, GCGEMMFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::GoogLeNetInceptionV1GEMMDataset(),
                                                                                        data_types),
                                                            reshape_b_only_once));
REGISTER_FIXTURE_DATA_TEST_CASE(MatrixMultiplyGEMM, GCGEMMFixture, framework::DatasetMode::ALL, framework::dataset::combine(framework::dataset::combine(datasets::MatrixMultiplyGEMMDataset(),
                                data_types),
                                reshape_b_only_once));
REGISTER_FIXTURE_DATA_TEST_CASE(GoogleNetGEMM, GCGEMMFixture, framework::DatasetMode::NIGHTLY, framework::dataset::combine(framework::dataset::combine(datasets::GoogleNetGEMMDataset(),
                                data_types),
                                reshape_b_only_once));

TEST_SUITE_END()
} // namespace benchmark
} // namespace test
} // namespace arm_compute
