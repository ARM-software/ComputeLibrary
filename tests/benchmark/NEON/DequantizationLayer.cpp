/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/DequantizationLayerFixture.h"
#include "tests/datasets/ShapeDatasets.h"
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
const auto data_types_src = framework::dataset::make("DataType", { DataType::QASYMM8 });
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const auto data_types_dst = framework::dataset::make("DataType", { DataType::F16, DataType::F32 });
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const auto data_types_dst = framework::dataset::make("DataType", { DataType::F32 });
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace

using NEDequantizationLayerFixture = DequantizationLayerFixture<Tensor, NEDequantizationLayer, Accessor>;

TEST_SUITE(NEON)

REGISTER_FIXTURE_DATA_TEST_CASE(DequantizationLayer, NEDequantizationLayerFixture,
                                framework::DatasetMode::ALL,
                                framework::dataset::combine(framework::dataset::combine(datasets::Small3DShapes(), data_types_src), data_types_dst));

TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
