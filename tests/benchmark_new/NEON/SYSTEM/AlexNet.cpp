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
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/SubTensor.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/NEON/Accessor.h"
#include "tests/TypePrinter.h"
#include "tests/fixtures_new/AlexNetFixture.h"

namespace arm_compute
{
namespace test
{
namespace
{
#ifdef ARM_COMPUTE_ENABLE_FP16
const auto alex_net_data_types = framework::dataset::make("DataType", { DataType::F16, DataType::F32, DataType::QS8 });
#else  /* ARM_COMPUTE_ENABLE_FP16 */
const auto alex_net_data_types = framework::dataset::make("DataType", { DataType::F32, DataType::QS8 });
#endif /* ARM_COMPUTE_ENABLE_FP16 */
} // namespace

using NEAlexNetFixture = AlexNetFixture<ITensor,
      Tensor,
      SubTensor,
      Accessor,
      NEActivationLayer,
      NEConvolutionLayer,
      NEDirectConvolutionLayer,
      NEFullyConnectedLayer,
      NENormalizationLayer,
      NEPoolingLayer,
      NESoftmaxLayer>;

TEST_SUITE(NEON)
TEST_SUITE(SYSTEM_TEST)

REGISTER_FIXTURE_DATA_TEST_CASE(AlexNet, NEAlexNetFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(alex_net_data_types,
                                                            framework::dataset::make("Batches", { 1, 4, 8 })));

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace test
} // namespace arm_compute
