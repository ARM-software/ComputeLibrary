/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "src/gpu/cl/kernels/ClIndirectConv2dAddressPrecalculationKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/IndirectConv2dAddressPrecalculationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::opencl::kernels;

using CLIndirectConv2dAddressPrecalculation = CLSynthetizeOperator<ClIndirectConv2dAddressPrecalculationKernel>;

using CLIndirectConv2dAddressPrecalculationFixture = IndirectConv2dAddressPrecalculationValidationFixture<CLTensor, CLAccessor, CLIndirectConv2dAddressPrecalculation>;

// *INDENT-OFF*
// clang-format off
/** Data types */

namespace
{
const auto src_w_values  = framework::dataset::make("src_w", {91});
const auto src_h_values  = framework::dataset::make("src_h", {103});
const auto src_b_values  = framework::dataset::make("src_b", {1, 2});
const auto wei_w_values  = framework::dataset::make("wei_w", {3, 5});
const auto wei_h_values  = framework::dataset::make("wei_h", {1, 6});
const auto pad_values    = framework::dataset::make("pad", {1, 2, 3});
const auto stride_values = framework::dataset::make("stride", {1, 2});
const auto m0_values     = framework::dataset::make("M0", { 1, 2, 4, 5, 7 });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(IndirectConv2dAddressPrecalculation)

FIXTURE_DATA_TEST_CASE(RunSmall, CLIndirectConv2dAddressPrecalculationFixture, framework::DatasetMode::ALL,
                combine(combine(combine(combine(combine(combine(combine(src_w_values,
                                                                        src_h_values),
                                                                        src_b_values),
                                                                        wei_w_values),
                                                                        wei_h_values),
                                                                        pad_values),
                                                                        stride_values),
                                                                        m0_values))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // IndirectConv2dAddressPrecalculation
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
