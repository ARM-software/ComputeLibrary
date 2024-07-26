/*
 * Copyright (c) 2017-2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuWinogradConv2d.h"

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/experimental/Types.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/Utils.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"
#include "tests/validation/fixtures/CpuWinogradConv2dFixture.h"
#include "tests/validation/Validation.h"
/*
 * Tests for arm_compute::experimental::op::CpuWinogradConv2d which is a shallow wrapper for
 * arm_compute::cpu::CpuWinogradConv2d. Any future testing to the functionalities of cpu::CpuWinogradConv2d will
 * be tested in tests/validation/NEON/ConvolutionLayer.cpp given that op::CpuWinogradConv2d remain a shallow wrapper.
*/

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;
namespace
{
const AbsoluteTolerance<float> abs_tolerance_f32(0.002f); /**< Absolute tolerance for FP32 types */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuWinogradConv2d)
/** Test case for memory injection in @ref arm_compute::experimental::op::CpuWinogradConv2d.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(OpCpuWinogradConv2dMemoryInjection, framework::DatasetMode::ALL)
{
    auto                winograd = std::make_unique<cpu::CpuWinogradConv2d>();
    const auto          src_info = TensorInfo(TensorShape(8U, 8U, 32U), 1, DataType::F32);
    const auto          w_info   = TensorInfo(TensorShape(1U), 1, DataType::F32);
    const auto          b_info   = TensorInfo(TensorShape(1U, 3U, 32U, 1U), 1, DataType::F32);
    auto                dst_info = TensorInfo(TensorShape(8U, 6U, 1U), 1, DataType::F32);
    const PadStrideInfo pad_info{};

    winograd->configure(&src_info, &b_info, &w_info, &dst_info, pad_info);

    // telhs are newly created every call of this lambda function
    auto a = create_tensor<Tensor>(src_info);
    auto b = create_tensor<Tensor>(b_info);
    auto c = create_tensor<Tensor>(w_info);
    a.allocator()->allocate();
    b.allocator()->allocate();
    c.allocator()->allocate();

    ITensorPack run_pack{{TensorType::ACL_SRC_0, &a}, {TensorType::ACL_SRC_1, &b}, {TensorType::ACL_SRC_2, &c}};
    ITensorPack prep_pack{{TensorType::ACL_SRC_1, &b}, {TensorType::ACL_SRC_2, &c}};

    auto mg       = MemoryGroup{};
    auto ws       = manage_workspace<Tensor>(winograd->workspace(), mg, run_pack, prep_pack);
    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();

        run_pack.add_tensor(TensorType::ACL_DST, &dst);
        library->fill_tensor_value(Accessor(a), 1.f);
        library->fill_tensor_value(Accessor(b), 2.f);
        library->fill_tensor_value(Accessor(c), 3.f);

        // This operator is configured once and captured by this lambda.
        winograd->prepare(prep_pack);
        winograd->run(run_pack);
        return dst;
    };

    auto result_0 = run_conv();
    auto result_1 = run_conv();

    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i],
                           framework::LogLevel::ERRORS);
    }
}

using CpuWinogradConv2dFixture =
    CpuWinogradConv2dValidationFixture<Tensor, Accessor, experimental::op::CpuWinogradConv2d>;

const auto ActivationFunctionsDataset =
    make("ActivationInfo",
         {ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
          ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f)});

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuWinogradConv2dFixture,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallWinogradConvolutionLayer3x3Dataset(), ActivationFunctionsDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, abs_tolerance_f32);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // CpuWinogradConv2d
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
