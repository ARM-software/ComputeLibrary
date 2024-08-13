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
#include "arm_compute/runtime/experimental/operators/CpuGemmConv2d.h"

#include "arm_compute/core/CoreTypes.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/datasets/TinyConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/Utils.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"
#include "tests/validation/fixtures/CpuGemmConv2dFixture.h"
#include "tests/validation/Validation.h"
/*
 * Tests for arm_compute::experimental::op::CpuGemmGemmConv2d which is a shallow wrapper for
 * arm_compute::cpu::CpuGemmConv2d. Any future testing to the functionalities of cpu::CpuGemmConv2d will
 * be tested in tests/validation/NEON/ConvolutionLayer.cpp given that op::CpuGemmConv2d remain a shallow wrapper.
*/

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const RelativeTolerance<float> rel_tolerance_f32(0.01f);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuGemmConv2d)
/** Test case for memory injection in @ref arm_compute::experimental::op::CpuGemmConv2d.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(OpCpuGemmConv2dMemoryInjection, framework::DatasetMode::ALL)
{
    auto conv = std::make_unique<arm_compute::experimental::op::CpuGemmConv2d>();

    const auto src_info     = TensorInfo(TensorShape(1U, 5U, 2U), 1, DataType::F32, DataLayout::NCHW);
    const auto weights_info = TensorInfo(TensorShape(1U, 3U, 2U, 3U), 1, DataType::F32, DataLayout::NCHW);
    const auto biases_info  = TensorInfo(TensorShape(3U), 1, DataType::F32, DataLayout::NCHW);
    auto       dst_info     = TensorInfo(TensorShape(1U, 7U, 3U), 1, DataType::F32, DataLayout::NCHW);
    const auto pad_info     = PadStrideInfo(1, 1, 0, 0, 2, 2, DimensionRoundingType::FLOOR);

    conv->configure(&src_info, &weights_info, &biases_info, &dst_info, pad_info);
    auto const status = conv->validate(&src_info, &weights_info, &biases_info, &dst_info, pad_info);
    ARM_COMPUTE_ASSERT(status);

    auto src     = create_tensor<Tensor>(src_info);
    auto weights = create_tensor<Tensor>(weights_info);
    auto biases  = create_tensor<Tensor>(biases_info);

    src.allocator()->allocate();
    weights.allocator()->allocate();
    biases.allocator()->allocate();

    ITensorPack run_pack{
        {TensorType::ACL_SRC_0, &src}, {TensorType::ACL_SRC_1, &weights}, {TensorType::ACL_SRC_2, &biases}};
    ITensorPack prep_pack{{TensorType::ACL_SRC_1, &weights}, {TensorType::ACL_SRC_2, &biases}};

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(conv->workspace(), mg, run_pack, prep_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(src), 1.f);
        library->fill_tensor_value(Accessor(weights), 2.f);
        library->fill_tensor_value(Accessor(biases), 3.f);
        // This operator is configured once and captured by this lambda.
        conv->prepare(prep_pack);
        conv->run(run_pack);
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT((reinterpret_cast<float *>(result_0.buffer()))[i] ==
                               (reinterpret_cast<float *>(result_1.buffer()))[i],
                           framework::LogLevel::ERRORS);
    }
}

using CpuGemmConv2dFixture = CpuGemmConv2dValidationFixture<Tensor, Accessor, experimental::op::CpuGemmConv2d>;

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuGemmConv2dFixture,
                       framework::DatasetMode::PRECOMMIT,
                       datasets::TinyConvolutionLayerDataset())
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32);
}
TEST_SUITE_END() // F32

TEST_SUITE_END() // CpuGemmConv2d
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
