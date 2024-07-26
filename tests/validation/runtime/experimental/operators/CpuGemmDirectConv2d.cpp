/*
 * Copyright (c) 2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuGemmDirectConv2d.h"

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/experimental/Types.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/datasets/TinyConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/Utils.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"
#include "tests/validation/fixtures/CpuGemmDirectConv2dFixture.h"
#include "tests/validation/Validation.h"
/*
 * Tests for arm_compute::experimental::op::CpuGemmDirectConv2d which is a shallow wrapper for
 * arm_compute::cpu::CpuGemmDirectConv2d. Any future testing to the functionalities of cpu::CpuGemmDirectConv2d will
 * be tested in tests/validation/NEON/ConvolutionLayer.cpp given that op::CpuGemmDirectConv2d remain a shallow wrapper.
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
}
TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)

TEST_SUITE(CpuGemmDirectConv2d)
/** Test case for memory injection in @ref arm_compute::experimental::op::CpuGemmDirectConv2d.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(OpCpuGemmDirectConv2dMemoryInjection, framework::DatasetMode::ALL)
{
    auto conv = std::make_unique<arm_compute::experimental::op::CpuGemmDirectConv2d>();

    auto src_shape     = TensorShape(23U, 27U, 5U);
    auto weights_shape = TensorShape(23U, 3U, 5U, 21U);
    auto bias_shape    = TensorShape(21U);
    auto output_shape  = TensorShape(11U, 25U, 21U);

    const auto src_info     = TensorInfo(src_shape, 1, DataType::F32, DataLayout::NHWC);
    const auto weights_info = TensorInfo(weights_shape, 1, DataType::F32, DataLayout::NHWC);
    const auto biases_info  = TensorInfo(bias_shape, 1, DataType::F32, DataLayout::NHWC);
    auto       dst_info     = TensorInfo(output_shape, 1, DataType::F32, DataLayout::NHWC);
    const auto conv_info    = Conv2dInfo{PadStrideInfo(2, 1, 0, 0), Size2D(1, 1), ActivationLayerInfo(), false, 1};

    conv->configure(&src_info, &weights_info, &biases_info, &dst_info, conv_info);
    auto const status = conv->validate(&src_info, &weights_info, &biases_info, &dst_info, conv_info);
    ARM_COMPUTE_ASSERT(status);

    // tensors are newly created every call of this lambda function
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
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i],
                           framework::LogLevel::ERRORS);
    }
}

using CpuGemmDirectConv2dFixture =
    CpuGemmDirectConv2dValidationFixture<Tensor, Accessor, experimental::op::CpuGemmDirectConv2d>;

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuGemmDirectConv2dFixture,
                       framework::DatasetMode::PRECOMMIT,
                       datasets::TinyConvolutionLayerDataset())
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f32);
}
TEST_SUITE_END() // F32

TEST_SUITE_END() // CPUGEMMDIRECTCONV2D
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
