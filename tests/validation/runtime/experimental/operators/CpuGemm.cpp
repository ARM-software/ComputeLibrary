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
#include "arm_compute/runtime/experimental/operators/CpuGemm.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/GEMMFixture.h"

/*
 * Tests for arm_compute::experimental::op::CpuGemm which is a shallow wrapper for
 * arm_compute::cpu::CpuGemm. Any future testing to the functionalities of cpu::CpuGemm will
 * be tested in tests/NEON/GEMM.cpp given that op::CpuGemm remain a shallow wrapper.
*/

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)

TEST_SUITE(CpuGemm)
/** Test case for memory injection in @ref arm_compute::experimental::op::CpuGemm.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(OpCpuGemmMemoryInjection, framework::DatasetMode::ALL)
{
    auto       gemm      = std::make_unique<arm_compute::experimental::op::CpuGemm>();
    const auto lhs_info  = TensorInfo(TensorShape(3U, 3U), 1, DataType::F32);
    const auto rhs_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto c_info    = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    auto       dst_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto gemm_info = GEMMInfo{};
    gemm->configure(&lhs_info, &rhs_info, &c_info, &dst_info, 1.f, 1.f, gemm_info);

    // telhs are newly created every call of this lambda function
    auto lhs = create_tensor<Tensor>(lhs_info);
    auto rhs = create_tensor<Tensor>(rhs_info);
    auto c   = create_tensor<Tensor>(c_info);
    lhs.allocator()->allocate();
    rhs.allocator()->allocate();
    c.allocator()->allocate();

    ITensorPack run_pack{{TensorType::ACL_SRC_0, &lhs}, {TensorType::ACL_SRC_1, &rhs}, {TensorType::ACL_SRC_2, &c}};
    ITensorPack prep_pack{{TensorType::ACL_SRC_1, &rhs}, {TensorType::ACL_SRC_2, &c}};

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(gemm->workspace(), mg, run_pack, prep_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(lhs), 1.f);
        library->fill_tensor_value(Accessor(rhs), 2.f);
        library->fill_tensor_value(Accessor(c), 3.f);
        // This operator is configured once and captured by this lambda.
        gemm->prepare(prep_pack);
        gemm->run(run_pack);
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

DATA_TEST_CASE(OpCpuGemmValidateAccumulate,
               framework::DatasetMode::ALL,
               combine(zip(make("In0", {TensorShape(21U, 13U)}),
                           make("In1", {TensorShape(33U, 21U)}),
                           make("Dst", {TensorShape(33U, 13U)})),
                       zip(make("alpha", {1.0, 100.0, 1.0, 1.0}),
                           make("beta", {0.0, 0.0, 1.0, 1.0}),
                           make("is_c_null", {false, false, false, true}),
                           make("Expected", {true, false, false, true}))),
               shape_a,
               shape_b,
               shape_dst,
               alpha,
               beta,
               is_c_null,
               expected)
{
    /* Accumulation test for GEMM kernels */
    // Create tensors
    TensorInfo in_a(shape_a, 1, DataType::F32);
    TensorInfo in_b(shape_b, 1, DataType::F32);
    TensorInfo in_c(shape_dst, 1, DataType::F32);
    TensorInfo dst(shape_dst, 1, DataType::F32);

    GEMMInfo gemm_info = GEMMInfo();
    gemm_info.set_accumulate(true);

    // Validate accumulation
    arm_compute::experimental::op::CpuGemm gemm;
    Status status = gemm.validate(&in_a, &in_b, (is_c_null ? nullptr : &in_c), &dst, alpha, beta, gemm_info);
    ARM_COMPUTE_EXPECT((expected == bool(status)), framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // CpuGemm
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
