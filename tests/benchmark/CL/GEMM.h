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
#ifndef __ARM_COMPUTE_TEST_BENCHMARK_CL_GEMM_H__
#define __ARM_COMPUTE_TEST_BENCHMARK_CL_GEMM_H__

#include "TensorLibrary.h"
#include "Utils.h"
#include "dataset/GEMMDataset.h"

#include <memory>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename DataSet, typename TensorType, typename Accessor, typename Function, DataType data_type>
class GEMM : public ::benchmark::Fixture
{
public:
    void SetUp(::benchmark::State &state) override
    {
        ARM_COMPUTE_ERROR_ON_MSG(data_type != DataType::F16 && data_type != DataType::F32, "Unsupported data type for GEMM operation");

        profiler.add(std::make_shared<WallClockTimer>());

        const GEMMDataObject gemm_obj = *(DataSet().begin() + state.range(0));

        TensorShape shape_a = gemm_obj.shape_a;
        TensorShape shape_b = gemm_obj.shape_b;
        TensorShape shape_c = gemm_obj.shape_c;
        TensorShape shape_d = gemm_obj.shape_d;

        // Create tensors
        a = create_tensor(shape_a, data_type);
        b = create_tensor(shape_b, data_type);
        c = create_tensor(shape_c, data_type);
        d = create_tensor(shape_d, data_type);

        // Create and configure function
        gemm_layer = std::unique_ptr<Function>(new Function());
        gemm_layer->configure(&a, &b, &c, &d, gemm_obj.alpha, gemm_obj.beta);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        c.allocator()->allocate();
        d.allocator()->allocate();
    }

    void TearDown(::benchmark::State &state) override
    {
        gemm_layer.reset();

        a.allocator()->free();
        b.allocator()->free();
        c.allocator()->free();
        d.allocator()->free();

        profiler.submit(state);
    }

    std::unique_ptr<Function> gemm_layer{ nullptr };
    Profiler                  profiler{};

private:
    TensorType a{};
    TensorType b{};
    TensorType c{};
    TensorType d{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_BENCHMARK_CL_GEMM_H__
