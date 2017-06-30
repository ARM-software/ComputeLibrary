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
#include "CL/CLAccessor.h"
#include "Globals.h"
#include "TensorLibrary.h"
#include "benchmark/Datasets.h"
#include "benchmark/Profiler.h"
#include "benchmark/WallClockTimer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLBitwiseAnd.h"

#include "benchmark/benchmark_api.h"

#include <memory>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::cl;

namespace
{
template <typename DataSet>
class BitwiseAnd : public ::benchmark::Fixture
{
public:
    void SetUp(::benchmark::State &state) override
    {
        ::benchmark::Fixture::SetUp(state);

        profiler.add(std::make_shared<WallClockTimer>());

        const std::string image_name = *(DataSet().begin() + state.range(0));
        const RawTensor &raw        = library->get(image_name);

        // Create tensors
        src1 = create_tensor<CLTensor>(raw.shape(), DataType::U8);
        src2 = create_tensor<CLTensor>(raw.shape(), DataType::U8);
        dst  = create_tensor<CLTensor>(raw.shape(), DataType::U8);

        // Create and configure function
        band.configure(&src1, &src2, &dst);

        // Allocate tensors
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill source tensors
        library->fill(CLAccessor(src1), image_name, Channel::R);
        library->fill(CLAccessor(src2), image_name, Channel::G);
    }

    void TearDown(::benchmark::State &state) override
    {
        profiler.submit(state);

        ::benchmark::Fixture::TearDown(state);
    }

    CLBitwiseAnd band{};
    Profiler     profiler{};

private:
    CLTensor src1{};
    CLTensor src2{};
    CLTensor dst{};
};

using BitwiseAndSmall = BitwiseAnd<SmallImages>;
using BitwiseAndLarge = BitwiseAnd<LargeImages>;
} // namespace

BENCHMARK_DEFINE_F(BitwiseAndSmall, cl_bitwise_and)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        band.run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(BitwiseAndSmall, cl_bitwise_and)
->Threads(1)
->Apply(DataSetArgs<SmallImages>);

BENCHMARK_DEFINE_F(BitwiseAndLarge, cl_bitwise_and)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        band.run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(BitwiseAndLarge, cl_bitwise_and)
->Threads(1)
->Apply(DataSetArgs<LargeImages>);
