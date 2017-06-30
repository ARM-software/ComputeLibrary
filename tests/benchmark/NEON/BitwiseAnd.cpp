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
#include "Globals.h"
#include "NEON/NEAccessor.h"
#include "TensorLibrary.h"
#include "benchmark/Datasets.h"
#include "benchmark/Profiler.h"
#include "benchmark/WallClockTimer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseAnd.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "benchmark/benchmark_api.h"

#include <memory>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::neon;

namespace
{
template <typename DataSet>
class BitwiseAnd : public ::benchmark::Fixture
{
public:
    void SetUp(::benchmark::State &state) override
    {
        profiler.add(std::make_shared<WallClockTimer>());

        const std::string image_name = *(DataSet().begin() + state.range(0));
        const RawTensor &raw        = library->get(image_name);

        // Create tensors
        src1 = create_tensor<Tensor>(raw.shape(), DataType::U8);
        src2 = create_tensor<Tensor>(raw.shape(), DataType::U8);
        dst  = create_tensor<Tensor>(raw.shape(), DataType::U8);

        // Create and configure function
        band.configure(&src1, &src2, &dst);

        // Allocate tensors
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill source tensors
        library->fill(NEAccessor(src1), image_name, Channel::R);
        library->fill(NEAccessor(src2), image_name, Channel::G);
    }

    void TearDown(::benchmark::State &state) override
    {
        profiler.submit(state);
    }

    NEBitwiseAnd band{};
    Profiler     profiler{};

private:
    Tensor src1{};
    Tensor src2{};
    Tensor dst{};
};

using BitwiseAndSmall = BitwiseAnd<SmallImages>;
using BitwiseAndLarge = BitwiseAnd<LargeImages>;
} // namespace

BENCHMARK_DEFINE_F(BitwiseAndSmall, neon_bitwise_and)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        band.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(BitwiseAndSmall, neon_bitwise_and)
->Threads(1)
->Apply(DataSetArgs<SmallImages>);

BENCHMARK_DEFINE_F(BitwiseAndLarge, neon_bitwise_and)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        band.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(BitwiseAndLarge, neon_bitwise_and)
->Threads(1)
->Apply(DataSetArgs<LargeImages>);
