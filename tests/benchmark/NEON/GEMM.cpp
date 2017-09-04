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
#include "NEON/Helper.h"
#include "NEON/NEAccessor.h"
#include "TensorLibrary.h"
#include "benchmark/Datasets.h"
#include "benchmark/Profiler.h"
#include "benchmark/WallClockTimer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "benchmark/benchmark_api.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::neon;

#include "benchmark/NEON/GEMM.h"

namespace
{
#ifdef ENABLE_FP16
using GEMMFP16GoogLeNet1 = GEMM<GoogLeNetGEMMDataset1, Tensor, NEAccessor, NEGEMM, DataType::F16>;
using GEMMFP16GoogLeNet2 = GEMM<GoogLeNetGEMMDataset2, Tensor, NEAccessor, NEGEMM, DataType::F16>;
#endif /* ENABLE_FP16 */
using GEMMFP32GoogLeNet1 = GEMM<GoogLeNetGEMMDataset1, Tensor, NEAccessor, NEGEMM, DataType::F32>;
using GEMMFP32GoogLeNet2 = GEMM<GoogLeNetGEMMDataset2, Tensor, NEAccessor, NEGEMM, DataType::F32>;
using GEMMQS8GoogLeNet1  = GEMM<GoogLeNetGEMMDataset1, Tensor, NEAccessor, NEGEMM, DataType::QS8>;
using GEMMQS8GoogLeNet2  = GEMM<GoogLeNetGEMMDataset2, Tensor, NEAccessor, NEGEMM, DataType::QS8>;
} // namespace
#ifdef ENABLE_FP16
BENCHMARK_DEFINE_F(GEMMFP16GoogLeNet1, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        gemm_layer->run();
        profiler.stop();
    }
}

BENCHMARK_DEFINE_F(GEMMFP16GoogLeNet2, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        gemm_layer->run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 0>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 1>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 2>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 3>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 4>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 5>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 6>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 7>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 8>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 9>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 10>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 11>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 12>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 13>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 14>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 15>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 16>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 17>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 18>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 19>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 20>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 21>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 22>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 23>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 24>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 25>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 26>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 27>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 28>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 29>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 30>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 31>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 0>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 1>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 2>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 3>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 4>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 5>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 6>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 7>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 8>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 9>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 10>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 11>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 12>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 13>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 14>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 15>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 16>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 17>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 18>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 19>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 20>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 21>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 22>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 23>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 24>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 25>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 26>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 27>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 28>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 29>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 30>);
BENCHMARK_REGISTER_F(GEMMFP16GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 31>);
#endif /* ENABLE_FP16 */

BENCHMARK_DEFINE_F(GEMMFP32GoogLeNet1, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        gemm_layer->run();
        profiler.stop();
    }
}

BENCHMARK_DEFINE_F(GEMMFP32GoogLeNet2, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        gemm_layer->run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 0>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 1>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 2>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 3>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 4>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 5>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 6>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 7>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 8>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 9>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 10>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 11>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 12>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 13>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 14>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 15>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 16>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 17>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 18>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 19>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 20>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 21>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 22>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 23>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 24>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 25>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 26>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 27>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 28>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 29>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 30>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 31>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 0>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 1>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 2>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 3>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 4>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 5>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 6>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 7>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 8>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 9>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 10>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 11>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 12>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 13>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 14>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 15>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 16>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 17>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 18>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 19>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 20>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 21>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 22>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 23>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 24>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 25>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 26>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 27>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 28>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 29>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 30>);
BENCHMARK_REGISTER_F(GEMMFP32GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 31>);

BENCHMARK_DEFINE_F(GEMMQS8GoogLeNet1, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        gemm_layer->run();
        profiler.stop();
    }
}

BENCHMARK_DEFINE_F(GEMMQS8GoogLeNet2, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        gemm_layer->run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 0>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 1>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 2>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 3>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 4>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 5>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 6>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 7>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 8>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 9>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 10>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 11>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 12>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 13>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 14>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 15>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 16>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 17>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 18>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 19>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 20>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 21>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 22>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 23>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 24>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 25>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 26>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 27>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 28>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 29>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 30>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet1, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset1, 31>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 0>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 1>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 2>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 3>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 4>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 5>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 6>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 7>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 8>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 9>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 10>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 11>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 12>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 13>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 14>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 15>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 16>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 17>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 18>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 19>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 20>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 21>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 22>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 23>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 24>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 25>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 26>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 27>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 28>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 29>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 30>);
BENCHMARK_REGISTER_F(GEMMQS8GoogLeNet2, neon_googlenet)
->Threads(1)
->Apply(DataSetArg<GoogLeNetGEMMDataset2, 31>);
