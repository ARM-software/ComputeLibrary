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
#ifndef __ARM_COMPUTE_TEST_BENCHMARK_DATASETS_H__
#define __ARM_COMPUTE_TEST_BENCHMARK_DATASETS_H__

#include "dataset/ActivationLayerDataset.h"
#include "dataset/BorderModeDataset.h"
#include "dataset/ConvolutionLayerDataset.h"
#include "dataset/DataTypeDatasets.h"
#include "dataset/FullyConnectedLayerDataset.h"
#include "dataset/GEMMDataset.h"
#include "dataset/ImageDatasets.h"
#include "dataset/InterpolationPolicyDataset.h"
#include "dataset/NormalizationLayerDataset.h"
#include "dataset/PoolingLayerDataset.h"
#include "dataset/ShapeDatasets.h"

#include "benchmark/benchmark.h"

#include <array>

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename DataSet, int N>
void DataSetArg(::benchmark::internal::Benchmark *b)
{
    b->Arg(N);
    b->ArgName(std::string(*(DataSet().begin() + N)));
}

template <typename DataSet, int N, unsigned int... Args>
void DataSetArgBatched(::benchmark::internal::Benchmark *b)
{
    constexpr std::array<unsigned int, sizeof...(Args)> batches{ { Args... } };
    for(const auto &el : batches)
    {
        b->Args({ N, static_cast<int>(el) });
    }
    b->ArgNames({ std::string(*(DataSet().begin() + N)), "batch_size" });
}

template <typename DataSet>
void DataSetArgs(::benchmark::internal::Benchmark *b)
{
    for(size_t i = 0; i < DataSet().size(); ++i)
    {
        b->Arg(i);
        b->ArgName(*(DataSet().begin() + i));
    }
}
}
}
}
#endif
