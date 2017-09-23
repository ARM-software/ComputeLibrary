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
#ifndef ARM_COMPUTE_TEST_GOOGLENET_GEMM_DATASET
#define ARM_COMPUTE_TEST_GOOGLENET_GEMM_DATASET

#include "tests/datasets/GEMMDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GoogleNetGEMMDataset final : public GEMMDataset
{
public:
    GoogleNetGEMMDataset()
    {
        add_config(TensorShape{ 147U, 12544U }, TensorShape{ 64U, 147U }, TensorShape{ 12544U, 64U }, TensorShape{ 12544U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 192U, 784U }, TensorShape{ 96U, 192U }, TensorShape{ 784U, 96U }, TensorShape{ 784U, 96U }, 1.f, 0.f);
        add_config(TensorShape{ 192U, 784U }, TensorShape{ 32U, 192U }, TensorShape{ 784U, 32U }, TensorShape{ 784U, 32U }, 1.f, 0.f);
        add_config(TensorShape{ 256U, 784U }, TensorShape{ 32U, 256U }, TensorShape{ 784U, 32U }, TensorShape{ 784U, 32U }, 1.f, 0.f);
        add_config(TensorShape{ 480U, 196U }, TensorShape{ 96U, 480U }, TensorShape{ 196U, 96U }, TensorShape{ 196U, 96U }, 1.f, 0.f);
        add_config(TensorShape{ 480U, 196U }, TensorShape{ 64U, 480U }, TensorShape{ 196U, 64U }, TensorShape{ 196U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 508U, 196U }, TensorShape{ 24U, 508U }, TensorShape{ 196U, 24U }, TensorShape{ 196U, 24U }, 1.f, 0.f);
        add_config(TensorShape{ 512U, 196U }, TensorShape{ 128U, 512U }, TensorShape{ 196U, 128U }, TensorShape{ 196U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 512U, 196U }, TensorShape{ 64U, 512U }, TensorShape{ 196U, 64U }, TensorShape{ 196U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 512U, 196U }, TensorShape{ 32U, 512U }, TensorShape{ 196U, 32U }, TensorShape{ 196U, 32U }, 1.f, 0.f);
        add_config(TensorShape{ 528U, 196U }, TensorShape{ 160U, 528U }, TensorShape{ 196U, 160U }, TensorShape{ 196U, 160U }, 1.f, 0.f);
        add_config(TensorShape{ 528U, 196U }, TensorShape{ 128U, 528U }, TensorShape{ 196U, 128U }, TensorShape{ 196U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 832U, 49U }, TensorShape{ 48U, 832U }, TensorShape{ 49U, 48U }, TensorShape{ 49U, 48U }, 1.f, 0.f);
        add_config(TensorShape{ 832U, 49U }, TensorShape{ 192U, 832U }, TensorShape{ 49U, 192U }, TensorShape{ 49U, 192U }, 1.f, 0.f);
        add_config(TensorShape{ 832U, 49U }, TensorShape{ 128U, 832U }, TensorShape{ 49U, 128U }, TensorShape{ 49U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 528U, 16U }, TensorShape{ 128U, 528U }, TensorShape{ 16U, 128U }, TensorShape{ 16U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 64U, 3136U }, TensorShape{ 64U, 64U }, TensorShape{ 3136U, 64U }, TensorShape{ 3136U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 864U, 784U }, TensorShape{ 128U, 864U }, TensorShape{ 784U, 128U }, TensorShape{ 784U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 256U, 784U }, TensorShape{ 128U, 256U }, TensorShape{ 784U, 128U }, TensorShape{ 784U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 800U, 784U }, TensorShape{ 96U, 800U }, TensorShape{ 784U, 96U }, TensorShape{ 784U, 96U }, 1.f, 0.f);
        add_config(TensorShape{ 864U, 196U }, TensorShape{ 204U, 864U }, TensorShape{ 196U, 204U }, TensorShape{ 196U, 204U }, 1.f, 0.f);
        add_config(TensorShape{ 508U, 196U }, TensorShape{ 160U, 508U }, TensorShape{ 196U, 160U }, TensorShape{ 196U, 160U }, 1.f, 0.f);
        add_config(TensorShape{ 600U, 196U }, TensorShape{ 64U, 600U }, TensorShape{ 196U, 64U }, TensorShape{ 196U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 1152U, 196U }, TensorShape{ 256U, 1152U }, TensorShape{ 196U, 256U }, TensorShape{ 196U, 256U }, 1.f, 0.f);
        add_config(TensorShape{ 512U, 196U }, TensorShape{ 112U, 512U }, TensorShape{ 196U, 112U }, TensorShape{ 196U, 112U }, 1.f, 0.f);
        add_config(TensorShape{ 800U, 196U }, TensorShape{ 64U, 800U }, TensorShape{ 196U, 64U }, TensorShape{ 196U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 1440U, 196U }, TensorShape{ 320U, 1440U }, TensorShape{ 196U, 320U }, TensorShape{ 196U, 320U }, 1.f, 0.f);
        add_config(TensorShape{ 832U, 49U }, TensorShape{ 256U, 832U }, TensorShape{ 49U, 256U }, TensorShape{ 49U, 256U }, 1.f, 0.f);
        add_config(TensorShape{ 1200U, 49U }, TensorShape{ 128U, 1200U }, TensorShape{ 49U, 128U }, TensorShape{ 49U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 1728U, 49U }, TensorShape{ 384U, 1728U }, TensorShape{ 49U, 384U }, TensorShape{ 49U, 384U }, 1.f, 0.f);
        add_config(TensorShape{ 508U, 16U }, TensorShape{ 128U, 508U }, TensorShape{ 16U, 128U }, TensorShape{ 16U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 2048U, 1U }, TensorShape{ 1024U, 2048U }, TensorShape{ 1U, 1024U }, TensorShape{ 1U, 1024U }, 1.f, 0.f);
        add_config(TensorShape{ 576U, 3136U }, TensorShape{ 192U, 576U }, TensorShape{ 3136U, 192U }, TensorShape{ 3136U, 192U }, 1.f, 0.f);
        add_config(TensorShape{ 192U, 784U }, TensorShape{ 16U, 192U }, TensorShape{ 784U, 16U }, TensorShape{ 784U, 16U }, 1.f, 0.f);
        add_config(TensorShape{ 256U, 784U }, TensorShape{ 128U, 256U }, TensorShape{ 784U, 128U }, TensorShape{ 784U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 256U, 784U }, TensorShape{ 64U, 256U }, TensorShape{ 784U, 64U }, TensorShape{ 784U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 480U, 196U }, TensorShape{ 16U, 480U }, TensorShape{ 196U, 16U }, TensorShape{ 196U, 16U }, 1.f, 0.f);
        add_config(TensorShape{ 508U, 196U }, TensorShape{ 112U, 508U }, TensorShape{ 196U, 112U }, TensorShape{ 196U, 112U }, 1.f, 0.f);
        add_config(TensorShape{ 508U, 196U }, TensorShape{ 64U, 508U }, TensorShape{ 196U, 64U }, TensorShape{ 196U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 512U, 196U }, TensorShape{ 24U, 512U }, TensorShape{ 196U, 24U }, TensorShape{ 196U, 24U }, 1.f, 0.f);
        add_config(TensorShape{ 512U, 196U }, TensorShape{ 144U, 512U }, TensorShape{ 196U, 144U }, TensorShape{ 196U, 144U }, 1.f, 0.f);
        add_config(TensorShape{ 512U, 196U }, TensorShape{ 64U, 512U }, TensorShape{ 196U, 64U }, TensorShape{ 196U, 64U }, 1.f, 0.f);
        add_config(TensorShape{ 528U, 196U }, TensorShape{ 32U, 528U }, TensorShape{ 196U, 32U }, TensorShape{ 196U, 32U }, 1.f, 0.f);
        add_config(TensorShape{ 832U, 49U }, TensorShape{ 160U, 832U }, TensorShape{ 49U, 160U }, TensorShape{ 49U, 160U }, 1.f, 0.f);
        add_config(TensorShape{ 832U, 49U }, TensorShape{ 128U, 832U }, TensorShape{ 49U, 128U }, TensorShape{ 49U, 128U }, 1.f, 0.f);
        add_config(TensorShape{ 832U, 49U }, TensorShape{ 48U, 832U }, TensorShape{ 49U, 48U }, TensorShape{ 49U, 48U }, 1.f, 0.f);
        add_config(TensorShape{ 2048U, 1U }, TensorShape{ 1024U, 2048U }, TensorShape{ 1U, 1024U }, TensorShape{ 1U, 1024U }, 1.f, 0.f);
        add_config(TensorShape{ 1024U, 1U }, TensorShape{ 1008U, 1024U }, TensorShape{ 1U, 1008U }, TensorShape{ 1U, 1008U }, 1.f, 0.f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENET_GEMM_DATASET */
