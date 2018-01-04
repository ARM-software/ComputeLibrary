/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_CONVOLUTION_LAYER_DATASET

#include "tests/datasets/ConvolutionLayerDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GoogLeNetInceptionV1WinogradLayerDataset final : public ConvolutionLayerDataset
{
public:
    // GoogLeNetInceptionV1 inception v1 dataset
    GoogLeNetInceptionV1WinogradLayerDataset()
    {
        add_config(TensorShape(56U, 56U, 64U), TensorShape(3U, 3U, 64U, 192U), TensorShape(192U), TensorShape(56U, 56U, 192U), PadStrideInfo(1, 1, 1, 1));
        // inception_3a/3x3
        add_config(TensorShape(28U, 28U, 96U), TensorShape(3U, 3U, 96U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 1, 1));
        // inception_3b/3x3
        add_config(TensorShape(28U, 28U, 128U), TensorShape(3U, 3U, 128U, 192U), TensorShape(192U), TensorShape(28U, 28U, 192U), PadStrideInfo(1, 1, 1, 1));
        // inception_4a/3x3
        add_config(TensorShape(14U, 14U, 96U), TensorShape(3U, 3U, 96U, 208U), TensorShape(208U), TensorShape(14U, 14U, 208U), PadStrideInfo(1, 1, 1, 1));
        // inception_4b/3x3
        add_config(TensorShape(14U, 14U, 112U), TensorShape(3U, 3U, 112U, 224U), TensorShape(224U), TensorShape(14U, 14U, 224U), PadStrideInfo(1, 1, 1, 1));
        // inception_4c/3x3
        add_config(TensorShape(14U, 14U, 128U), TensorShape(3U, 3U, 128U, 256U), TensorShape(256U), TensorShape(14U, 14U, 256U), PadStrideInfo(1, 1, 1, 1));
        // inception_4d/3x3
        add_config(TensorShape(14U, 14U, 144U), TensorShape(3U, 3U, 144U, 288U), TensorShape(288U), TensorShape(14U, 14U, 288U), PadStrideInfo(1, 1, 1, 1));
        // inception_4e/3x3
        add_config(TensorShape(14U, 14U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(14U, 14U, 320U), PadStrideInfo(1, 1, 1, 1));
        // inception_5a/3x3
        add_config(TensorShape(7U, 7U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(7U, 7U, 320U), PadStrideInfo(1, 1, 1, 1));
        // inception_5b/3x3
        add_config(TensorShape(7U, 7U, 192U), TensorShape(3U, 3U, 192U, 384U), TensorShape(384U), TensorShape(7U, 7U, 384U), PadStrideInfo(1, 1, 1, 1));
    }
};

class GoogLeNetInceptionV1ConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    // GoogLeNetInceptionV1 inception v1 dataset
    GoogLeNetInceptionV1ConvolutionLayerDataset()
    {
        // conv1/7x7_s2
        add_config(TensorShape(224U, 224U, 3U), TensorShape(7U, 7U, 3U, 64U), TensorShape(64U), TensorShape(112U, 112U, 64U), PadStrideInfo(2, 2, 3, 3));
        // conv2/3x3_reduce
        add_config(TensorShape(56U, 56U, 64U), TensorShape(1U, 1U, 64U, 64U), TensorShape(64U), TensorShape(56U, 56U, 64U), PadStrideInfo(1, 1, 0, 0));
        // conv2/3x3
        add_config(TensorShape(56U, 56U, 64U), TensorShape(3U, 3U, 64U, 192U), TensorShape(192U), TensorShape(56U, 56U, 192U), PadStrideInfo(1, 1, 1, 1));
        // inception_3a/1x1
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_3a/3x3_reduce
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 96U), TensorShape(96U), TensorShape(28U, 28U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_3a/3x3
        add_config(TensorShape(28U, 28U, 96U), TensorShape(3U, 3U, 96U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 1, 1));
        // inception_3a/5x5_reduce
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 16U), TensorShape(16U), TensorShape(28U, 28U, 16U), PadStrideInfo(1, 1, 0, 0));
        // inception_3a/5x5
        add_config(TensorShape(28U, 28U, 16U), TensorShape(5U, 5U, 16U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 2, 2));
        // inception_3a/pool_proj
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_3b/1x1, inception_3b/3x3_reduce
        add_config(TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_3b/3x3
        add_config(TensorShape(28U, 28U, 128U), TensorShape(3U, 3U, 128U, 192U), TensorShape(192U), TensorShape(28U, 28U, 192U), PadStrideInfo(1, 1, 1, 1));
        // inception_3b/5x5_reduce
        add_config(TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_3b/5x5
        add_config(TensorShape(28U, 28U, 32U), TensorShape(5U, 5U, 32U, 96U), TensorShape(96U), TensorShape(28U, 28U, 96U), PadStrideInfo(1, 1, 2, 2));
        // inception_3b/pool_proj
        add_config(TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/1x1
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 192U), TensorShape(192U), TensorShape(14U, 14U, 192U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/3x3_reduce
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 96U), TensorShape(96U), TensorShape(14U, 14U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/3x3
        add_config(TensorShape(14U, 14U, 96U), TensorShape(3U, 3U, 96U, 208U), TensorShape(208U), TensorShape(14U, 14U, 208U), PadStrideInfo(1, 1, 1, 1));
        // inception_4a/5x5_reduce
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 16U), TensorShape(16U), TensorShape(14U, 14U, 16U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/5x5
        add_config(TensorShape(14U, 14U, 16U), TensorShape(5U, 5U, 16U, 48U), TensorShape(48U), TensorShape(14U, 14U, 48U), PadStrideInfo(1, 1, 2, 2));
        // inception_4a/pool_proj
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/1x1
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 160U), TensorShape(160U), TensorShape(14U, 14U, 160U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/3x3_reduce, inception_4d/1x1
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 112U), TensorShape(112U), TensorShape(14U, 14U, 112U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/3x3
        add_config(TensorShape(14U, 14U, 112U), TensorShape(3U, 3U, 112U, 224U), TensorShape(224U), TensorShape(14U, 14U, 224U), PadStrideInfo(1, 1, 1, 1));
        // inception_4b/5x5_reduce, inception_4c/5x5_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 24U), TensorShape(24U), TensorShape(14U, 14U, 24U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/5x5, inception_4c/5x5
        add_config(TensorShape(14U, 14U, 24U), TensorShape(5U, 5U, 24U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 2, 2));
        // inception_4b/pool_proj, inception_4c/pool_proj, inception_4d/pool_proj
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_4c/1x1, inception_4c/3x3_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_4c/3x3
        add_config(TensorShape(14U, 14U, 128U), TensorShape(3U, 3U, 128U, 256U), TensorShape(256U), TensorShape(14U, 14U, 256U), PadStrideInfo(1, 1, 1, 1));
        // inception_4d/3x3_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 144U), TensorShape(144U), TensorShape(14U, 14U, 144U), PadStrideInfo(1, 1, 0, 0));
        // inception_4d/3x3
        add_config(TensorShape(14U, 14U, 144U), TensorShape(3U, 3U, 144U, 288U), TensorShape(288U), TensorShape(14U, 14U, 288U), PadStrideInfo(1, 1, 1, 1));
        // inception_4d/5x5_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 32U), TensorShape(32U), TensorShape(14U, 14U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_4d/5x5
        add_config(TensorShape(14U, 14U, 32U), TensorShape(5U, 5U, 32U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 2, 2));
        // inception_4e/1x1
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 256U), TensorShape(256U), TensorShape(14U, 14U, 256U), PadStrideInfo(1, 1, 0, 0));
        // inception_4e/3x3_reduce
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 160U), TensorShape(160U), TensorShape(14U, 14U, 160U), PadStrideInfo(1, 1, 0, 0));
        // inception_4e/3x3
        add_config(TensorShape(14U, 14U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(14U, 14U, 320U), PadStrideInfo(1, 1, 1, 1));
        // inception_4e/5x5_reduce
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 32U), TensorShape(32U), TensorShape(14U, 14U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_4e/5x5
        add_config(TensorShape(14U, 14U, 32U), TensorShape(5U, 5U, 32U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 2, 2));
        // inception_4e/pool_proj
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/1x1
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 256U), TensorShape(256U), TensorShape(7U, 7U, 256U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/3x3_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 160U), TensorShape(160U), TensorShape(7U, 7U, 160U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/3x3
        add_config(TensorShape(7U, 7U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(7U, 7U, 320U), PadStrideInfo(1, 1, 1, 1));
        // inception_5a/5x5_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 32U), TensorShape(32U), TensorShape(7U, 7U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/5x5
        add_config(TensorShape(7U, 7U, 32U), TensorShape(5U, 5U, 32U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 2, 2));
        // inception_5a/pool_proj, inception_5b/pool_proj
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/1x1
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 384U), TensorShape(384U), TensorShape(7U, 7U, 384U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/3x3_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 192U), TensorShape(192U), TensorShape(7U, 7U, 192U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/3x3
        add_config(TensorShape(7U, 7U, 192U), TensorShape(3U, 3U, 192U, 384U), TensorShape(384U), TensorShape(7U, 7U, 384U), PadStrideInfo(1, 1, 1, 1));
        // inception_5b/5x5_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 48U), TensorShape(48U), TensorShape(7U, 7U, 48U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/5x5
        add_config(TensorShape(7U, 7U, 48U), TensorShape(5U, 5U, 48U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 2, 2));
    }
};

class GoogLeNetInceptionV1DirectConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    // subset of GoogLeNetInceptionV1 inception v1 dataset
    GoogLeNetInceptionV1DirectConvolutionLayerDataset()
    {
        // conv2/3x3_reduce
        add_config(TensorShape(56U, 56U, 64U), TensorShape(1U, 1U, 64U, 64U), TensorShape(64U), TensorShape(56U, 56U, 64U), PadStrideInfo(1, 1, 0, 0));
        // conv2/3x3
        add_config(TensorShape(56U, 56U, 64U), TensorShape(3U, 3U, 64U, 192U), TensorShape(192U), TensorShape(56U, 56U, 192U), PadStrideInfo(1, 1, 1, 1));
        // inception_3a/1x1
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_3a/3x3_reduce
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 96U), TensorShape(96U), TensorShape(28U, 28U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_3a/3x3
        add_config(TensorShape(28U, 28U, 96U), TensorShape(3U, 3U, 96U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 1, 1));
        // inception_3a/5x5_reduce
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 16U), TensorShape(16U), TensorShape(28U, 28U, 16U), PadStrideInfo(1, 1, 0, 0));
        // inception_3a/5x5
        add_config(TensorShape(28U, 28U, 16U), TensorShape(5U, 5U, 16U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 2, 2));
        // inception_3a/pool_proj
        add_config(TensorShape(28U, 28U, 192U), TensorShape(1U, 1U, 192U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_3b/1x1, inception_3b/3x3_reduce
        add_config(TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 128U), TensorShape(128U), TensorShape(28U, 28U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_3b/3x3
        add_config(TensorShape(28U, 28U, 128U), TensorShape(3U, 3U, 128U, 192U), TensorShape(192U), TensorShape(28U, 28U, 192U), PadStrideInfo(1, 1, 1, 1));
        // inception_3b/5x5_reduce
        add_config(TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 32U), TensorShape(32U), TensorShape(28U, 28U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_3b/5x5
        add_config(TensorShape(28U, 28U, 32U), TensorShape(5U, 5U, 32U, 96U), TensorShape(96U), TensorShape(28U, 28U, 96U), PadStrideInfo(1, 1, 2, 2));
        // inception_3b/pool_proj
        add_config(TensorShape(28U, 28U, 256U), TensorShape(1U, 1U, 256U, 64U), TensorShape(64U), TensorShape(28U, 28U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/1x1
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 192U), TensorShape(192U), TensorShape(14U, 14U, 192U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/3x3_reduce
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 96U), TensorShape(96U), TensorShape(14U, 14U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/3x3
        add_config(TensorShape(14U, 14U, 96U), TensorShape(3U, 3U, 96U, 208U), TensorShape(208U), TensorShape(14U, 14U, 208U), PadStrideInfo(1, 1, 1, 1));
        // inception_4a/5x5_reduce
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 16U), TensorShape(16U), TensorShape(14U, 14U, 16U), PadStrideInfo(1, 1, 0, 0));
        // inception_4a/5x5
        add_config(TensorShape(14U, 14U, 16U), TensorShape(5U, 5U, 16U, 48U), TensorShape(48U), TensorShape(14U, 14U, 48U), PadStrideInfo(1, 1, 2, 2));
        // inception_4a/pool_proj
        add_config(TensorShape(14U, 14U, 480U), TensorShape(1U, 1U, 480U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/1x1
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 160U), TensorShape(160U), TensorShape(14U, 14U, 160U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/3x3_reduce, inception_4d/1x1
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 112U), TensorShape(112U), TensorShape(14U, 14U, 112U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/3x3
        add_config(TensorShape(14U, 14U, 112U), TensorShape(3U, 3U, 112U, 224U), TensorShape(224U), TensorShape(14U, 14U, 224U), PadStrideInfo(1, 1, 1, 1));
        // inception_4b/5x5_reduce, inception_4c/5x5_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 24U), TensorShape(24U), TensorShape(14U, 14U, 24U), PadStrideInfo(1, 1, 0, 0));
        // inception_4b/5x5, inception_4c/5x5
        add_config(TensorShape(14U, 14U, 24U), TensorShape(5U, 5U, 24U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 2, 2));
        // inception_4b/pool_proj, inception_4c/pool_proj, inception_4d/pool_proj
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_4c/1x1, inception_4c/3x3_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_4c/3x3
        add_config(TensorShape(14U, 14U, 128U), TensorShape(3U, 3U, 128U, 256U), TensorShape(256U), TensorShape(14U, 14U, 256U), PadStrideInfo(1, 1, 1, 1));
        // inception_4d/3x3_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 144U), TensorShape(144U), TensorShape(14U, 14U, 144U), PadStrideInfo(1, 1, 0, 0));
        // inception_4d/3x3
        add_config(TensorShape(14U, 14U, 144U), TensorShape(3U, 3U, 144U, 288U), TensorShape(288U), TensorShape(14U, 14U, 288U), PadStrideInfo(1, 1, 1, 1));
        // inception_4d/5x5_reduce
        add_config(TensorShape(14U, 14U, 512U), TensorShape(1U, 1U, 512U, 32U), TensorShape(32U), TensorShape(14U, 14U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_4d/5x5
        add_config(TensorShape(14U, 14U, 32U), TensorShape(5U, 5U, 32U, 64U), TensorShape(64U), TensorShape(14U, 14U, 64U), PadStrideInfo(1, 1, 2, 2));
        // inception_4e/1x1
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 256U), TensorShape(256U), TensorShape(14U, 14U, 256U), PadStrideInfo(1, 1, 0, 0));
        // inception_4e/3x3_reduce
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 160U), TensorShape(160U), TensorShape(14U, 14U, 160U), PadStrideInfo(1, 1, 0, 0));
        // inception_4e/3x3
        add_config(TensorShape(14U, 14U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(14U, 14U, 320U), PadStrideInfo(1, 1, 1, 1));
        // inception_4e/5x5_reduce
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 32U), TensorShape(32U), TensorShape(14U, 14U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_4e/5x5
        add_config(TensorShape(14U, 14U, 32U), TensorShape(5U, 5U, 32U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 2, 2));
        // inception_4e/pool_proj
        add_config(TensorShape(14U, 14U, 528U), TensorShape(1U, 1U, 528U, 128U), TensorShape(128U), TensorShape(14U, 14U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/1x1
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 256U), TensorShape(256U), TensorShape(7U, 7U, 256U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/3x3_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 160U), TensorShape(160U), TensorShape(7U, 7U, 160U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/3x3
        add_config(TensorShape(7U, 7U, 160U), TensorShape(3U, 3U, 160U, 320U), TensorShape(320U), TensorShape(7U, 7U, 320U), PadStrideInfo(1, 1, 1, 1));
        // inception_5a/5x5_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 32U), TensorShape(32U), TensorShape(7U, 7U, 32U), PadStrideInfo(1, 1, 0, 0));
        // inception_5a/5x5
        add_config(TensorShape(7U, 7U, 32U), TensorShape(5U, 5U, 32U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 2, 2));
        // inception_5a/pool_proj, inception_5b/pool_proj
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/1x1
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 384U), TensorShape(384U), TensorShape(7U, 7U, 384U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/3x3_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 192U), TensorShape(192U), TensorShape(7U, 7U, 192U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/3x3
        add_config(TensorShape(7U, 7U, 192U), TensorShape(3U, 3U, 192U, 384U), TensorShape(384U), TensorShape(7U, 7U, 384U), PadStrideInfo(1, 1, 1, 1));
        // inception_5b/5x5_reduce
        add_config(TensorShape(7U, 7U, 832U), TensorShape(1U, 1U, 832U, 48U), TensorShape(48U), TensorShape(7U, 7U, 48U), PadStrideInfo(1, 1, 0, 0));
        // inception_5b/5x5
        add_config(TensorShape(7U, 7U, 48U), TensorShape(5U, 5U, 48U, 128U), TensorShape(128U), TensorShape(7U, 7U, 128U), PadStrideInfo(1, 1, 2, 2));
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV1_CONVOLUTION_LAYER_DATASET */
