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
#ifndef ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_CONVOLUTION_LAYER_DATASET
#define ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_CONVOLUTION_LAYER_DATASET

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
class GoogLeNetInceptionV4WinogradLayerDataset final : public ConvolutionLayerDataset
{
public:
    // GoogLeNetInceptionV4 inception v1 dataset
    GoogLeNetInceptionV4WinogradLayerDataset()
    {
        // conv2_3x3_s1
        add_config(TensorShape(149U, 149U, 32U), TensorShape(3U, 3U, 32U, 32U), TensorShape(32U), TensorShape(147U, 147U, 32U), PadStrideInfo(1, 1, 0, 0));
        // conv3_3x3_s1
        add_config(TensorShape(147U, 147U, 32U), TensorShape(3U, 3U, 32U, 64U), TensorShape(64U), TensorShape(147U, 147U, 64U), PadStrideInfo(1, 1, 1, 1));
        // inception_stem2_3x3, inception_stem2_3x3_2
        add_config(TensorShape(73U, 73U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(71U, 71U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_a1_3x3, inception_a1_3x3_2, inception_a2_3x3, inception_a2_3x3_2, inception_a3_3x3, inception_a3_3x3_2, inception_a4_3x3, inception_a4_3x3_2
        add_config(TensorShape(35U, 35U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 1, 1));
        // inception_a1_3x3_3, inception_a2_3x3_3, inception_a3_3x3_3, inception_a4_3x3_3
        add_config(TensorShape(35U, 35U, 96U), TensorShape(3U, 3U, 96U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 1, 1));
        // reduction_a_3x3_2
        add_config(TensorShape(35U, 35U, 192U), TensorShape(3U, 3U, 192U, 224U), TensorShape(224U), TensorShape(35U, 35U, 224U), PadStrideInfo(1, 1, 1, 1));
    }
};

class GoogLeNetInceptionV4ConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    // GoogLeNetInceptionV4 inception v1 dataset
    GoogLeNetInceptionV4ConvolutionLayerDataset()
    {
        // conv1_3x3_s2
        add_config(TensorShape(299U, 299U, 3U), TensorShape(3U, 3U, 3U, 32U), TensorShape(32U), TensorShape(149U, 149U, 32U), PadStrideInfo(2, 2, 0, 0));
        // conv2_3x3_s1
        add_config(TensorShape(149U, 149U, 32U), TensorShape(3U, 3U, 32U, 32U), TensorShape(32U), TensorShape(147U, 147U, 32U), PadStrideInfo(1, 1, 0, 0));
        // conv3_3x3_s1
        add_config(TensorShape(147U, 147U, 32U), TensorShape(3U, 3U, 32U, 64U), TensorShape(64U), TensorShape(147U, 147U, 64U), PadStrideInfo(1, 1, 1, 1));
        // inception_stem1_3x3_s2
        add_config(TensorShape(147U, 147U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(73U, 73U, 96U), PadStrideInfo(2, 2, 0, 0));
        // inception_stem2_3x3_reduce, inception_stem2_1x7_reduce
        add_config(TensorShape(73U, 73U, 160U), TensorShape(1U, 1U, 160U, 64U), TensorShape(64U), TensorShape(73U, 73U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_stem2_3x3, inception_stem2_3x3_2
        add_config(TensorShape(73U, 73U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(71U, 71U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_stem2_1x7
        add_config(TensorShape(73U, 73U, 64U), TensorShape(7U, 1U, 64U, 64U), TensorShape(64U), TensorShape(73U, 73U, 64U), PadStrideInfo(1, 1, 3, 0));
        // inception_stem2_7x1
        add_config(TensorShape(73U, 73U, 64U), TensorShape(1U, 7U, 64U, 64U), TensorShape(64U), TensorShape(73U, 73U, 64U), PadStrideInfo(1, 1, 0, 3));
        // inception_stem3_3x3_s2
        add_config(TensorShape(71U, 71U, 192U), TensorShape(3U, 3U, 192U, 192U), TensorShape(192U), TensorShape(35U, 35U, 192U), PadStrideInfo(2, 2, 0, 0));
        // inception_a1_1x1_2, inception_a1_1x1, inception_a2_1x1_2, inception_a2_1x1, inception_a3_1x1_2, inception_a3_1x1, inception_a4_1x1_2, inception_a4_1x1
        add_config(TensorShape(35U, 35U, 384U), TensorShape(1U, 1U, 384U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_a1_3x3_reduce, inception_a1_3x3_2_reduce, inception_a2_3x3_reduce, inception_a2_3x3_2_reduce, inception_a3_3x3_reduce, inception_a3_3x3_2_reduce, inception_a4_3x3_reduce, inception_a4_3x3_2_reduce
        add_config(TensorShape(35U, 35U, 384U), TensorShape(1U, 1U, 384U, 64U), TensorShape(64U), TensorShape(35U, 35U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_a1_3x3, inception_a1_3x3_2, inception_a2_3x3, inception_a2_3x3_2, inception_a3_3x3, inception_a3_3x3_2, inception_a4_3x3, inception_a4_3x3_2
        add_config(TensorShape(35U, 35U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 1, 1));
        // inception_a1_3x3_3, inception_a2_3x3_3, inception_a3_3x3_3, inception_a4_3x3_3
        add_config(TensorShape(35U, 35U, 96U), TensorShape(3U, 3U, 96U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 1, 1));
        // reduction_a_3x3
        add_config(TensorShape(35U, 35U, 384U), TensorShape(3U, 3U, 384U, 384U), TensorShape(384U), TensorShape(17U, 17U, 384U), PadStrideInfo(2, 2, 0, 0));
        // reduction_a_3x3_2_reduce
        add_config(TensorShape(35U, 35U, 384U), TensorShape(1U, 1U, 384U, 192U), TensorShape(192U), TensorShape(35U, 35U, 192U), PadStrideInfo(1, 1, 0, 0));
        // reduction_a_3x3_2
        add_config(TensorShape(35U, 35U, 192U), TensorShape(3U, 3U, 192U, 224U), TensorShape(224U), TensorShape(35U, 35U, 224U), PadStrideInfo(1, 1, 1, 1));
        // reduction_a_3x3_3
        add_config(TensorShape(35U, 35U, 224U), TensorShape(3U, 3U, 224U, 256U), TensorShape(256U), TensorShape(17U, 17U, 256U), PadStrideInfo(2, 2, 0, 0));
        // inception_b1_1x1_2, inception_b2_1x1_2, inception_b3_1x1_2, inception_b4_1x1_2, inception_b5_1x1_2, inception_b6_1x1_2, inception_b7_1x1_2
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 384U), TensorShape(384U), TensorShape(17U, 17U, 384U), PadStrideInfo(1, 1, 0, 0));
        // inception_b1_1x7_reduce, inception_b1_7x1_2_reduce, inception_b2_1x7_reduce, inception_b2_7x1_2_reduce, inception_b3_1x7_reduce, inception_b3_7x1_2_reduce, inception_b4_1x7_reduce, inception_b4_7x1_2_reduce, inception_b5_1x7_reduce, inception_b5_7x1_2_reduce, inception_b6_1x7_reduce, inception_b6_7x1_2_reduce, inception_b7_1x7_reduce, inception_b7_7x1_2_reduce, reduction_b_3x3_reduce
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 192U), TensorShape(192U), TensorShape(17U, 17U, 192U), PadStrideInfo(1, 1, 0, 0));
        // inception_b1_1x7, inception_b1_1x7_2, inception_b2_1x7, inception_b2_1x7_2, inception_b3_1x7, inception_b3_1x7_2, inception_b4_1x7, inception_b4_1x7_2, inception_b5_1x7, inception_b5_1x7_2, inception_b6_1x7, inception_b6_1x7_2, inception_b7_1x7, inception_b7_1x7_2
        add_config(TensorShape(17U, 17U, 192U), TensorShape(7U, 1U, 192U, 224U), TensorShape(224U), TensorShape(17U, 17U, 224U), PadStrideInfo(1, 1, 3, 0));
        // inception_b1_7x1, inception_b2_7x1, inception_b3_7x1, inception_b4_7x1, inception_b5_7x1, inception_b6_7x1, inception_b7_7x1
        add_config(TensorShape(17U, 17U, 224U), TensorShape(1U, 7U, 224U, 256U), TensorShape(256U), TensorShape(17U, 17U, 256U), PadStrideInfo(1, 1, 0, 3));
        // inception_b1_7x1_2, inception_b2_7x1_2, inception_b3_7x1_2, inception_b4_7x1_2, inception_b5_7x1_2, inception_b6_7x1_2, inception_b7_7x1_2
        add_config(TensorShape(17U, 17U, 192U), TensorShape(1U, 7U, 192U, 192U), TensorShape(192U), TensorShape(17U, 17U, 192U), PadStrideInfo(1, 1, 0, 3));
        // inception_b1_7x1_3, inception_b2_7x1_3, inception_b3_7x1_3, inception_b4_7x1_3, inception_b5_7x1_3, inception_b6_7x1_3, inception_b7_7x1_3
        add_config(TensorShape(17U, 17U, 224U), TensorShape(1U, 7U, 224U, 224U), TensorShape(224U), TensorShape(17U, 17U, 224U), PadStrideInfo(1, 1, 0, 3));
        // inception_b1_1x7_3, inception_b2_1x7_3, inception_b3_1x7_3, inception_b4_1x7_3, inception_b5_1x7_3, inception_b6_1x7_3, inception_b7_1x7_3
        add_config(TensorShape(17U, 17U, 224U), TensorShape(7U, 1U, 224U, 256U), TensorShape(256U), TensorShape(17U, 17U, 256U), PadStrideInfo(1, 1, 3, 0));
        // inception_b1_1x1, inception_b2_1x1, inception_b3_1x1, inception_b4_1x1, inception_b5_1x1, inception_b6_1x1, inception_b7_1x1
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 128U), TensorShape(128U), TensorShape(17U, 17U, 128U), PadStrideInfo(1, 1, 0, 0));
        // reduction_b_3x3
        add_config(TensorShape(17U, 17U, 192U), TensorShape(3U, 3U, 192U, 192U), TensorShape(192U), TensorShape(8U, 8U, 192U), PadStrideInfo(2, 2, 0, 0));
        // reduction_b_1x7_reduce
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 256U), TensorShape(256U), TensorShape(17U, 17U, 256U), PadStrideInfo(1, 1, 0, 0));
        // reduction_b_1x7
        add_config(TensorShape(17U, 17U, 256U), TensorShape(7U, 1U, 256U, 256U), TensorShape(256U), TensorShape(17U, 17U, 256U), PadStrideInfo(1, 1, 3, 0));
        // reduction_b_7x1
        add_config(TensorShape(17U, 17U, 256U), TensorShape(1U, 7U, 256U, 320U), TensorShape(320U), TensorShape(17U, 17U, 320U), PadStrideInfo(1, 1, 0, 3));
        // reduction_b_3x3_2
        add_config(TensorShape(17U, 17U, 320U), TensorShape(3U, 3U, 320U, 320U), TensorShape(320U), TensorShape(8U, 8U, 320U), PadStrideInfo(2, 2, 0, 0));
        // inception_c1_1x1_2, inception_c1_1x1, inception_c2_1x1_2, inception_c2_1x1, inception_c3_1x1_2, inception_c3_1x1
        add_config(TensorShape(8U, 8U, 1536U), TensorShape(1U, 1U, 1536U, 256U), TensorShape(256U), TensorShape(8U, 8U, 256U), PadStrideInfo(1, 1, 0, 0));
        // inception_c1_1x1_3, inception_c1_1x1_4, inception_c2_1x1_3, inception_c2_1x1_4, inception_c3_1x1_3, inception_c3_1x1_4
        add_config(TensorShape(8U, 8U, 1536U), TensorShape(1U, 1U, 1536U, 384U), TensorShape(384U), TensorShape(8U, 8U, 384U), PadStrideInfo(1, 1, 0, 0));
        // inception_c1_1x3, inception_c2_1x3, inception_c3_1x3
        add_config(TensorShape(8U, 8U, 384U), TensorShape(3U, 1U, 384U, 256U), TensorShape(256U), TensorShape(8U, 8U, 256U), PadStrideInfo(1, 1, 1, 0));
        // inception_c1_3x1, inception_c2_3x1, inception_c3_3x1
        add_config(TensorShape(8U, 8U, 384U), TensorShape(1U, 3U, 384U, 256U), TensorShape(256U), TensorShape(8U, 8U, 256U), PadStrideInfo(1, 1, 0, 1));
        // inception_c1_3x1_2, inception_c2_3x1_2, inception_c3_3x1_2
        add_config(TensorShape(8U, 8U, 384U), TensorShape(1U, 3U, 384U, 448U), TensorShape(448U), TensorShape(8U, 8U, 448U), PadStrideInfo(1, 1, 0, 1));
        // inception_c1_1x3_2, inception_c2_1x3_2, inception_c3_1x3_2
        add_config(TensorShape(8U, 8U, 448U), TensorShape(3U, 1U, 448U, 512U), TensorShape(512U), TensorShape(8U, 8U, 512U), PadStrideInfo(1, 1, 1, 0));
        // inception_c1_1x3_3, inception_c2_1x3_3, inception_c3_1x3_3
        add_config(TensorShape(8U, 8U, 512U), TensorShape(3U, 1U, 512U, 256U), TensorShape(256U), TensorShape(8U, 8U, 256U), PadStrideInfo(1, 1, 1, 0));
        // inception_c1_3x1_3, inception_c2_3x1_3, inception_c3_3x1_3
        add_config(TensorShape(8U, 8U, 512U), TensorShape(1U, 3U, 512U, 256U), TensorShape(256U), TensorShape(8U, 8U, 256U), PadStrideInfo(1, 1, 0, 1));
    }
};

/** A subset of GoogLeNetInceptionV4 convolution layers with filter dimensions supported by DirectConvolution kernel */
class GoogLeNetInceptionV4DirectConvolutionLayerDataset final : public ConvolutionLayerDataset
{
public:
    // subset of GoogLeNetInceptionV4 inception v1 dataset
    GoogLeNetInceptionV4DirectConvolutionLayerDataset()
    {
        // conv1_3x3_s2
        add_config(TensorShape(299U, 299U, 3U), TensorShape(3U, 3U, 3U, 32U), TensorShape(32U), TensorShape(149U, 149U, 32U), PadStrideInfo(2, 2, 0, 0));
        // conv2_3x3_s1
        add_config(TensorShape(149U, 149U, 32U), TensorShape(3U, 3U, 32U, 32U), TensorShape(32U), TensorShape(147U, 147U, 32U), PadStrideInfo(1, 1, 0, 0));
        // conv3_3x3_s1
        add_config(TensorShape(147U, 147U, 32U), TensorShape(3U, 3U, 32U, 64U), TensorShape(64U), TensorShape(147U, 147U, 64U), PadStrideInfo(1, 1, 1, 1));
        // inception_stem1_3x3_s2
        add_config(TensorShape(147U, 147U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(73U, 73U, 96U), PadStrideInfo(2, 2, 0, 0));
        // inception_stem2_3x3_reduce, inception_stem2_1x7_reduce
        add_config(TensorShape(73U, 73U, 160U), TensorShape(1U, 1U, 160U, 64U), TensorShape(64U), TensorShape(73U, 73U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_stem2_3x3, inception_stem2_3x3_2
        add_config(TensorShape(73U, 73U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(71U, 71U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_stem3_3x3_s2
        add_config(TensorShape(71U, 71U, 192U), TensorShape(3U, 3U, 192U, 192U), TensorShape(192U), TensorShape(35U, 35U, 192U), PadStrideInfo(2, 2, 0, 0));
        // inception_a1_1x1_2, inception_a1_1x1, inception_a2_1x1_2, inception_a2_1x1, inception_a3_1x1_2, inception_a3_1x1, inception_a4_1x1_2, inception_a4_1x1
        add_config(TensorShape(35U, 35U, 384U), TensorShape(1U, 1U, 384U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 0, 0));
        // inception_a1_3x3_reduce, inception_a1_3x3_2_reduce, inception_a2_3x3_reduce, inception_a2_3x3_2_reduce, inception_a3_3x3_reduce, inception_a3_3x3_2_reduce, inception_a4_3x3_reduce, inception_a4_3x3_2_reduce
        add_config(TensorShape(35U, 35U, 384U), TensorShape(1U, 1U, 384U, 64U), TensorShape(64U), TensorShape(35U, 35U, 64U), PadStrideInfo(1, 1, 0, 0));
        // inception_a1_3x3, inception_a1_3x3_2, inception_a2_3x3, inception_a2_3x3_2, inception_a3_3x3, inception_a3_3x3_2, inception_a4_3x3, inception_a4_3x3_2
        add_config(TensorShape(35U, 35U, 64U), TensorShape(3U, 3U, 64U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 1, 1));
        // inception_a1_3x3_3, inception_a2_3x3_3, inception_a3_3x3_3, inception_a4_3x3_3
        add_config(TensorShape(35U, 35U, 96U), TensorShape(3U, 3U, 96U, 96U), TensorShape(96U), TensorShape(35U, 35U, 96U), PadStrideInfo(1, 1, 1, 1));
        // reduction_a_3x3
        add_config(TensorShape(35U, 35U, 384U), TensorShape(3U, 3U, 384U, 384U), TensorShape(384U), TensorShape(17U, 17U, 384U), PadStrideInfo(2, 2, 0, 0));
        // reduction_a_3x3_2_reduce
        add_config(TensorShape(35U, 35U, 384U), TensorShape(1U, 1U, 384U, 192U), TensorShape(192U), TensorShape(35U, 35U, 192U), PadStrideInfo(1, 1, 0, 0));
        // reduction_a_3x3_2
        add_config(TensorShape(35U, 35U, 192U), TensorShape(3U, 3U, 192U, 224U), TensorShape(224U), TensorShape(35U, 35U, 224U), PadStrideInfo(1, 1, 1, 1));
        // reduction_a_3x3_3
        add_config(TensorShape(35U, 35U, 224U), TensorShape(3U, 3U, 224U, 256U), TensorShape(256U), TensorShape(17U, 17U, 256U), PadStrideInfo(2, 2, 0, 0));
        // inception_b1_1x1_2, inception_b2_1x1_2, inception_b3_1x1_2, inception_b4_1x1_2, inception_b5_1x1_2, inception_b6_1x1_2, inception_b7_1x1_2
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 384U), TensorShape(384U), TensorShape(17U, 17U, 384U), PadStrideInfo(1, 1, 0, 0));
        // inception_b1_1x7_reduce, inception_b1_7x1_2_reduce, inception_b2_1x7_reduce, inception_b2_7x1_2_reduce, inception_b3_1x7_reduce, inception_b3_7x1_2_reduce, inception_b4_1x7_reduce, inception_b4_7x1_2_reduce, inception_b5_1x7_reduce, inception_b5_7x1_2_reduce, inception_b6_1x7_reduce, inception_b6_7x1_2_reduce, inception_b7_1x7_reduce, inception_b7_7x1_2_reduce, reduction_b_3x3_reduce
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 192U), TensorShape(192U), TensorShape(17U, 17U, 192U), PadStrideInfo(1, 1, 0, 0));
        // inception_b1_1x1, inception_b2_1x1, inception_b3_1x1, inception_b4_1x1, inception_b5_1x1, inception_b6_1x1, inception_b7_1x1
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 128U), TensorShape(128U), TensorShape(17U, 17U, 128U), PadStrideInfo(1, 1, 0, 0));
        // reduction_b_3x3
        add_config(TensorShape(17U, 17U, 192U), TensorShape(3U, 3U, 192U, 192U), TensorShape(192U), TensorShape(8U, 8U, 192U), PadStrideInfo(2, 2, 0, 0));
        // reduction_b_1x7_reduce
        add_config(TensorShape(17U, 17U, 1024U), TensorShape(1U, 1U, 1024U, 256U), TensorShape(256U), TensorShape(17U, 17U, 256U), PadStrideInfo(1, 1, 0, 0));
        // reduction_b_3x3_2
        add_config(TensorShape(17U, 17U, 320U), TensorShape(3U, 3U, 320U, 320U), TensorShape(320U), TensorShape(8U, 8U, 320U), PadStrideInfo(2, 2, 0, 0));
        // inception_c1_1x1_2, inception_c1_1x1, inception_c2_1x1_2, inception_c2_1x1, inception_c3_1x1_2, inception_c3_1x1
        add_config(TensorShape(8U, 8U, 1536U), TensorShape(1U, 1U, 1536U, 256U), TensorShape(256U), TensorShape(8U, 8U, 256U), PadStrideInfo(1, 1, 0, 0));
        // inception_c1_1x1_3, inception_c1_1x1_4, inception_c2_1x1_3, inception_c2_1x1_4, inception_c3_1x1_3, inception_c3_1x1_4
        add_config(TensorShape(8U, 8U, 1536U), TensorShape(1U, 1U, 1536U, 384U), TensorShape(384U), TensorShape(8U, 8U, 384U), PadStrideInfo(1, 1, 0, 0));
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GOOGLENETINCEPTIONV4_CONVOLUTION_LAYER_DATASET */
