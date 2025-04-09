/*
 * Copyright (c) 2017-2019, 2024-2025 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_LARGEGEMMDATASET_H
#define ACL_TESTS_DATASETS_LARGEGEMMDATASET_H

#include "arm_compute/core/TensorShape.h"

#include "tests/datasets/GEMMDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class LargeGEMMDataset final : public GEMMDataset
{
public:
    LargeGEMMDataset()
    {
        add_config(1, 623, 941, 0.4f, 0.7f);
        add_config(1, 783, 1021, 1.0f, 0.0f);
        add_config(1, 783, 1021, 1.0f, 1.0f);
        add_config(429, 871, 923, 1.0f, 0.0f);
        add_config(1023, 213, 681, 0.2f, 1.2f);
    }
};

class LargeGEMMOutput3DDataset final : public GEMMDataset
{
public:
    LargeGEMMOutput3DDataset()
    {
        add_config(TensorShape(364U, 3025U), TensorShape(96U, 364U), TensorShape(96U), TensorShape(96U, 605U, 5U), 1.0f, 0.0f);
        add_config(TensorShape(681U, 1025U), TensorShape(213U, 681U), TensorShape(213U), TensorShape(213U, 205U, 5U), 1.0f, 0.0f);
        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U), TensorShape(871U, 143U, 3U), 1.0f, 0.0f);
        add_config(TensorShape(1201U, 729U), TensorShape(128U, 1201U), TensorShape(128U), TensorShape(128U, 243U, 3U), 1.0f, 0.0f);
        add_config(TensorShape(1729U, 170U), TensorShape(128U, 1729U), TensorShape(128U), TensorShape(128U, 17U, 10U), 1.0f, 0.0f);
        add_config(TensorShape(1729U, 170U), TensorShape(192U, 1729U), TensorShape(192U), TensorShape(192U, 85U, 2U), 1.0f, 0.0f);
        add_config(TensorShape(2305U, 169U), TensorShape(384U, 2305U), TensorShape(384U), TensorShape(384U, 13U, 13U), 1.0f, 0.0f);
    }
};

class LargeGEMMInputOutput3DDataset final : public GEMMDataset
{
public:
    LargeGEMMInputOutput3DDataset()
    {
        add_config(TensorShape(364U, 605U, 5U), TensorShape(96U, 364U), TensorShape(96U), TensorShape(96U, 605U, 5U), 0.2f, 1.2f);
        add_config(TensorShape(681U, 205U, 5U), TensorShape(213U, 681U), TensorShape(213U), TensorShape(213U, 205U, 5U), 1.0f, 0.0f);
        add_config(TensorShape(923U, 143U, 3U), TensorShape(871U, 923U), TensorShape(871U), TensorShape(871U, 143U, 3U), 1.0f, 0.0f);
        add_config(TensorShape(1201U, 243U, 3U), TensorShape(128U, 1201U), TensorShape(128U), TensorShape(128U, 243U, 3U), 1.0f, 0.0f);
        add_config(TensorShape(1729U, 17U, 10U, 3U), TensorShape(128U, 1729U), TensorShape(128U), TensorShape(128U, 17U, 10U, 3U), 1.0f, 0.3f);
        add_config(TensorShape(1729U, 85U, 2U, 2U), TensorShape(192U, 1729U), TensorShape(192U), TensorShape(192U, 85U, 2U, 2U), 1.0f, 0.0f);
        add_config(TensorShape(2305U, 13U, 13U), TensorShape(384U, 2305U), TensorShape(384U), TensorShape(384U, 13U, 13U), 0.4f, 0.7f);
    }
};

class LargeAccumulateGEMMDataset final : public GEMMDataset
{
public:
    LargeAccumulateGEMMDataset()
    {
        add_config(1, 623, 941, 1.0f, 0.0f);
        add_config(1, 783, 1021, 1.0f, 0.0f);
        add_config(429, 871, 923, 1.0f, 0.0f);
    }
};

class LargeGEMMVectorBiasDataset final : public GEMMDataset
{
public:
    LargeGEMMVectorBiasDataset()
    {
        add_config(1, 783, 1021, 1.0f, 1.0f);
        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U, 1U), TensorShape(871U, 429U), 1.0f, 1.0f);
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_LARGEGEMMDATASET_H
