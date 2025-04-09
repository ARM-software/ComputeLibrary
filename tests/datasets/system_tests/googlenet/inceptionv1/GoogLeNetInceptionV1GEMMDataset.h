/*
 * Copyright (c) 2017, 2025 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_SYSTEM_TESTS_GOOGLENET_INCEPTIONV1_GOOGLENETINCEPTIONV1GEMMDATASET_H
#define ACL_TESTS_DATASETS_SYSTEM_TESTS_GOOGLENET_INCEPTIONV1_GOOGLENETINCEPTIONV1GEMMDATASET_H

#include "tests/datasets/GEMMDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GoogLeNetInceptionV1GEMMDataset final : public GEMMDataset
{
public:
    GoogLeNetInceptionV1GEMMDataset()
    {
        add_config(1, 1008, 1024, 1.0f, 0.0f);
        add_config(1, 1024, 2048, 1.0f, 0.0f);
        add_config(16, 128, 508, 1.0f, 0.0f);
        add_config(16, 128, 528, 1.0f, 0.0f);
        add_config(49, 128, 1200, 1.0f, 0.0f);
        add_config(49, 128, 832, 1.0f, 0.0f);
        add_config(49, 160, 832, 1.0f, 0.0f);
        add_config(49, 192, 832, 1.0f, 0.0f);
        add_config(49, 256, 832, 1.0f, 0.0f);
        add_config(49, 320, 1440, 1.0f, 0.0f);
        add_config(49, 384, 1728, 1.0f, 0.0f);
        add_config(49, 384, 832, 1.0f, 0.0f);
        add_config(49, 48, 832, 1.0f, 0.0f);
        add_config(196, 112, 508, 1.0f, 0.0f);
        add_config(196, 112, 512, 1.0f, 0.0f);
        add_config(196, 128, 512, 1.0f, 0.0f);
        add_config(196, 128, 528, 1.0f, 0.0f);
        add_config(196, 128, 800, 1.0f, 0.0f);
        add_config(196, 144, 512, 1.0f, 0.0f);
        add_config(196, 16, 480, 1.0f, 0.0f);
        add_config(196, 160, 508, 1.0f, 0.0f);
        add_config(196, 160, 528, 1.0f, 0.0f);
        add_config(196, 192, 480, 1.0f, 0.0f);
        add_config(196, 204, 864, 1.0f, 0.0f);
        add_config(196, 224, 1008, 1.0f, 0.0f);
        add_config(196, 24, 508, 1.0f, 0.0f);
        add_config(196, 24, 512, 1.0f, 0.0f);
        add_config(196, 256, 1152, 1.0f, 0.0f);
        add_config(196, 256, 528, 1.0f, 0.0f);
        add_config(196, 288, 1296, 1.0f, 0.0f);
        add_config(196, 32, 512, 1.0f, 0.0f);
        add_config(196, 32, 528, 1.0f, 0.0f);
        add_config(196, 320, 1440, 1.0f, 0.0f);
        add_config(196, 48, 400, 1.0f, 0.0f);
        add_config(196, 64, 480, 1.0f, 0.0f);
        add_config(196, 64, 508, 1.0f, 0.0f);
        add_config(196, 64, 512, 1.0f, 0.0f);
        add_config(196, 64, 600, 1.0f, 0.0f);
        add_config(196, 64, 800, 1.0f, 0.0f);
        add_config(196, 96, 480, 1.0f, 0.0f);
        add_config(784, 128, 256, 1.0f, 0.0f);
        add_config(784, 128, 864, 1.0f, 0.0f);
        add_config(784, 16, 192, 1.0f, 0.0f);
        add_config(784, 192, 1152, 1.0f, 0.0f);
        add_config(784, 32, 192, 1.0f, 0.0f);
        add_config(784, 32, 256, 1.0f, 0.0f);
        add_config(784, 32, 400, 1.0f, 0.0f);
        add_config(784, 64, 192, 1.0f, 0.0f);
        add_config(784, 64, 256, 1.0f, 0.0f);
        add_config(784, 96, 192, 1.0f, 0.0f);
        add_config(784, 96, 800, 1.0f, 0.0f);
        add_config(3136, 192, 576, 1.0f, 0.0f);
        add_config(3136, 64, 64, 1.0f, 0.0f);
        add_config(12544, 64, 147, 1.0f, 0.0f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_SYSTEM_TESTS_GOOGLENET_INCEPTIONV1_GOOGLENETINCEPTIONV1GEMMDATASET_H
