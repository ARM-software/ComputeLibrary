/*
 * Copyright (c) 2017, 2018, 2025 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_ALEXNETGEMMDATASET_H
#define ACL_TESTS_DATASETS_ALEXNETGEMMDATASET_H

#include "tests/datasets/GEMMDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class AlexNetGEMMDataset final : public GEMMDataset
{
public:
    AlexNetGEMMDataset()
    {
        add_config(1, 1000, 4096, 1.f, 0.f);
        add_config(1, 4096, 4096, 1.f, 0.f);
        add_config(1, 4096, 9216, 1.f, 0.f);
        add_config(169, 128, 1729, 1.f, 0.f);
        add_config(169, 192, 1729, 1.f, 0.f);
        add_config(169, 384, 2305, 1.f, 0.f);
        add_config(729, 128, 1201, 1.f, 0.f);
        add_config(3025, 96, 364, 1.f, 0.f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_ALEXNETGEMMDATASET_H
