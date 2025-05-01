/*
 * Copyright (c) 2017-2018, 2025 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_MATRIXMULTIPLYGEMMDATASET_H
#define ACL_TESTS_DATASETS_MATRIXMULTIPLYGEMMDATASET_H

#include "tests/datasets/GEMMDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class MatrixMultiplyGEMMDataset final : public GEMMDataset
{
public:
    MatrixMultiplyGEMMDataset()
    {
        add_config(1, 1000, 1024, 1.0f, 0.0f);
        add_config(128, 512, 512, 0.5f, 0.0f);
        add_config(128, 512, 512, 1.2f, 1.1f);
        add_config(256, 128, 128, 1.1f, 0.0f);
        add_config(256, 128, 128, 1.1f, 1.5f);
        add_config(256, 256, 256, 0.5f, 0.0f);
        add_config(256, 256, 256, 0.5f, 1.3f);
        add_config(784, 64, 256, 1.0f, 0.0f);
        add_config(2704, 256, 1152, 1.0f, 0.0f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_MATRIXMULTIPLYGEMMDATASET_H
