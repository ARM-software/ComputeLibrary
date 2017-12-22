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
#ifndef __ARM_COMPUTE_GRAPH_CLUNMAP_H__
#define __ARM_COMPUTE_GRAPH_CLUNMAP_H__

#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

namespace graph
{
class ITensorObject;
/** OpenCL un-map function */
class CLUnmap : public arm_compute::IFunction
{
public:
    /** Constructor
     *
     * @param[in] tensor Tensor to un-map
     */
    CLUnmap(ITensorObject *tensor);
    /** Prevent instances from being copy constructed */
    CLUnmap(const CLUnmap &) = delete;
    /** Prevent instances from being copy assigned */
    const CLUnmap &operator=(const CLUnmap &) = delete;
    /** Allow instances to be move constructed */
    CLUnmap(CLUnmap &&) = default;
    /** Allow instances to be move assigned */
    CLUnmap &operator=(CLUnmap &&) = default;

    // Inherited methods overriden:
    void run() override;

private:
    arm_compute::ICLTensor *_tensor; /**< Tensor */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_CLUNMAP_H__ */
