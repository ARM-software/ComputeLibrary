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
#ifndef __ARM_COMPUTE_GRAPH_CLMAP_H__
#define __ARM_COMPUTE_GRAPH_CLMAP_H__

#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

namespace graph
{
class ITensorObject;
/** OpenCL map function */
class CLMap : public arm_compute::IFunction
{
public:
    /** Constructor
     *
     * @param[in] tensor   Tensor to map
     * @param[in] blocking Flag to specify if the map should be blocking or not (defaults to false)
     */
    CLMap(ITensorObject *tensor, bool blocking = false);
    /** Prevent instances from being copy constructed */
    CLMap(const CLMap &) = delete;
    /** Prevent instances from being copy assigned */
    const CLMap &operator=(const CLMap &) = delete;
    /** Allow instances to be move constructed */
    CLMap(CLMap &&) = default;
    /** Allow instances to be move assigned */
    CLMap &operator=(CLMap &&) = default;

    // Inherited methods overriden:
    void run() override;

private:
    arm_compute::ICLTensor *_tensor;   /**< Tensor */
    bool                    _blocking; /**< Blocking flag */
};
} // namespace graph
} // namespace arm_compute

#endif /* __ARM_COMPUTE_GRAPH_CLMAP_H__ */
