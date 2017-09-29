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
#ifndef __ARM_COMPUTE_GRAPH_TYPES_H__
#define __ARM_COMPUTE_GRAPH_TYPES_H__

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"

namespace arm_compute
{
namespace graph
{
using arm_compute::ActivationLayerInfo;
using arm_compute::ITensor;
using arm_compute::TensorInfo;
using arm_compute::DataType;
using arm_compute::TensorShape;
using arm_compute::PadStrideInfo;
using arm_compute::WeightsInfo;
using arm_compute::PoolingLayerInfo;
using arm_compute::PoolingType;

/**< Execution hint to the graph executor */
enum class Hint
{
    DONT_CARE, /**< Run node in any device */
    OPENCL,    /**< Run node on an OpenCL capable device (GPU) */
    NEON       /**< Run node on a NEON capable device */
};

} // namespace graph
} // namespace arm_compute
#endif /*__ARM_COMPUTE_GRAPH_TYPES_H__*/
