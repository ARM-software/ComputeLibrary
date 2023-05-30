/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_FULLYCONNECTEDLAYERINFO_H
#define ARM_COMPUTE_FULLYCONNECTEDLAYERINFO_H

#include "arm_compute/core/ActivationLayerInfo.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** Fully connected layer info */
struct FullyConnectedLayerInfo
{
    /* Fused-activation parameters */
    ActivationLayerInfo activation_info{}; /**<  Fused activation to apply after the matrix multiplication. */
    /* Information about weights */
    DataLayout weights_trained_layout{ DataLayout::NCHW }; /**<  Layout that the weights have been trained with. */
    bool       transpose_weights{ true };                  /**<  Transpose weights if true. */
    bool       are_weights_reshaped{ false };              /**<  @deprecated Reshape the weights tensor if false. */
    bool       retain_internal_weights{ false };           /**<  Retain internal reshaped weights. */
    bool       enable_fast_math{ false };                  /**<  Enable fast math computation. */
    /* Other parameters */
    bool fp_mixed_precision{ false }; /**<  Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy. */

    /** Sets the weights trained data layout
     *
     * @param[in] layout Data layout that the weights were trained with
     *
     * @return Updated object
     */
    FullyConnectedLayerInfo &set_weights_trained_layout(DataLayout layout)
    {
        weights_trained_layout = layout;
        return *this;
    }
    /** Sets the transpose weights flag
     *
     * @param[in] should_transpose_weights Boolean flag indicating if weights should be transposed
     *
     * @return Updated object
     */
    FullyConnectedLayerInfo &set_transpose_weights(bool should_transpose_weights)
    {
        transpose_weights = should_transpose_weights;
        return *this;
    }
};

} // namespace arm_compute
#endif /* ARM_COMPUTE_FULLYCONNECTEDLAYERINFO_H */
