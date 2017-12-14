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
#ifndef __ARM_COMPUTE_NENORMALIZATIONLAYERKERNEL_H__
#define __ARM_COMPUTE_NENORMALIZATIONLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the normalization layer kernel.
 */
class NENormalizationLayerKernel : public INEKernel
{
public:
    /** Default constructor */
    NENormalizationLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NENormalizationLayerKernel(const NENormalizationLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NENormalizationLayerKernel &operator=(const NENormalizationLayerKernel &) = delete;
    /** Default Move Constructor. */
    NENormalizationLayerKernel(NENormalizationLayerKernel &&) = default;
    /** Default move assignment operator. */
    NENormalizationLayerKernel &operator=(NENormalizationLayerKernel &&) = default;
    /** Default destructor */
    ~NENormalizationLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input         Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                           and an optional 4th dimension for batch of inputs. Data types supported: QS8/QS16/FP16/F32.
     * @param[in]  input_squared Source with each element has been squared. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                           Data type supported: same as @p input
     * @param[out] output        Destination tensor. Output will have the same number of dimensions as input. Data type supported: same as @p input
     * @param[in]  norm_info     Normalization layer information like the normalization type, normalization size and other parameters.
     */
    void configure(const ITensor *input, const ITensor *input_squared, ITensor *output, NormalizationLayerInfo norm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NENormalizationLayerKernel
     *
     * @param[in] input         Source tensor. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                          and an optional 4th dimension for batch of inputs. Data types supported: QS8/QS16/FP16/F32.
     * @param[in] input_squared Source with each element has been squared. 3 lower dims represent a single input with dimensions [width, height, IFM],
     *                          Data type supported: same as @p input
     * @param[in] output        Destination tensor. Output will have the same number of dimensions as input. Data type supported: same as @p input
     * @param[in] norm_info     Normalization layer information like the normalization type, normalization size and other parameters.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *input_squared, const ITensorInfo *output, NormalizationLayerInfo norm_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Function to perform normalization depending on the given template
     *  dimension. The second template parameter specifies whether the
     *  normalization has to be 1D or 2D.
     *
     * @note Only supported normalizations are:
     *  - 1D over X or Z
     *  - 2D over X and Y
     *
     * @param[in] window Region on which to execute the kernel.
     */
    template <DataType dt, unsigned int dim, bool do_2D_norm>
    void normalize_float(const Window &window);

    /** Function to perform normalization for fixed-point values depending on
     * the given template dimension. The second template parameter specifies
     * whether the normalization has to be 1D or 2D.
     *
     * @note Only supported normalizations are:
     *  - 1D over X or Z
     *  - 2D over X and Y
     *
     * @param[in] window Region on which to execute the kernel.
     */
    template <DataType dt, unsigned int dim, bool do_2D_norm>
    void normalize_fixed_point(const Window &window);
    /** Common signature for all the specialised normalization functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using NormalizationFunction = void (NENormalizationLayerKernel::*)(const Window &window);

private:
    NormalizationFunction  _func;
    const ITensor         *_input;
    const ITensor         *_input_squared;
    ITensor               *_output;
    NormalizationLayerInfo _norm_info;
    BorderSize             _border_size;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NENORMALIZATIONLAYERKERNEL_H__ */
