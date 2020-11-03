/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NECONCATENATELAYER_H
#define ARM_COMPUTE_NECONCATENATELAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INEOperator.h"
#include "support/Requires.h"

#include <memory>
#include <vector>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;
class Status;

/** Basic function to execute concatenate tensors along a given axis. This function calls the following kernels:
 *
 * -# @ref NEWidthConcatenateLayerKernel (if underlying concatenation axis is 0).
 * -# @ref NEHeightConcatenateLayerKernel (if underlying concatenation axis is 1).
 * -# @ref NEDepthConcatenateLayerKernel (if underlying concatenation axis is 2).
 * -# @ref NEBatchConcatenateLayerKernel (if underlying concatenation axis is 3).
 */
class NEConcatenateLayer : public IFunction
{
public:
    /** Default constructor */
    NEConcatenateLayer();
    /** Destructor */
    ~NEConcatenateLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConcatenateLayer(const NEConcatenateLayer &) = delete;
    /** Default move constructor */
    NEConcatenateLayer(NEConcatenateLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConcatenateLayer &operator=(const NEConcatenateLayer &) = delete;
    /** Default move assignment operator */
    NEConcatenateLayer &operator=(NEConcatenateLayer &&);
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref NEWidthConcatenateLayerKernel, @ref NEHeightConcatenateLayerKernel and @ref NEDepthConcatenateLayerKernel.
     *
     * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]    output        Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     */
    void configure(std::vector<const ITensor *> inputs_vector, ITensor *output, size_t axis);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConcatenateLayer
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref NEWidthConcatenateLayerKernel, @ref NEHeightConcatenateLayerKernel and @ref NEDepthConcatenateLayerKernel.
     *
     * @param[in] inputs_vector The vectors containing all the tensors info to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output        Output tensor info. Data types supported: Same as @p input.
     * @param[in] axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     *
     * @return a status
     */
    static Status validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

namespace experimental
{
/** Basic function to execute concatenate tensors along a given axis. This function calls the following kernels:
 *
 * -# @ref NEWidthConcatenateLayerKernel (if underlying concatenation axis is 0).
 * -# @ref NEHeightConcatenateLayerKernel (if underlying concatenation axis is 1).
 * -# @ref NEDepthConcatenateLayerKernel (if underlying concatenation axis is 2).
 * -# @ref NEBatchConcatenateLayerKernel (if underlying concatenation axis is 3).
 */
class NEConcatenation : public INEOperator
{
public:
    /** Constructor */
    NEConcatenation();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConcatenation(const NEConcatenation &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConcatenation &operator=(const NEConcatenation &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConcatenation(NEConcatenation &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConcatenation &operator=(NEConcatenation &&) = delete;
    /** Default destructor */
    ~NEConcatenation() = default;
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref NEWidthConcatenateLayerKernel, @ref NEHeightConcatenateLayerKernel and @ref NEDepthConcatenateLayerKernel.
     *
     * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]    output        Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     */
    void configure(const std::vector<const ITensorInfo *> &inputs_vector, ITensorInfo *output, size_t axis);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConcatenateLayer
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref NEWidthConcatenateLayerKernel, @ref NEHeightConcatenateLayerKernel and @ref NEDepthConcatenateLayerKernel.
     *
     * @param[in] inputs_vector The vectors containing all the tensors info to concatenate. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output        Output tensor info. Data types supported: Same as @p input.
     * @param[in] axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     *
     * @return a status
     */
    static Status validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::vector<std::unique_ptr<ICPPKernel>> _concat_kernels;
    unsigned int                             _num_inputs;
    unsigned int                             _axis;
};
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_NECONCATENATELAYER_H */
