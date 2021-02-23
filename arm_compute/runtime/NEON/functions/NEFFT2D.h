/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEFFT2D_H
#define ARM_COMPUTE_NEFFT2D_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEFFT1D.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
// Forward declaration
class ITensor;

/** Basic function to execute two dimensional FFT. This function calls the following Neon kernels:
 *
 * -# @ref NEFFT1D 1D FFT is performed on the first given axis
 * -# @ref NEFFT1D 1D FFT is performed on the second given axis
 */
class NEFFT2D : public IFunction
{
public:
    /** Default Constructor */
    NEFFT2D(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFT2D(const NEFFT2D &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFT2D &operator=(const NEFFT2D &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEFFT2D(NEFFT2D &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEFFT2D &operator=(NEFFT2D &&) = delete;
    /** Default destructor */
    ~NEFFT2D();
    /** Initialise the function's source and destinations
     *
     * @param[in]  input  Source tensor. Data types supported: F32.
     * @param[out] output Destination tensor. Data types and data layouts supported: Same as @p input.
     * @param[in]  config FFT related configuration
     */
    void configure(const ITensor *input, ITensor *output, const FFT2DInfo &config);
    /** Static function to check if given info will lead to a valid configuration of @ref NEFFT2D.
     *
     * @param[in] input  Source tensor info. Data types supported: F32.
     * @param[in] output Destination tensor info. Data types and data layouts supported: Same as @p input.
     * @param[in] config FFT related configuration
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const FFT2DInfo &config);

    // Inherited methods overridden:
    void run() override;

protected:
    MemoryGroup _memory_group;
    NEFFT1D     _first_pass_func;
    NEFFT1D     _second_pass_func;
    Tensor      _first_pass_tensor;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEFFT2D_H */
