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
#ifndef ARM_COMPUTE_NEMEMSETKERNEL_H
#define ARM_COMPUTE_NEMEMSETKERNEL_H

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for filling the planes of a tensor */
class NEMemsetKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEMemsetKernel";
    }
    /** Default constructor */
    NEMemsetKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMemsetKernel(const NEMemsetKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMemsetKernel &operator=(const NEMemsetKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEMemsetKernel(NEMemsetKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEMemsetKernel &operator=(NEMemsetKernel &&) = default;
    /** Default destructor */
    ~NEMemsetKernel() = default;
    /** Initialise the kernel's tensor and filling value
     *
     * @param[in,out] tensor         Input tensor to fill. Supported data types: All
     * @param[in]     constant_value The value used to fill the planes of the tensor
     */
    void configure(ITensor *tensor, const PixelValue &constant_value);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    ITensor   *_tensor;
    PixelValue _constant_value;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEMEMSETKERNEL_H */
