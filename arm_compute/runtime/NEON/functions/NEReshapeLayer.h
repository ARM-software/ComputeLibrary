/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NERESHAPELAYER_H
#define ARM_COMPUTE_NERESHAPELAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INEOperator.h"
#include "arm_compute/runtime/Types.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to run @ref cpu::kernels::CpuReshapeKernel */
class NEReshapeLayer : public IFunction
{
public:
    /** Default Constructor */
    NEReshapeLayer();
    /** Default Destructor */
    ~NEReshapeLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReshapeLayer(const NEReshapeLayer &) = delete;
    /** Default move constructor */
    NEReshapeLayer(NEReshapeLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReshapeLayer &operator=(const NEReshapeLayer &) = delete;
    /** Default move assignment operator */
    NEReshapeLayer &operator=(NEReshapeLayer &&);
    /** Initialise the kernel's inputs and outputs
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src    |dst    |
     * |:------|:------|
     * |All    |All    |
     *
     * @param[in]  input  Input tensor. Data type supported: All
     * @param[out] output Output tensor. Data type supported: Same as @p input
     */
    void configure(const ITensor *input, ITensor *output);

    /** Static function to check if given info will lead to a valid configuration of @ref NEReshapeLayer
     *
     * @param[in] input  Input tensor info. Data type supported: All
     * @param[in] output Output tensor info. Data type supported: Same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NERESHAPELAYER_H */
