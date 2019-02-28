/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMRESHAPELHSMATRIXKERNEL_H__
#define __ARM_COMPUTE_CLGEMMRESHAPELHSMATRIXKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to reshape the LHS matrix when performing the matrix multiplication.
 *  In particular, this function splits the input matrix in blocks of size M0xK0 (defined through GEMMLHSInfo) and
 *  stores each one in the output matrix unrolling the values
 */
class CLGEMMReshapeLHSMatrixKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLGEMMReshapeLHSMatrixKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMReshapeLHSMatrixKernel(const CLGEMMReshapeLHSMatrixKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMReshapeLHSMatrixKernel &operator=(const CLGEMMReshapeLHSMatrixKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMReshapeLHSMatrixKernel(CLGEMMReshapeLHSMatrixKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMReshapeLHSMatrixKernel &operator=(CLGEMMReshapeLHSMatrixKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input                   Input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[out] output                  Output tensor. Data type supported: same as @p input
     * @param[in]  lhs_info                LHS matrix information to be used for reshaping. This object contains all the necessary
     *                                     information to reshape the input tensor. Only the following values are supported:
     *                                     lhs_info.m0: 2,3,4,5,6,7,8
     *                                     lhs_info.k0: 2,3,4,8,16
     *                                     lhs_info.v0: greater than 0
     *                                     lhs_info.transpose: true, false
     *                                     lhs_info.interleave: true, false
     * @param[in]  reinterpret_input_as_3d (Optional) True if the input has to be reinterpreted as 3D tensor
     */
    void configure(const ICLTensor *input, ICLTensor *output, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_input_as_3d = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMReshapeLHSMatrixKernel
     *
     * @param[in] input                   Input tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in] output                  Output tensor info which stores the interleaved matrix. Data type supported: same as @p input.
     * @param[in] lhs_info                LHS matrix information to be used for reshaping. This object contains all the necessary
     *                                    information to reshape the input tensor. Only the following values are supported:
     *                                    lhs_info.m0: 2,3,4,5,6,7,8
     *                                    lhs_info.k0: 2,3,4,8,16
     *                                    lhs_info.v0: greater than 0
     *                                    lhs_info.transpose: true, false
     *                                    lhs_info.interleave: true, false
     * @param[in] reinterpret_input_as_3d True if the input has to be reinterpreted as 3D tensor
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_input_as_3d);

    // Inherited methods overridden
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    bool             _reinterpret_input_as_3d;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMRESHAPELHSMATRIXKERNEL_H__ */