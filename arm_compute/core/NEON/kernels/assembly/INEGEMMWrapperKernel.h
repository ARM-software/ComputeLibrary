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
#ifndef __ARM_COMPUTE_INEGEMMWRAPPERKERNEL_H__
#define __ARM_COMPUTE_INEGEMMWRAPPERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Common interface for all the arm_gemm Gemms
 */
class INEGEMMWrapperKernel : public INEKernel
{
public:
    /** Parameters defining the dimensions of the matrices being multiplied */
    struct Params
    {
        unsigned int M{ 0 };       /**< Rows in output matrix C (and input matrix A). */
        unsigned int N{ 0 };       /**< Columns in output matrix C (and input matrix B). */
        unsigned int K{ 0 };       /**< Columns of input matrix A (= rows of input matrix B). */
        unsigned int batches{ 0 }; /**< Number of "batched" GEMMs (unique A and C, shared B). */
        unsigned int multis{ 0 };  /**< Number of "multi" GEMMs (unique A, B and C). */
    };

    static Params extract_parameters(const ITensor *a, const ITensor *b, const ITensor *c, const GEMMInfo &gemm_info);

    /** Constructor */
    INEGEMMWrapperKernel();
    /** Prevent instances of this class from being copied */
    INEGEMMWrapperKernel(const INEGEMMWrapperKernel &) = delete;
    /** Prevent instances of this class from being copied */
    INEGEMMWrapperKernel &operator=(const INEGEMMWrapperKernel &) = delete;
    /** Allow instances of this class to be moved */
    INEGEMMWrapperKernel(INEGEMMWrapperKernel &&) = default;
    /** Allow instances of this class to be moved */
    INEGEMMWrapperKernel &operator=(INEGEMMWrapperKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @note The input and output tensor must have the same dimensions
     *
     * @param[in]  a         Input tensor (Matrix A)
     * @param[in]  b         Input tensor (Matrix B)
     * @param[out] c         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  alpha     Scalar multiplier to apply to AB matrix product.
     * @param[in]  beta      Scalar multiplier to apply to input C matrix before adding product.
     * @param[in]  gemm_info GEMM meta-data
     */
    void configure(const ITensor *a, const ITensor *b, ITensor *c, float alpha, float beta, const GEMMInfo &gemm_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

protected:
    /** Called as part of configure() after _a, _b, _c and _params have been set.
     *
     * @param[in] alpha Scalar multiplier to apply to AB matrix product.
     * @param[in] beta  Scalar multiplier to apply to input C matrix before adding product.
     *
     * @return A 3D execution window.
     */
    virtual Window configure_internal(float alpha, float beta) = 0;

    /** Run the kernel from the start to the end offset in window.
     *
     * @param[in] window       Window to use for the iteration
     * @param[in] start_offset Where to start iterating from (In Window coordinates)
     * @param[in] end_offset   Where to stop iterating (In Window coordinates).
     * @param[in] info         Info about executing thread and CPU.
     */
    virtual void run_internal(const Window &window, const Coordinates &start_offset, const Coordinates &end_offset, const ThreadInfo &info) = 0;

    const ITensor *_a;
    const ITensor *_b;
    ITensor       *_c;
    Params         _params;
    GEMMInfo       _gemm_info;

private:
    Window      _window3d;
    TensorShape _window_shape;
};

} // namespace arm_compute

#endif /* __ARM_COMPUTE_INEGEMMRAPPERKERNEL_H__ */
