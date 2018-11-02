/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDWRAPPER_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDWRAPPER_H__

#include "arm_compute/core/NEON/kernels/assembly/Helpers.h"
#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IScheduler.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class NEGEMMInterleavedPrepareBWrapperKernel;
class PrepareBWorkload;
class TransformAWorkload;
class MatrixMultiplyWorkload;
class NEGEMMInterleavedTransformAWrapper;
class NEGEMMInterleavedMatrixMultiplyWrapper;

/** Equivalent to arm_gemm::GemmInterleaved but using Compute Library types.
 */
class NEGEMMInterleavedWrapper : public IFunction
{
public:
    NEGEMMInterleavedWrapper(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    NEGEMMInterleavedWrapper(const NEGEMMInterleavedWrapper &) = delete;
    NEGEMMInterleavedWrapper &operator=(const NEGEMMInterleavedWrapper &) = delete;

    /** Initialise the kernel's input and output.
     *
     * @note The input and output tensor must have the same dimensions
     *
     * @param[in]  a              Input tensor (Matrix A)
     * @param[in]  b              Input tensor (Matrix B)
     * @param[out] c              Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  alpha          Scalar multiplier to apply to AB matrix product.
     * @param[in]  beta           Scalar multiplier to apply to input C matrix before adding product.
     * @param[in]  pretranspose_b If true, pretranspose B once during the prepare() stage instead of on the fly every time.
     * @param[in]  use_dot        (Optional) If the input's type is U8/S8/QASYMM8 then use the dot product flavour or the matrix multiply routine. (Must be supported by the hardware).
     */
    void configure(const ITensor *a, const ITensor *b, ITensor *c, float alpha, float beta, bool pretranspose_b, bool use_dot = false);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup                                             _memory_group;
    bool                                                    _is_prepared{ false };
    bool                                                    _pretranspose_b{ false };
    Window                                                  _block_walker{};
    Window                                                  _batch_window{};
    const ITensor                                          *_a{ nullptr };
    const ITensor                                          *_b{ nullptr };
    ITensor                                                *_c{ nullptr };
    Tensor                                                  _transformed_b{};
    Tensor                                                  _transformed_a{};
    Tensor                                                  _tmp_c{};
    INEGEMMWrapperKernel::Params                            _params{};
    BlockSizes                                              _block_sizes{};
    std::unique_ptr<NEGEMMInterleavedPrepareBWrapperKernel> _prepare_b{ nullptr };
    std::unique_ptr<NEGEMMInterleavedTransformAWrapper>     _transform_a{ nullptr };
    std::unique_ptr<NEGEMMInterleavedMatrixMultiplyWrapper> _matrix_multiply{ nullptr };
    std::vector<TransformAWorkload>                         _a_workloads{};
    std::vector<PrepareBWorkload>                           _b_workloads{};
    std::vector<MatrixMultiplyWorkload>                     _mm_workloads{};
    std::vector<IScheduler::Workload>                       _workloads{};
};

} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDWRAPPER_H__ */
