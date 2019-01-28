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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDWRAPPER_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDWRAPPER_H__

#include "arm_compute/core/NEON/kernels/assembly/Helpers.h"
#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedMatrixMultiplyWrapper.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedPrepareBWrapperKernel.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMInterleavedTransformAWrapper.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IScheduler.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Buffer manager used when reshaping B on the fly
 *
 * The typical workflow is:
 * - lock_to_reshape_if_needed()
 * - If the previous lock was successful: mark_as_reshaped()
 * - wait_for_reshaping() wait for the reshaping to be complete
 * - mark_as_unused() once the thread is done using this given buffer.
 *
 * Calls for different indices might be interleaved, however the calls for a given index must always be in that order.
 */
class IBufferManager
{
public:
    /** Lock a buffer for the given index if it's available else return
     *
     * @param[in] index Index of the buffer to lock
     *
     * @return True if the buffer has been successfully locked, false if it's already reshaped / being reshaped.
     */
    virtual bool lock_to_reshape_if_needed(unsigned int index) = 0;
    /** Mark a buffer previously locked as reshaped
     *
     * @pre The thread calling this function must have locked the given buffer through lock_to_reshape_if_needed()
     *
     * @param[in] index Index of the buffer to mark as reshaped
     */
    virtual void mark_as_reshaped(unsigned int index) = 0;
    /** Block until the given buffer is marked as reshaped
     *
     * @param[in] index Index of the buffer
     */
    virtual void wait_for_reshaping(unsigned int index) = 0;
    /** Mark a reshaped buffer as unused
     *
     * Once all the users have marked a buffer as unused then it goes back to being free
     */
    virtual void mark_as_unused(unsigned int index) = 0;

    /** Number of buffers used internally
     *
     * @return The number of buffers used by the manager.
     */
    virtual unsigned int num_buffers() const = 0;
    /** Default destructor */
    virtual ~IBufferManager() = default;
};

/** Equivalent to arm_gemm::GemmInterleaved but using Compute Library types.
 */
class NEGEMMInterleavedWrapper : public IFunction
{
public:
    NEGEMMInterleavedWrapper(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    ~NEGEMMInterleavedWrapper()                                             = default;

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
     */
    void configure(const ITensor *a, const ITensor *b, ITensor *c, float alpha, float beta, bool pretranspose_b);

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
    std::unique_ptr<IBufferManager>                         _buffer_manager{ nullptr };
    std::vector<TransformAWorkload>                         _a_workloads{};
    std::vector<PrepareBWorkload>                           _b_workloads{};
    std::vector<MatrixMultiplyWorkload>                     _mm_workloads{};
    std::vector<IScheduler::Workload>                       _workloads{};
    std::string                                             _tag{};
    unsigned int                                            _num_windows{ 1 };
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDWRAPPER_H__ */
