/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_SRC_CPU_OPERATORS_CPUMATMUL
#define ACL_SRC_CPU_OPERATORS_CPUMATMUL

#include "arm_compute/core/TensorInfo.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuTransposeKernel.h"
#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

namespace arm_compute
{
// Forward Declarations
class MatMulInfo;
class CpuMatMulSettings;

namespace cpu
{
/** Function to execute MatMul Operation. This function calls the following functions/kernels:
 *
 * If adjoint/adj flag is enabled for either input lhs or rhs (or both) :
 *  -# @ref cpu::kernels::CpuTransposeKernel
 * Then :
 *  -# @ref cpu::CpuGemmAssemblyDispatch
 */
class CpuMatMul : public ICpuOperator
{
public:
    /* Constructor */
    CpuMatMul();
    /* Destructor */
    ~CpuMatMul() = default;

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuMatMul);
    /** Configure operator for a given list of arguments
     *
     * Note: Check documentation of @ref NEMatMul for a list of supported datatypes and layouts
     *
     *
     * @param[in]  lhs      Left-hand side tensor info.
     * @param[in]  rhs      Right-hand side tensor info.
     * @param[out] dst      Output tensor to store the result of the batched matrix multiplication. Data types supported: same as @p lhs / @p rhs.
     * @param[in]  info     Contains MatMul operation information described in @ref MatMulInfo.
     * @param[in]  settings The settings for matmul operation (i.e fast math)
     */
    void configure(ITensorInfo *lhs, ITensorInfo *rhs, ITensorInfo *dst, const MatMulInfo &info, const CpuMatMulSettings &settings);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuMatMul::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, const MatMulInfo &info, const CpuMatMulSettings &settings);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum InternalTensorIdx
    {
        AsmGemmWorkspace = 0, // Pre-allocate workspace tensors for CpuGemmAssemblyDispatch
        PretransposeRHS,      // Pre-allocate workspace tensors for CpuGemmAssemblyDispatch
        TransposeLHS,
        TransposeRHS,
        Count
    };

    // Define unique pointers to kernels/operators used by matmul
    std::unique_ptr<kernels::CpuTransposeKernel> _transpose_kernel_lhs{ nullptr };
    std::unique_ptr<kernels::CpuTransposeKernel> _transpose_kernel_rhs{ nullptr };
    std::unique_ptr<CpuGemmAssemblyDispatch>     _asm_glue{ nullptr };

    // TensorInfo for tensors stored in auxillary memory
    TensorInfo _lhs_transposed{};
    TensorInfo _rhs_transposed{};

    // Original tensor shapes prior to reshaping tensors and collapsing dimensions
    TensorShape _original_lhs_shape{};
    TensorShape _original_rhs_shape{};
    TensorShape _original_dst_shape{};

    // Note : adj_lhs means the same as transposing lhs
    bool                             _adj_lhs{ false };
    bool                             _adj_rhs{ false };
    bool                             _fast_math{ false };
    AsmGemmInfo                      _gemm_info{};
    experimental::MemoryRequirements _aux_mem{ Count };
};
}
}

#endif /* ACL_SRC_CPU_OPERATORS_CPUMATMUL */
