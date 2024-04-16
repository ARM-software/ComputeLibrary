#ifndef ARM_COMPUTE_CPU_LINEAR_H
#define ARM_COMPUTE_CPU_LINEAR_H

#include "arm_compute/core/TensorInfo.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"
#include "src/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/cpu/kernels/CpuGemmTranspose1xWKernel.h"

namespace arm_compute
{
namespace cpu
{

/** Basic function to run @ref kernels::CpuLinearKernel 
 * @note Performs linear function [alpha * A * B + beta * C]
*/
class CpuLinear : public ICpuOperator
{
public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  a      An input tensor. Data type supported: f32.
     * @param[in]  b      An input tensor. Data type supported: f32.
     * @param[in]  c      An input bias ensor. Data type supported: f32.
     * @param[out] d      Output tensor. Data type supported: F32.
     * @param[in]  alpha  Weight of the matrix product
     * @param[in]  beta   Weight of matrix C
     * @param[in]  info   (Optional)Linear layer operation information
     */
    void configure(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float              beta, 
                   const LinearLayerInfo& info = LinearLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CpuLinearKernel
     *
     * Similar to @ref CpuGemm::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a,
                           const ITensorInfo *b,
                           const ITensorInfo *c,
                           ITensorInfo       *d,
                           float              alpha,
                           float              beta,
                           const LinearLayerInfo& info = LinearLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    enum AuxTensorIdx
    {
        /* Slots 0 - 2 reserved for CpuGemmAssemblyDispatch */
        InterleavedLHS = 3,
        PreTransposedRHS,
        Transposed1xWRHS,
        TempResult,
        Count
    };

    TensorInfo _tmp_a{};
    TensorInfo _tmp_b{};
    TensorInfo _tmp_d{};

    bool _run_vector_matrix_multiplication{false};
    bool _run_bias_addition{false};
    bool _reshape_b_only_on_first_run{false};
    bool _run_interleave_transpose{
        true}; /**< If we run CpuGemmInterleave4x4Kernel on lhs and CpuGemmTranspose1xWKernel on rhs */

    std::unique_ptr<kernels::CpuGemmMatrixMultiplyKernel> _mm_kernel{nullptr};
    std::unique_ptr<kernels::CpuGemmInterleave4x4Kernel>  _interleave_kernel{nullptr};
    std::unique_ptr<kernels::CpuGemmTranspose1xWKernel>   _transpose1xW_b_kernel{nullptr};

    experimental::MemoryRequirements _aux_mem{Count};
};

} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_LINEAR_H */
