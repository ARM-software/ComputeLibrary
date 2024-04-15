#ifndef ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H
#define ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuScaleKernel.h"
#include "src/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/cpu/operators/CpuTranspose.h"
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"
#include "src/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/cpu/kernels/CpuGemmTranspose1xWKernel.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
/** Function implementation for scale dot production, uses kernels:
 * @ref kernels::CpuEmbedKernel
*/
class CpuScaleDotProduction : public ICpuOperator
{
public:
    /** Constructor */
    CpuScaleDotProduction() = default;
    /** Destructor */
    ~CpuScaleDotProduction() = default;
    
    /** Configure operator for a given list of arguments
     *
     * @param[in]  key             Attention key tensor info. Data types supported: U8.
     * @param[in]  value           Attention value tensor info. Data types supported: U8.
     * @param[in]  query           Attention key tensor info. Data types supported: U8.
     * @param[out] output          Destination tensor info. Data type supported: F32
     */
    void configure(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuScaleDotProduction::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output);

    void transpose(ITensorPack &tensors);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        /* Slots 0 - 2 reserved for CpuGemmAssemblyDispatch */
        InterleavedLHS = 3,
        PreTransposedRHS,
        Transposed1xWRHS,
        Count
    };

    std::unique_ptr<kernels::CpuGemmInterleave4x4Kernel>    _interleave_kernel{nullptr};
    std::unique_ptr<CpuTranspose>                           _pretranspose_key_func{nullptr};
    std::unique_ptr<kernels::CpuGemmMatrixMultiplyKernel>   _mm_kernel{nullptr};
    std::unique_ptr<kernels::CpuGemmTranspose1xWKernel>     _transpose1xW_key_kernel{nullptr};

    TensorInfo _tmp_query{};
    TensorInfo _pretransposed_key{};
    TensorInfo _tmp_key{};

    bool _run_pretranspose{false};
    bool _run_vector_matrix_multiplication{false};
    bool _run_interleave_transpose{
        true}; /**< If we run CpuGemmInterleave4x4Kernel on lhs and CpuGemmTranspose1xWKernel on rhs */

    experimental::MemoryRequirements _aux_mem{Count};

};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H */
