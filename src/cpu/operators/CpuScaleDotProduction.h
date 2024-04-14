#ifndef ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H
#define ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuScaleKernel.h"
#include "src/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/cpu/operators/CpuTranspose.h"
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"

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
        TempResult,
        Count
    };

    std::unique_ptr<kernels::CpuGemmMatrixMultiplyKernel> _mm_kernel{nullptr};
    std::unique_ptr<CpuTranspose>                         _pretranspose_b_func{nullptr};
    std::unique_ptr<kernels::CpuScaleKernel>              _s_kernel{nullptr};
    std::unique_ptr<kernels::CpuSoftmaxKernel>            _sm_kernel{nullptr};

    TensorInfo _pretransposed_b{};

    bool _is_prepared{false};
    bool _reshape_b_only_on_first_run{false};

    experimental::MemoryRequirements _aux_mem{Count};

};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H */
