#ifndef ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H
#define ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuScaleKernel.h"
#include "src/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/cpu/operators/CpuTranspose.h"
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"


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

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &tensors) override;

private:
    enum AuxTensorIdx
    {
        KeyTransposeBuffer = 0,
        Count
    };
    std::unique_ptr<kernels::CpuGemmMatrixMultiplyKernel> _mm_kernel{nullptr};
    std::unique_ptr<CpuTranspose>                         _t_func{nullptr};
    std::unique_ptr<kernels::CpuScaleKernel>              _s_kernel{nullptr};
    std::unique_ptr<kernels::CpuSoftmaxKernel>            _sm_kernel{nullptr};

    TensorInfo _buffer_t_info{};
    bool _is_prepared{false};

};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SCALE_DOT_PRODUCTION_H */
