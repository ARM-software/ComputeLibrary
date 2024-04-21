#ifndef ARM_COMPUTE_CPU_EMBED_SUM_H
#define ARM_COMPUTE_CPU_EMBED_SUM_H

#include "src/cpu/ICpuOperator.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/kernels/CpuAddKernel.h"


namespace arm_compute
{
namespace cpu
{
/** A function use @ref kernels::CpuAddKernel to sum 3 embedding output*/
class CpuEmbedSum : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  token        Token embedding input, Data type supported: F32
     * @param[in]  segemnt      Token embedding input, Data type supported: F32
     * @param[in]  position     Token embedding input, Data type supported: F32
     * @param[out] output       Destination tensor info. Data type supported: F32
     * @param[in]  emb_info     Embedding layer parameters.
     */
    void configure(const ITensorInfo *token,
                   const ITensorInfo *segemnt,
                   const ITensorInfo *position,
                   ITensorInfo *output, 
                   const EmbeddingLayerInfo &emb_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuEmbedSum::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *token,
                           const ITensorInfo *segemnt,
                           const ITensorInfo *position,
                           ITensorInfo *output, 
                           const EmbeddingLayerInfo &emb_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
private:
    std::unique_ptr<kernels::CpuAddKernel> _add_kernel_1{nullptr};
    std::unique_ptr<kernels::CpuAddKernel> _add_kernel_2{nullptr};
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_EMBED_SUM_H */
