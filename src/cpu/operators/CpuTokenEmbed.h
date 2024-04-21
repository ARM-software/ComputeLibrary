#ifndef ARM_COMPUTE_CPU_TOKEN_EMBED_H
#define ARM_COMPUTE_CPU_TOKEN_EMBED_H

#include "src/cpu/ICpuOperator.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"


namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuVectorizeKernel */
class CpuTokenEmbed : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  input           Source tensor info. Data types supported: U8.
     * @param[in]  vocab           Char 2 Vec const tensor info, Data type supported: F32
     * @param[out] output          Destination tensor info. Data type supported: F32
     * @param[in]  tkemb_info      Token embed layer parameters.
     */
    void configure(const ITensorInfo *input, const ITensorInfo *vocab, ITensorInfo *output, const EmbeddingLayerInfo &tkemb_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuTokenEmbed::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *vocab, const ITensorInfo *output, const EmbeddingLayerInfo &tkemb_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
private: 
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_TOKEN_EMBED_H */
