#ifndef ARM_COMPUTE_CPU_SEGMENT_EMBED_H
#define ARM_COMPUTE_CPU_SEGMENT_EMBED_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuPositionalEncodingKernel.h"


namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuEmbedKernel */
class CpuSegmentEmbed : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  input           Source tensor info. Data types supported: U8.
     * @param[in]  segment         Const segment vector, Data type supported: F32
     * @param[out] output          Destination tensor info. Data type supported: F32
     * @param[in]  tkemb_info      Segment embed layer parameters.
     */
    void configure(const ITensorInfo *input, const ITensorInfo *segment, ITensorInfo *output, const TokenEmbeddingLayerInfo &tkemb_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuSegmentEmbed::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *segment, const ITensorInfo *output, const TokenEmbeddingLayerInfo &tkemb_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SEGMENT_EMBED_H */
