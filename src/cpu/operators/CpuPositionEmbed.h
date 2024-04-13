#ifndef ARM_COMPUTE_CPU_POSITION_EMBED_H
#define ARM_COMPUTE_CPU_POSITION_EMBED_H


#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"


namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuEmbedKernel */
class CpuPositionEmbed : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  input           Source tensor info. Data types supported: U8.
     * @param[in]  Position         Const Position vector, Data type supported: F32
     * @param[out] output          Destination tensor info. Data type supported: F32
     */
    void configure(const ITensorInfo *input, const ITensorInfo *Position, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuPositionEmbed::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *Position, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_POSITION_EMBED_H */
