#ifndef ARM_COMPUTE_CPU_TOKEN_EMBED_H
#define ARM_COMPUTE_CPU_TOKEN_EMBED_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuActivationKernel */
class CpuTokenEmbed : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in]  input           Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out] output          Destination tensor info. Data type supported: same as @p src
     */
    void configure(const ITensorInfo *input, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuActivation::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ACTIVATION_H */
