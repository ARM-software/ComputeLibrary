#ifndef ARM_COMPUTE_CPU_SIMPLE_FORWARD_H
#define ARM_COMPUTE_CPU_SIMPLE_FORWARD_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuSimpleForwardKernel */
class CpuSimpleForward : public ICpuOperator
{
public:
    /** Configure kernel for a given list of arguments
     *
     * @param[in]  src Srouce tensor to copy. Data types supported: All
     * @param[out] dst Destination tensor. Data types supported: Same as @p src
     */
    void configure(const ITensorInfo *src1,
                   const ITensorInfo *src2,
                   const ITensorInfo *src3,
                   ITensorInfo *dst1,
                   ITensorInfo *dst2,
                   ITensorInfo *dst3);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SIMPLE_FORWARD_H */
