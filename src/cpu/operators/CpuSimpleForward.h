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
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  tensors Tensor packs contains input and output. Data type supported: All.
     */
    void configure(unsigned int total_nodes);
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SIMPLE_FORWARD_H */
