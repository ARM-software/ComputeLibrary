#ifndef ARM_COMPUTE_NESIMPLEFORWARDKERNEL_H
#define ARM_COMPUTE_NESIMPLEFORWARDKERNEL_H

#include "src/core/KernelTypes.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
namespace kernels
{
/** Simply forward input tensor to output tensor*/
class NESimpleForwardKernel : public INEKernel
{
public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  tensors Tensor packs contains input and output. Data type supported: All.
     */
    void configure(ITensorPack& tensors, unsigned int total_nodes);

    // Inherited methods overridden:
    void        run(const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    ITensorPack& _tensors;
    unsigned int _total_nodes;
};
} // namespace kernels
} // namespace arm_compute
#endif /* ARM_COMPUTE_NESIMPLEFORWARDKERNEL_H */
