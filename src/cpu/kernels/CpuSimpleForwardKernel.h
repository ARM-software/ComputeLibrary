#ifndef SRC_CPU_KERNELS_CPU_SIMPLEFORWARDKERNEL_H
#define SRC_CPU_KERNELS_CPU_SIMPLEFORWARDKERNEL_H

#include "src/core/common/Macros.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

/** Simply forward input tensors to output tensors*/
class CpuSimpleForwardKernel : public ICpuKernel<CpuSimpleForwardKernel>
{

public:
    CpuSimpleForwardKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuSimpleForwardKernel);

    /** Set the amount of input/output pair
     *
     * @param[in]  total_nodes The amount of nodes input need to be forward
     */
    void configure(unsigned int total_nodes);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuSimpleForwardKernel::configure()
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ConvertPolicy policy);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    std::string   _name{};
    unsigned int _total_nodes;
};

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* SRC_CPU_KERNELS_CPU_SIMPLEFORWARDKERNEL_H */
