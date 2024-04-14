#ifndef SRC_CPU_KERNELS_CPU_SIMPLEFORWARDKERNEL_H
#define SRC_CPU_KERNELS_CPU_SIMPLEFORWARDKERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

/** Kernel which transposes the elements of a matrix */
class CpuSimpleForwardKernel : public ICpuKernel<CpuSimpleForwardKernel>
{
public:
    CpuSimpleForwardKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuSimpleForwardKernel);
    /** Set the amount of input/output pair
     *
     * @param[in]  total_nodes The amount of nodes input need to be forward
     */
    void configure(const ITensorInfo *src1,
                   const ITensorInfo *src2,
                   const ITensorInfo *src3,
                   ITensorInfo *dst1,
                   ITensorInfo *dst2,
                   ITensorInfo *dst3);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;
private:
};

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* SRC_CPU_KERNELS_CPU_SIMPLEFORWARDKERNEL_H */
