#include "src/cpu/operators/CpuSimpleForward.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/common/utils/Log.h"

#include "src/cpu/kernels/CpuSimpleForwardKernel.h"

#include "src/core/helpers/AutoConfiguration.h"

namespace arm_compute
{
namespace cpu
{
void CpuSimpleForward::configure(const ITensorInfo *src1,
                                 const ITensorInfo *src2,
                                 const ITensorInfo *src3,
                                 ITensorInfo *dst1,
                                 ITensorInfo *dst2,
                                 ITensorInfo *dst3)
{   
    auto k = std::make_unique<kernels::CpuSimpleForwardKernel>();
    k->configure(src1, src2, src3, dst1, dst2, dst3);
    _kernel = std::move(k);
}

void CpuSimpleForward::run(ITensorPack &tensors)
{
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}

} // namespace cpu
} // namespace arm_compute
