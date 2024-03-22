#include "src/cpu/operators/CpuSimpleForward.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/cpu/kernels/CpuSimpleForwardKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuSimpleForward::configure(unsigned int total_nodes)
{
    ARM_COMPUTE_LOG_PARAMS(total_nodes);
    auto k = std::make_unique<cpu::CpuSimpleForward>();
    k->configure(total_nodes);
    _kernel = std::move(k);
}

void CpuSimpleForward::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}


} // namespace cpu
} // namespace arm_compute
