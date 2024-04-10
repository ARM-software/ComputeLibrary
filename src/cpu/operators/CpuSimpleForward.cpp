#include "src/cpu/operators/CpuSimpleForward.h"

#include "src/common/utils/Log.h"
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
}

} // namespace cpu
} // namespace arm_compute
