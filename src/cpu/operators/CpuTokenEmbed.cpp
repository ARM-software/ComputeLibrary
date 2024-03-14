#include "src/cpu/operators/CpuTokenEmbed.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"

namespace arm_compute
{
namespace cpu
{
void CpuTokenEmbed::configure(const ITensorInfo *input, ITensorInfo *output)
{
    std::cout<< "CpuTokenEmbed::configure" << std::endl;
    std::cout<< input->tensor_shape().total_size() << std::endl;
    std::cout<< output->tensor_shape().total_size()  << std::endl;
    
}

Status
CpuTokenEmbed::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return Status{};
}

void CpuTokenEmbed::run(ITensorPack &tensors)
{
    std::cout<< "CpuTokenEmbed::run" << std::endl;
    std::cout<< tensors.size() << std::endl;
}


} // namespace cpu
} // namespace arm_compute
