#include "src/cpu/operators/CpuLinear.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/cpu/kernels/CpuLinearKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuLinear::configure(const ITensorInfo *input,  ITensorInfo *output, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, linear_info);
    auto k = std::make_unique<kernels::CpuLinearKernel>();
    k->configure(input, output, linear_info);
    _kernel = std::move(k);
    
}

Status
CpuLinear::validate(const ITensorInfo *input, const ITensorInfo *output, const LinearLayerInfo &linear_info)
{
    std::cout << "src/cpu/operators/CpuLinear.cpp: validate " << std::endl;
    std::cout << input->id() << std::endl;
    std::cout << output->id() << std::endl;
    std::cout << linear_info.d_model() << std::endl;
    return Status{};
}

void CpuLinear::run(ITensorPack &tensors)
{
    std::cout << "src/runtime/NEON/functions/NELinearLayer.cpp run" << std::endl;
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}


} // namespace cpu
} // namespace arm_compute
