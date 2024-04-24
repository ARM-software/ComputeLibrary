#include "src/cpu/operators/CpuLayerNorm.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/cpu/kernels/CpuLayerNormKernel.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

namespace arm_compute
{
namespace cpu
{
void CpuLayerNorm::configure(const ITensorInfo *input,
                          ITensorInfo       *output,
                          const LayerNormLayerInfo &info)
{
    _layer_norm_kernel = std::make_unique<kernels::CpuLayerNormKernel>();
    _layer_norm_kernel ->configure(input,output,info);
}

Status
CpuLayerNorm::validate(const ITensorInfo *input,
                    ITensorInfo       *output,
                    const LayerNormLayerInfo &info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(info);
    return Status{};
}

void CpuLayerNorm::run(ITensorPack &tensors)
{
    NEScheduler::get().schedule_op(_layer_norm_kernel.get(), Window::DimY,
                                            _layer_norm_kernel->window(), tensors);
}


} // namespace cpu
} // namespace arm_compute
