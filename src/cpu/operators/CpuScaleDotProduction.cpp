#include "src/cpu/operators/CpuScaleDotProduction.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"
#include "src/cpu/kernels/CpuVectorizeKernel.h"


namespace arm_compute
{
namespace cpu
{
void CpuScaleDotProduction::configure(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);

    ARM_COMPUTE_UNUSED(key);
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);

}

Status
CpuScaleDotProduction::validate(const ITensorInfo *key, const ITensorInfo *value, const ITensorInfo *query, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(key);
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);
    return Status{};
}

void CpuScaleDotProduction::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto split_dimension = static_cast<kernels::CpuVectorizeKernel *>(_kernel.get())->get_split_dimension_hint();

    ARM_COMPUTE_UNUSED(split_dimension);
    ARM_COMPUTE_UNUSED(tensors);
    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp" << std::endl;


    //NEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
}


} // namespace cpu
} // namespace arm_compute
