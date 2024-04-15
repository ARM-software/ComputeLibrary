#include "src/cpu/operators/CpuScaleDotProduction.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"


namespace arm_compute
{
namespace cpu
{

void CpuScaleDotProduction::configure(const ITensorInfo *key,
                                      const ITensorInfo *value,
                                      const ITensorInfo *query,
                                      ITensorInfo *output)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);

    /* Pretranspose Key, K=K^T*/
    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp" << std::endl;
     std::cout << "->tensor_shape().x(): " << key->tensor_shape().x() << std::endl
              << "a->tensor_shape().y(): " << key->tensor_shape().y() << std::endl
              << "a->tensor_shape().z(): " << key->tensor_shape().z() << std::endl
              << "b->tensor_shape().x(): " << value->tensor_shape().x() << std::endl
              << "b->tensor_shape().y(): " << value->tensor_shape().y() << std::endl
              << "b->tensor_shape().z(): " << value->tensor_shape().z() << std::endl
              << "c->tensor_shape().x(): " << query->tensor_shape().x() << std::endl
              << "c->tensor_shape().y(): " << query->tensor_shape().y() << std::endl
              << "c->tensor_shape().z(): " << query->tensor_shape().z() << std::endl
              << "d->tensor_shape().x(): " << output->tensor_shape().x() << std::endl
              << "d->tensor_shape().y(): " << output->tensor_shape().y() << std::endl
              << "d->tensor_shape().z(): " << output->tensor_shape().z() << std::endl
            << std::endl;
    std::cout << "src/cpu/operators/CpuScaleDotProduction.cpp" << std::endl;

    /* Matrix multiply Query adn Key, QK */


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
    ARM_COMPUTE_UNUSED(tensors);
}

experimental::MemoryRequirements CpuScaleDotProduction::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
