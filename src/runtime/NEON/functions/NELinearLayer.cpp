#include "arm_compute/runtime/NEON/functions/NELinearLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuLinear.h"

namespace arm_compute
{

struct  NELinearLayer::Impl
{
    const ITensor                      *src{nullptr};
    const ITensor                      *weight{nullptr};
    const ITensor                      *bias{nullptr};
    ITensor                            *dst{nullptr};
    std::unique_ptr<cpu::CpuLinear>    kernel{nullptr};
};

NELinearLayer::NELinearLayer() : _impl(std::make_unique<Impl>())
{
}
NELinearLayer::~NELinearLayer() = default;

void NELinearLayer::configure(const ITensor *input1, 
                              const ITensor *weight, 
                              const ITensor *bias, ITensor *output, const LinearLayerInfo& linear_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, output);
    ARM_COMPUTE_LOG_PARAMS(input1, output);

    std::cout << weight->info()->tensor_shape().x() << std::endl;
    std::cout << weight->info()->tensor_shape().y() << std::endl;
    std::cout << bias->info()->tensor_shape().x() << std::endl;
    std::cout << bias->info()->tensor_shape().y() << std::endl;

    _impl->src      = input1;
    _impl->weight   = weight;
    _impl->bias     = bias;
    _impl->dst      = output;

    _impl->kernel = std::make_unique<cpu::CpuLinear>();
    _impl->kernel->configure(input1->info(), output->info(), linear_info);
}

Status NELinearLayer::validate(const ITensorInfo *input1, const ITensorInfo *output, const LinearLayerInfo& linear_info)
{
    return cpu::CpuLinear::validate(input1,output,linear_info);
}

void NELinearLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weight);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->bias);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    
    _impl->kernel->run(pack);
}

} // namespace arm_compute
