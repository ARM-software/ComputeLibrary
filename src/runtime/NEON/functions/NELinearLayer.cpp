#include "arm_compute/runtime/NEON/functions/NELinearLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuGemm.h"

namespace arm_compute
{

struct  NELinearLayer::Impl
{
    const ITensor                      *src{nullptr};
    const ITensor                      *weight{nullptr};
    const ITensor                      *bias{nullptr};
    ITensor                            *dst{nullptr};
    std::unique_ptr<cpu::CpuGemm>    kernel{nullptr};
};

NELinearLayer::NELinearLayer() : _impl(std::make_unique<Impl>())
{
}
NELinearLayer::~NELinearLayer() = default;

void NELinearLayer::configure(const ITensor *input, 
                              const ITensor *weight, 
                              const ITensor *bias, ITensor *output, const LinearLayerInfo& linear_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output);
    ARM_COMPUTE_UNUSED(linear_info);

    _impl->src      = input;
    _impl->weight   = weight;
    _impl->bias     = bias;
    _impl->dst      = output;

    _impl->kernel = std::make_unique<cpu::CpuGemm>();
    _impl->kernel->configure(input->info(), weight->info(), bias->info(), output->info(), 1.0f, 1.0f);
}

Status NELinearLayer::validate(const ITensor *input, 
                              const ITensor *weight, 
                              const ITensor *bias, ITensor *output, const LinearLayerInfo& linear_info)
{
    ARM_COMPUTE_UNUSED(linear_info);
    return cpu::CpuGemm::validate(input->info(), weight->info(), bias->info(), output->info(), 1.0f, 1.0f);
}

void NELinearLayer::run()
{
    ITensorPack pack;

    std::cout << "src/runtime/NEON/functions/NELinearLayer.cpp" << std::endl;
    std::cout << "src " << std::endl;
    std::cout << _impl->src->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->src->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->src->info()->tensor_shape().z() << std::endl;
    
    std::cout << "weight " << std::endl;
    std::cout << _impl->weight->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->weight->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->weight->info()->tensor_shape().z() << std::endl;

    std::cout << "bias " << std::endl;
    std::cout << _impl->bias->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->bias->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->bias->info()->tensor_shape().z() << std::endl;

    std::cout << "dst " << std::endl;
    std::cout << _impl->dst->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->dst->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->dst->info()->tensor_shape().z() << std::endl;


    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weight);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->bias);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    
    _impl->kernel->run(pack);

    std::cout << "*******    " << std::endl;
    std::cout << "*******    " << std::endl;
    std::cout << "src " << std::endl;
    std::cout << _impl->src->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->src->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->src->info()->tensor_shape().z() << std::endl;
    
    std::cout << "weight " << std::endl;
    std::cout << _impl->weight->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->weight->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->weight->info()->tensor_shape().z() << std::endl;

    std::cout << "bias " << std::endl;
    std::cout << _impl->bias->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->bias->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->bias->info()->tensor_shape().z() << std::endl;

    std::cout << "dst " << std::endl;
    std::cout << _impl->dst->info()->tensor_shape().x() << std::endl;
    std::cout << _impl->dst->info()->tensor_shape().y() << std::endl;
    std::cout << _impl->dst->info()->tensor_shape().z() << std::endl;
}

} // namespace arm_compute
