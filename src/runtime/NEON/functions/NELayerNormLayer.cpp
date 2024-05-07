#include "arm_compute/runtime/NEON/functions/NELayerNormLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuLayerNorm.h"

namespace arm_compute
{

struct  NELayerNormLayer::Impl
{
    const ITensor                       *src{nullptr};
    ITensor                             *dst{nullptr};
    std::unique_ptr<cpu::CpuLayerNorm>  op{nullptr};
};

NELayerNormLayer::NELayerNormLayer() : _impl(std::make_unique<Impl>())
{
}
NELayerNormLayer::~NELayerNormLayer() = default;

void NELayerNormLayer::configure(const ITensor *input,
                              ITensor *output,
                              const LayerNormLayerInfo& LayerNorm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output);

    _impl->src      = input;
    _impl->dst      = output;

    _impl->op = std::make_unique<cpu::CpuLayerNorm>();
    _impl->op->configure(input->info(), output->info(), LayerNorm_info);
}

Status NELayerNormLayer::validate(const ITensor *input,
                                  ITensor *output, 
                                  const LayerNormLayerInfo& LayerNorm_info)
{
    ARM_COMPUTE_UNUSED(LayerNorm_info);
    return cpu::CpuLayerNorm::validate(input->info(), output->info(), LayerNorm_info);
}

void NELayerNormLayer::run()
{
    ITensorPack pack;

    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    
    _impl->op->run(pack);

}

} // namespace arm_compute
