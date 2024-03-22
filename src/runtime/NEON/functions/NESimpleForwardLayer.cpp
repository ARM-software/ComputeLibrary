#include "arm_compute/runtime/NEON/functions/NESimpleForwardLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuSimpleForward.h"

namespace arm_compute
{

struct  NESimpleForwardLayer::Impl
{
    ITensorPack                                         tensors{};
    std::unique_ptr<cpu::CpuSimpleForward>     kernel{nullptr};
};

NESimpleForwardLayer::NESimpleForwardLayer() : _impl(std::make_unique<Impl>())
{
}
NESimpleForwardLayer::~NESimpleForwardLayer() = default;

void NESimpleForwardLayer::configure(ITensorPack& tensors, unsigned int total_nodes)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensors);
    ARM_COMPUTE_LOG_PARAMS(tensors);

    _impl->tensors      = tensors;

    _impl->kernel = std::make_unique<cpu::CpuSimpleForward>();
    _impl->kernel->configure(total_nodes);
}

void NESimpleForwardLayer::run()
{
    _impl->kernel->run(_impl->tensors);
}

} // namespace arm_compute
