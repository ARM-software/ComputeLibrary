#include "arm_compute/runtime/NEON/functions/NELinearLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/NELinearLayerKernel.h"

namespace arm_compute
{
struct LogicalArgs
{
    std::unique_ptr<kernels::NELinearLayerKernel> kernel{nullptr};
    ITensorPack                                   pack{};
};

struct NELinearLayer::Impl : public LogicalArgs
{
};
NELinearLayer::NELinearLayer() : _impl(std::make_unique<Impl>())
{
}
NELinearLayer::~NELinearLayer() = default;

void NELinearLayer::configure(const ITensor *input1, ITensor *output, LinearLayerInfo linear_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, output);
    ARM_COMPUTE_LOG_PARAMS(input1, output);

    _impl->kernel = std::make_unique<kernels::NELinearLayerKernel>();
    _impl->kernel->configure(input1->info(), output->info(), LinearAttentionOperation::Key);

    _impl->pack = ITensorPack();
    _impl->pack.add_tensor(TensorType::ACL_SRC_0, input1);
    _impl->pack.add_tensor(TensorType::ACL_DST, output);
    
}

Status NELinearLayer::validate(const ITensorInfo *input1, const ITensorInfo *output)
{
    return kernels::NELinearLayerKernel::validate(input1, output, LinearAttentionOperation::Query);
}

void NELinearLayer::run()
{
    NEScheduler::get().schedule_op(_impl->kernel.get(), Window::DimY, _impl->kernel->window(), _impl->pack);
}

} // namespace arm_compute
