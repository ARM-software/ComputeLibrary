#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuTokenEmbed.h"
// operator to be added 

namespace arm_compute
{

struct NETokenEmbeddingLayer::Impl
{
    const ITensor                      *src{nullptr};
    ITensor                            *dst{nullptr};
    IRuntimeContext                    *ctx{nullptr};
    std::unique_ptr<cpu::CpuTokenEmbed> op{nullptr};
};

NETokenEmbeddingLayer::NETokenEmbeddingLayer(): _impl(std::make_unique<Impl>())
{
}

NETokenEmbeddingLayer::~NETokenEmbeddingLayer() = default;

void NETokenEmbeddingLayer::configure(ITensor *input, ITensor *output, TokenEmbeddingLayerInfo tkemb_info)
{
    _impl->src = input;
    _impl->dst = output == nullptr ? input : output;
    _impl->op  = std::make_unique<cpu::CpuTokenEmbed>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info(),tkemb_info);
}


void NETokenEmbeddingLayer::run()
{
    std::cout << " NETokenEmbeddingLayer::run!!!!!!!!!!!!!!!  " << std::endl;
    std::cout << _impl->src->info()->tensor_shape().total_size() << std::endl;
    std::cout << _impl->dst->info()->tensor_shape().total_size() << std::endl;
}

} // namespace arm_compute