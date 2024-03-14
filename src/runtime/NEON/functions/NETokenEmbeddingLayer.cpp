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
};


NETokenEmbeddingLayer::NETokenEmbeddingLayer(IRuntimeContext *ctx) : _impl(std::make_unique<Impl>())
{
    _impl->ctx = ctx;
}

NETokenEmbeddingLayer::~NETokenEmbeddingLayer() = default;

void NETokenEmbeddingLayer::configure(ITensor *input, ITensor *output, TokenEmbeddingLayerInfo tkemb_info)
{
        std::cout << " NETokenEmbeddingLayer::configure!!!!!!!!!!!!!!!  " << std::endl;
}

Status
NETokenEmbeddingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const TokenEmbeddingLayerInfo &tkemb_info)
{
    std::cout << "NETokenEmbeddingLayer::validate" << std::endl;
    std::cout << tkemb_info.d_model() << std::endl;
    return cpu::CpuTokenEmbed::validate(input, output);
}

void NETokenEmbeddingLayer::run()
{
    std::cout << " NETokenEmbeddingLayer::run!!!!!!!!!!!!!!!  " << std::endl;
}

} // namespace arm_compute