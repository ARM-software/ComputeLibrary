#include "arm_compute/runtime/NEON/functions/NESegmentEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuSegmentEmbed.h"

namespace arm_compute
{

struct NESegmentEmbeddingLayer::Impl
{
    const ITensor                      *src{nullptr};
    const ITensor                      *segment{nullptr};
    ITensor                            *dst{nullptr};
    IRuntimeContext                    *ctx{nullptr};
    std::unique_ptr<cpu::CpuSegmentEmbed> op{nullptr};
};

NESegmentEmbeddingLayer::NESegmentEmbeddingLayer(): _impl(std::make_unique<Impl>())
{
}

NESegmentEmbeddingLayer::~NESegmentEmbeddingLayer() = default;

void NESegmentEmbeddingLayer::configure(ITensor *input, ITensor *segment, ITensor *output)
{
    _impl->src      = input;
    _impl->segment    = segment;
    _impl->dst      = output;

    _impl->op  = std::make_unique<cpu::CpuSegmentEmbed>();
    _impl->op->configure(_impl->src->info(),_impl->segment->info(), _impl->dst->info(),TokenEmbeddingLayerInfo(768U,2U));
}

void NESegmentEmbeddingLayer::prepare()
{
}

void NESegmentEmbeddingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->segment);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

} // namespace arm_compute