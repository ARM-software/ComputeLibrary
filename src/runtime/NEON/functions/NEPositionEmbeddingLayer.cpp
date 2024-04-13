#include "arm_compute/runtime/NEON/functions/NEPositionEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuPositionEmbed.h"

namespace arm_compute
{

struct NEPositionEmbeddingLayer::Impl
{
    const ITensor                      *src{nullptr};
    const ITensor                      *position{nullptr};
    ITensor                            *dst{nullptr};
    IRuntimeContext                    *ctx{nullptr};
    std::unique_ptr<cpu::CpuPositionEmbed> op{nullptr};
};

NEPositionEmbeddingLayer::NEPositionEmbeddingLayer(): _impl(std::make_unique<Impl>())
{
}

NEPositionEmbeddingLayer::~NEPositionEmbeddingLayer() = default;

void NEPositionEmbeddingLayer::configure(ITensor *input, ITensor *position, ITensor *output)
{
    _impl->src      = input;
    _impl->position = position;
    _impl->dst      = output;

    _impl->op  = std::make_unique<cpu::CpuPositionEmbed>();
    _impl->op->configure(_impl->src->info(),_impl->position->info(), _impl->dst->info());
}

void NEPositionEmbeddingLayer::prepare()
{
}

void NEPositionEmbeddingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->Position);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

} // namespace arm_compute