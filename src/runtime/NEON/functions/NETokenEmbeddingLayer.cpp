#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
struct NETokenEmbeddingLayer::Impl
{
    const ITensor                                *src{nullptr};
    ITensor                                      *dst{nullptr};
};

NETokenEmbeddingLayer::NETokenEmbeddingLayer() : _impl(std::make_unique<Impl>())
{
}
NETokenEmbeddingLayer::NETokenEmbeddingLayer(NETokenEmbeddingLayer &&)            = default;
NETokenEmbeddingLayer &NETokenEmbeddingLayer::operator=(NETokenEmbeddingLayer &&) = default;
NETokenEmbeddingLayer::~NETokenEmbeddingLayer()                            = default;

void NETokenEmbeddingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
}

} // namespace arm_compute
