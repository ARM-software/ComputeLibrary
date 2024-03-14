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

void NETokenEmbeddingLayer::configure(const ITensor     *input,
                               ITensor           *output,
                               const Coordinates &starts,
                               const Coordinates &ends,
                               const BiStrides   &strides,
                               int32_t            begin_mask,
                               int32_t            end_mask,
                               int32_t            shrink_axis_mask)
{
    _impl->src = input;
    _impl->dst = output;
}

void NETokenEmbeddingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
}

Status NETokenEmbeddingLayer::validate(const ITensorInfo *input,
                                const ITensorInfo *output,
                                const Coordinates &starts,
                                const Coordinates &ends,
                                const BiStrides   &strides,
                                int32_t            begin_mask,
                                int32_t            end_mask,
                                int32_t            shrink_axis_mask)
{
    return NETokenEmbeddingLayer::validate(input, output, starts, ends, strides, begin_mask, end_mask,
                                                  shrink_axis_mask);
}
} // namespace arm_compute
