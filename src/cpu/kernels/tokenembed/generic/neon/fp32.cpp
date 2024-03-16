#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{
void neon_token_embed_char_2_float32(const ITensor *src, const ITensor *vocab, ITensor *dst, const TokenEmbeddingLayerInfo &tkemb_info, const Window &window)
{
    std::cout << "src/cpu/kernels/tokenembed/generic/neon/fp32.cpp: neon_token_embed_char_2_float32" << std::endl;

    std::cout << src->info()->id() << std::endl;
    std::cout << vocab->info()->id() << std::endl;
    std::cout << dst->info()->id() << std::endl;
    std::cout << tkemb_info.d_vocab() << std::endl;
    std::cout << window.DimX << std::endl;

}

} // namespace cpu
} // namespace arm_compute
