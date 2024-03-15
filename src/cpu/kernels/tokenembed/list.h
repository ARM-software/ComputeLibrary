#ifndef SRC_CPU_KERNELS_TOKEN_EMBED_LIST_H
#define SRC_CPU_KERNELS_TOKEN_EMBED_LIST_H

namespace arm_compute
{
namespace cpu
{
#define DECLARE_TOKEN_EMBED_KERNEL(func_name) \
    void func_name(const ITensor *src, ITensor *dst, const TokenEmbeddingLayerInfo &tkemb_info, const Window &window)

#ifdef __aarch64__
DECLARE_TOKEN_EMBED_KERNEL(neon_token_embed_char_2_float32);
#endif // __aarch64__

#undef DECLARE_ACTIVATION_KERNEL
} // namespace cpu
} // namespace arm_compute

#endif // SRC_CPU_KERNELS_TOKEN_EMBED_LIST_H

