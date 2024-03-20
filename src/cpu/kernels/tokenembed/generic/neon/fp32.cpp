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

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0,1,1));
    win.set(Window::DimY, Window::Dimension(0,1,1));
    const unsigned int window_start_x   = static_cast<unsigned int>(window.x().start());
    const unsigned int window_end_x     = src->info()->tensor_shape().x();
    unsigned int       x                = window_start_x;

    const unsigned int vector_depth     = tkemb_info.d_model();

    unsigned int id_src,offset_vocab, offset_dst;
    
    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);
    Iterator vocab_iter(vocab,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    const auto dst_ptr      = reinterpret_cast<float *>(dst_iter.ptr());
    const auto vocab_ptr    = reinterpret_cast<float *>(vocab_iter.ptr());

    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(; x < window_end_x; x++)
            {
                id_src = *(src_ptr+x);
                std::cout << id_src << std::endl;

                offset_dst      = x * vector_depth;
                offset_vocab    = id_src * vector_depth;

                std::memcpy(dst_ptr + offset_dst, vocab_ptr + offset_vocab, (vector_depth) * sizeof(*vocab_ptr));

                std::cout << *(dst_ptr + offset_dst) << std::endl;
                std::cout << *(dst_ptr + offset_dst + dst->info()->tensor_shape().y()-1) << std::endl;

            }
        },vocab_iter,src_iter,vocab_iter);

}

} // namespace cpu
} // namespace arm_compute
