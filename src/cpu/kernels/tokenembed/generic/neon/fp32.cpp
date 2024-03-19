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
    const unsigned int window_start_x   = static_cast<unsigned int>(window.x().start());
    const unsigned int window_end_x     = src->info()->tensor_shape().x();
    unsigned int       x                = window_start_x;

    std::cout << "Tensor shape" << std::endl;
    std::cout << dst->info()->tensor_shape().x() << std::endl;
    std::cout << dst->info()->tensor_shape().y() << std::endl;
    
    Iterator src_iter(src,win);
    Iterator vocab_iter(vocab,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    const auto vocab_ptr    = reinterpret_cast<float *>(vocab_iter.ptr());

    std::cout << "YeaHhhhhhhhhhhh " << std::endl;
    execute_window_loop(win,
        [&](const Coordinates &){
            for(; x < window_end_x; x++){
                std::cout << *(src_ptr+x) << std::endl;
            }
            std::cout << *(vocab_ptr) << std::endl;
        },vocab_iter);

}

} // namespace cpu
} // namespace arm_compute
