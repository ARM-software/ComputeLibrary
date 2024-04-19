#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{
void neon_vectorize_int_2_float32(const ITensor *src, const ITensor *vector, ITensor *dst, const Window &window)
{
    std::cout << " src/cpu/kernels/vectorize/generic/neon/fp32.cpp Window: " << std::endl;
    std::cout << "window.x start " << window.x().start() << " end " << window.x().end() << " step " << window.x().step() << std::endl; 
    std::cout << "window.y start " << window.y().start() << " end " << window.y().end() << " step " << window.y().step() << std::endl; 
    std::cout << "window.z start " << window.z().start() << " end " << window.z().end() << " step " << window.z().step() << std::endl; 
    
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0,1,1));
    win.set(Window::DimY, Window::Dimension(0,1,1));
    const unsigned int window_start_x   = static_cast<unsigned int>(window.x().start());
    const unsigned int window_end_x     = static_cast<unsigned int>(window.x().end());

    const unsigned int vector_depth     = vector->info()->tensor_shape().y();

    unsigned int offset_vector,offset_dst;

    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);
    Iterator vector_iter(vector,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    const auto dst_ptr      = reinterpret_cast<float *>(dst_iter.ptr());
    const auto vector_ptr   = reinterpret_cast<float *>(vector_iter.ptr());
    std::cout << " src/cpu/kernels/vectorize/generic/neon/fp32.cpp win:" << std::endl;
    std::cout << "window.x start " << win.x().start() << " end " << win.x().end() << " step " << win.x().step() << std::endl; 
    std::cout << "window.y start " << win.y().start() << " end " << win.y().end() << " step " << win.y().step() << std::endl; 
    std::cout << "window.z start " << win.z().start() << " end " << win.z().end() << " step " << win.z().step() << std::endl; 
    
    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(unsigned int x = window_start_x; x < window_end_x; x++)
            {
                offset_dst     = x * vector_depth;
                offset_vector  = *(src_ptr+x) * vector_depth;
                std::memcpy(dst_ptr + offset_dst, vector_ptr + offset_vector, (vector_depth) * sizeof(*vector_ptr));
                std::cout<< "x:  " << x <<":  " 
                << *(dst_ptr + offset_dst)<< " " 
                << *(dst_ptr + offset_dst + vector_depth -1) << std::endl;
            }
            
        }, src_iter);
    /*
    unsigned int id_src, offset_vector, offset_dst;
    
    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);
    Iterator vector_iter(vector,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    const auto dst_ptr      = reinterpret_cast<float *>(dst_iter.ptr());
    const auto vector_ptr    = reinterpret_cast<float *>(vector_iter.ptr());

    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(unsigned int x = window_start_x; x < window_end_x; x++)
            {
                id_src = *(src_ptr+x);
                std::cout << id_src << std::endl;

                offset_dst      = x * vector_depth;
                offset_vector    = id_src * vector_depth;

                std::memcpy(dst_ptr + offset_dst, vector_ptr + offset_vector, (vector_depth) * sizeof(*vector_ptr));

                std::cout << *(dst_ptr + offset_dst) << std::endl;
                std::cout << *(dst_ptr + offset_dst + dst->info()->tensor_shape().y()-1) << std::endl;

            }
        },vector_iter,src_iter);
    */
}

} // namespace cpu
} // namespace arm_compute
