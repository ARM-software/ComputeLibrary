#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
void neon_vectorize_int_2_float32(const ITensor *src, const ITensor *vector, ITensor *dst, const Window &window)
{  
    /* Runtime reshape valid tensor region if input has been reshaped during preprocess */
    size_t reshape_input_x = src->info()->valid_region().shape.x();
    if(src->info()->tensor_shape().x() != reshape_input_x)
    {
        dst->info()->set_valid_region(dst->info()->valid_region().set(0,0,reshape_input_x));
    }
    Window win(window);

    const unsigned int window_start_x   = static_cast<unsigned int>(win.x().start());
    const unsigned int window_end_x     = static_cast<unsigned int>(win.x().end());

    const unsigned int vector_depth     = vector->info()->tensor_shape().x(); 

    unsigned int offset_vector,offset_dst;

    win.set(Window::DimX, Window::Dimension(0,1,1));
    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);
    Iterator vector_iter(vector,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    const auto dst_ptr      = reinterpret_cast<float *>(dst_iter.ptr());
    const auto vector_ptr   = reinterpret_cast<float *>(vector_iter.ptr());
    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(unsigned int x = window_start_x; x < window_end_x; x++)
            {
                offset_dst     = x * vector_depth;
                offset_vector  = *(src_ptr+x) * vector_depth;
                std::memcpy(dst_ptr + offset_dst, vector_ptr + offset_vector, (vector_depth) * sizeof(*vector_ptr));
            }
            
        }, src_iter);
    /*
    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(unsigned int x = window_start_x; x < window_end_x; x++)
            {
                offset_dst     = x * vector_depth;
                offset_vector  = *(src_ptr+x) * vector_depth;
                std::memcpy(dst_ptr + offset_dst, vector_ptr + offset_vector, (vector_depth) * sizeof(*vector_ptr));
                std::cout<< "x:  " << x <<" "  << *(src_ptr+x) << ":  "
                << *(dst_ptr + offset_dst)<< " " 
                << *(dst_ptr + offset_dst + vector_depth -1) << std::endl;
            }
            
        }, src_iter);*/
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
