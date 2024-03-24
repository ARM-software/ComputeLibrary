#include "src/cpu/kernels/CpuPositionalEncodingKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

namespace
{
template <typename T>
void run_positional_encoding(const Window &window, ITensor *src, ITensor *dst, const unsigned int d_model)
{
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(d_model);

    std::cout << "src/cpu/kernels/CpuPositionalEncodingKernel.cpp" << std::endl;

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0,1,1));
    win.set(Window::DimY, Window::Dimension(0,1,1));
    /* token sequence */
    const unsigned int window_start_x   = static_cast<unsigned int>(window.x().start());
    const unsigned int window_end_x     = static_cast<unsigned int>(window.x().end());

    Iterator src_iter(src,win);
    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(unsigned int x = window_start_x; x < window_end_x; x++)
            {
                std::cout << x << std::endl;;
            }
    },src_iter);

    /*
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0,1,1));
    win.set(Window::DimY, Window::Dimension(0,1,1));
    const unsigned int window_start_x   = static_cast<unsigned int>(window.x().start());
    const unsigned int window_end_x     = static_cast<unsigned int>(window.x().end());
    unsigned int       x                = window_start_x;

    const unsigned int vector_depth     = d_model;

    unsigned int id_src, offset_dst;
    
    Iterator src_iter(src,win);
    Iterator dst_iter(dst,win);

    const auto src_ptr      = reinterpret_cast<unsigned int *>(src_iter.ptr());
    //const auto dst_ptr      = reinterpret_cast<float *>(dst_iter.ptr());

    execute_window_loop(win,
        [&](const Coordinates &)
        {
            for(; x < window_end_x; x++)
            {
                id_src = *(src_ptr+x);
                std::cout << id_src << std::endl;

                offset_dst      = x * vector_depth;

                std::cout << *(src_ptr + offset_dst) << std::endl;
                std::cout << *(src_ptr + offset_dst + dst->info()->tensor_shape().y()-1) << std::endl;

            }
        },src_iter);
*/

}

}

void CpuPositionalEncodingKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const unsigned int d_model)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    _d_model = d_model;

    // Configure output tensor info.
    auto_init_if_empty(*dst, TensorInfo(*src->clone()));

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICpuKernel::configure(win);
}


Status CpuPositionalEncodingKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const unsigned int d_model)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(d_model);

    return Status{};
}

void CpuPositionalEncodingKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    auto src = tensors.get_tensor(TensorType::ACL_DST);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    run_positional_encoding<float>(window, src, dst, _d_model);
}

const char * CpuPositionalEncodingKernel::name() const
{
    return "CpuPositionalEncodingKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
