#include "src/cpu/kernels/CpuLayerNormKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Validate.h"
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

    void layer_norm_fp32(const ITensor *src, ITensor *dst, const Window &window,float epsilon)
    {
        const int  window_step_x  = 1;
        const auto window_start_x = static_cast<int>(window.x().start());
        const auto window_end_x   = static_cast<int>(window.x().end());

        Window win = window.collapse_if_possible(window, Window::DimZ);
        win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input(src, win);
        Iterator output(dst, win);

        execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
            const auto output_ptr = reinterpret_cast<float *>(output.ptr());

            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                std::cout  <<x<<": Input: "<< *input_ptr << "Output: " << *output_ptr << std::endl;
            }

            
        },
        input, output);

    }

}

void CpuLayerNormKernel::configure(const ITensorInfo *input,
                                    ITensorInfo       *output,
                                    LayerNormLayerInfo   info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input, output, info));

    _info = info;

    Window      win;
    size_t      split_dimension;
    std::tie(win,split_dimension) = calculate_squashed_or_max_window(*input);
    TensorShape out_shape = input->tensor_shape();

    ICPPKernel::configure(win);

    // Auto initialize if empty
    set_shape_if_empty(*output, out_shape);
    set_data_type_if_unknown(*output, input->data_type());
}

Status CpuLayerNormKernel::validate(const ITensorInfo *input,
                                    const ITensorInfo *output,
                                    LayerNormLayerInfo   info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(info);
    return Status{};
}

void CpuLayerNormKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    std::cout << "src/cpu/kernels/CpuLayerNormKernel.cpp" << std::endl;
    ARM_COMPUTE_UNUSED(tensors);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);
}

const char *CpuLayerNormKernel::name() const
{
    return "NELayerNormLayerKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
