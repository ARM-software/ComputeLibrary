#ifndef SRC_CORE_NEON_KERNELS_ADD_VEC_IMPL_H
#define SRC_CORE_NEON_KERNELS_ADD_VEC_IMPL_H
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/core/Window.h"

#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/Utils.h"

namespace arm_compute
{
namespace cpu
{
template <typename ScalarType>
void add_vec_same_neon(
    const ITensor *src0, const ITensor *src1, ITensor *dst, size_t src0_target_dim, size_t src1_target_dim, const ConvertPolicy &policy, const Window &window)
{

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win;
    input2_win.use_tensor_dimensions(src1->info()->tensor_shape());
    input2_win = input2_win.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(src0_target_dim, Window::Dimension(0, 1, 1));

    constexpr int window_step_target0         = 16 / sizeof(ScalarType);
    const auto    window_start_target0        = static_cast<int>(window[src0_target_dim].start());
    const auto    window_end_target0          = static_cast<int>(window[src0_target_dim].end());

    // Clear target Dimension on execution window as we handle manually
    input1_win.set(src0_target_dim, Window::Dimension(0, 1, 1));
    input2_win.set(src1_target_dim, Window::Dimension(0, 0, 0)); // No increament

    Iterator input1(src0, input1_win);
    Iterator input2(src1, input2_win);
    Iterator output(dst, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_target0;
            for (; x <= (window_end_target0 - window_step_target0); x += window_step_target0)
            {
                const auto val1 = wrapper::vloadq(input1_ptr + x);
                const auto val2 = wrapper::vloadq(input2_ptr + x);
                const auto res =
                    (policy == ConvertPolicy::SATURATE) ? wrapper::vqadd(val1, val2) : wrapper::vadd(val1, val2);
                wrapper::vstore(output_ptr + x, res);
                
            }
            // Compute left-over elements
            for (; x < window_end_target0; ++x)
            {
                const auto val1 = *(input1_ptr + x);
                const auto val2 = *(input2_ptr + x);
                *(output_ptr + x) =
                    (policy == ConvertPolicy::SATURATE) ? wrapper::add_sat(val1, val2) : val1 + val2;
            }
        },
        input1, input2, output);
}

} // namespace cpu
} // namespace arm_compute
#endif // SRC_CORE_NEON_KERNELS_ADD_IMPL_H
