#include "src/cpu/kernels/add_vec/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void add_vec_fp32_neon(
    const ITensor *src0, const ITensor *src1, ITensor *dst, size_t src0_target_dim, size_t src1_target_dim, const ConvertPolicy &policy, const Window &window)
{
    return add_vec_same_neon<float>(src0, src1, dst, src0_target_dim, src1_target_dim, policy, window);
}
} // namespace cpu
} // namespace arm_compute
