#include "src/cpu/kernels/add_vec/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void add_fp32_neon(
    const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    return add_same_neon<float>(src0, src1, dst, policy, window);
}
} // namespace cpu
} // namespace arm_compute
