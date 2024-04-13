#ifndef SRC_CPU_KERNELS_VECTORIZE_LIST_H
#define SRC_CPU_KERNELS_VECTORIZE_LIST_H

namespace arm_compute
{
namespace cpu
{
#define DECLARE_VECTORIZE_KERNEL(func_name) \
    void func_name(const ITensor *src, const ITensor *vector, ITensor *dst, const Window &window)

#ifdef __aarch64__
DECLARE_VECTORIZE_KERNEL(neon_vectorize_int_2_float32);
#endif // __aarch64__

#undef DECLARE_ACTIVATION_KERNEL
} // namespace cpu
} // namespace arm_compute

#endif // SRC_CPU_KERNELS_VECTORIZE_LIST_H

