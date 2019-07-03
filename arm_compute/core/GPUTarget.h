/*
 * Copyright (c) 2018-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __ARM_COMPUTE_GPUTARGET_H__
#define __ARM_COMPUTE_GPUTARGET_H__

#include "arm_compute/core/Helpers.h"

#include <string>

namespace arm_compute
{
/** Available GPU Targets */
enum class GPUTarget
{
    UNKNOWN       = 0x101,
    GPU_ARCH_MASK = 0xF00,
    MIDGARD       = 0x100,
    BIFROST       = 0x200,
    VALHALL       = 0x300,
    T600          = 0x110,
    T700          = 0x120,
    T800          = 0x130,
    G71           = 0x210,
    G72           = 0x220,
    G51           = 0x230,
    G51BIG        = 0x231,
    G51LIT        = 0x232,
    G52           = 0x240,
    G52LIT        = 0x241,
    G76           = 0x250,
    G77           = 0x310,
    TBOX          = 0x320,
    TODX          = 0x330,
};

/** Enable bitwise operations on GPUTarget enumerations */
template <>
struct enable_bitwise_ops<arm_compute::GPUTarget>
{
    static constexpr bool value = true; /**< Enabled. */
};

/** Translates a given gpu device target to string.
 *
 * @param[in] target Given gpu target.
 *
 * @return The string describing the target.
 */
const std::string &string_from_target(GPUTarget target);

/** Helper function to get the GPU target from a device name
 *
 * @param[in] device_name A device name
 *
 * @return the GPU target
 */
GPUTarget get_target_from_name(const std::string &device_name);

/** Helper function to get the GPU arch
 *
 * @param[in] target GPU target
 *
 * @return the GPU target which shows the arch
 */
GPUTarget get_arch_from_target(GPUTarget target);
/** Helper function to check whether a gpu target is equal to the provided targets
 *
 * @param[in] target_to_check gpu target to check
 * @param[in] target          First target to compare against
 * @param[in] targets         (Optional) Additional targets to compare with
 *
 * @return True if the target is equal with at least one of the targets.
 */
template <typename... Args>
bool gpu_target_is_in(GPUTarget target_to_check, GPUTarget target, Args... targets)
{
    return (target_to_check == target) | gpu_target_is_in(target_to_check, targets...);
}

/** Variant of gpu_target_is_in for comparing two targets */
inline bool gpu_target_is_in(GPUTarget target_to_check, GPUTarget target)
{
    return target_to_check == target;
}
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GPUTARGET_H__ */
