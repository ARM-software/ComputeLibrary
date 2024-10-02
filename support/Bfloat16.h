/*
 * Copyright (c) 2020-2022,2024 Arm Limited.
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
#ifndef ACL_SUPPORT_BFLOAT16_H
#define ACL_SUPPORT_BFLOAT16_H

#include <cstdint>
#include <cstring>
#include <ostream>
namespace arm_compute
{
namespace
{
/** Convert float to bfloat16 in a portable way that works on older hardware
 *
 * @param[in] v Floating-point value to convert to bfloat
 *
 * @return Converted value
 */
inline uint16_t portable_float_to_bf16(const float v)
{
    const uint32_t *fromptr = reinterpret_cast<const uint32_t *>(&v);
    uint16_t        res     = (*fromptr >> 16);
    const uint16_t  error   = (*fromptr & 0x0000ffff);
    uint16_t        bf_l    = res & 0x0001;

    if ((error > 0x8000) || ((error == 0x8000) && (bf_l != 0)))
    {
        res += 1;
    }
    return res;
}

/** Convert float to bfloat16
 *
 * @param[in] v Floating-point value to convert to bfloat
 *
 * @return Converted value
 */
inline uint16_t float_to_bf16(const float v)
{
#if defined(ARM_COMPUTE_ENABLE_BF16)
    const uint32_t *fromptr = reinterpret_cast<const uint32_t *>(&v);
    uint16_t        res;

    __asm __volatile("ldr    s0, [%[fromptr]]\n"
                     ".inst    0x1e634000\n" // BFCVT h0, s0
                     "str    h0, [%[toptr]]\n"
                     :
                     : [fromptr] "r"(fromptr), [toptr] "r"(&res)
                     : "v0", "memory");
    return res;
#else  /* defined(ARM_COMPUTE_ENABLE_BF16) */
    return portable_float_to_bf16(v);
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */
}

/** Convert bfloat16 to float
 *
 * @param[in] v Bfloat16 value to convert to float
 *
 * @return Converted value
 */
inline float bf16_to_float(const uint16_t &v)
{
    const uint32_t lv = (v << 16);
    float          fp;
    memcpy(&fp, &lv, sizeof(lv));
    return fp;
}
} // namespace

/** Brain floating point representation class */
class bfloat16 final
{
public:
    /** Default Constructor */
    bfloat16() : value(0)
    {
    }
    /** Constructor
     *
     * @param[in] v Floating-point value
     */
    explicit bfloat16(float v) : value(float_to_bf16(v))
    {
    }
    /** Constructor
     *
     * @param[in] v        Floating-point value
     * @param[in] portable bool to indicate the conversion is to be done in a backward compatible way
     */
    bfloat16(float v, bool portable) : value(0)
    {
        value = portable ? portable_float_to_bf16(v) : float_to_bf16(v);
    }
    /** Assignment operator
     *
     * @param[in] v Floating point value to assign
     *
     * @return The updated object
     */
    bfloat16 &operator=(float v)
    {
        value = float_to_bf16(v);
        return *this;
    }
    /** Floating point conversion operator
     *
     * @return Floating point representation of the value
     */
    operator float() const
    {
        return bf16_to_float(value);
    }
    /** Lowest representative value
     *
     * @return Returns the lowest finite value representable by bfloat16
     */
    static bfloat16 lowest()
    {
        bfloat16 val;
        val.value = 0xFF7F;
        return val;
    }
    /** Largest representative value
     *
     * @return Returns the largest finite value representable by bfloat16
     */
    static bfloat16 max()
    {
        bfloat16 val;
        val.value = 0x7F7F;
        return val;
    }

    bfloat16 &operator+=(float v)
    {
        value = float_to_bf16(bf16_to_float(value) + v);
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &os, const bfloat16 &arg);

private:
    uint16_t value;
};
} // namespace arm_compute
#endif // ACL_SUPPORT_BFLOAT16_H
