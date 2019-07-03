/*
 * Copyright (c) 2017-2019 ARM Limited.
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
namespace arm_compute
{
inline qasymm8x16_t vmlaq_qasymm8(qasymm8x16_t vd, float32x4_t vs, float32x4_t vo)
{
    // Convert uint8 vectors to uint16 vectors
    const uint8x8_t vd_low        = vget_low_u8(vd);
    const uint8x8_t vd_high       = vget_high_u8(vd);
    uint16x8_t      vd_low_u16x8  = vmovl_u8(vd_low);
    uint16x8_t      vd_high_u16x8 = vmovl_u8(vd_high);
    // Convert uint16 vectors to uint32 vectors
    uint32x4_t A_u32x4 = vmovl_u16(vget_low_u16(vd_low_u16x8));
    uint32x4_t B_u32x4 = vmovl_u16(vget_high_u16(vd_low_u16x8));
    uint32x4_t C_u32x4 = vmovl_u16(vget_low_u16(vd_high_u16x8));
    uint32x4_t D_u32x4 = vmovl_u16(vget_high_u16(vd_high_u16x8));
    // Convert uint32 vectors to float32 vectors
    float32x4_t A_f32x4 = vcvtq_f32_u32(A_u32x4);
    float32x4_t B_f32x4 = vcvtq_f32_u32(B_u32x4);
    float32x4_t C_f32x4 = vcvtq_f32_u32(C_u32x4);
    float32x4_t D_f32x4 = vcvtq_f32_u32(D_u32x4);
    // vd = vd*vs + vo
    A_f32x4 = vmlaq_f32(vo, A_f32x4, vs);
    B_f32x4 = vmlaq_f32(vo, B_f32x4, vs);
    C_f32x4 = vmlaq_f32(vo, C_f32x4, vs);
    D_f32x4 = vmlaq_f32(vo, D_f32x4, vs);
    // Convert float32 vectors to uint32 vectors
    A_u32x4 = vcvtq_u32_f32(A_f32x4);
    B_u32x4 = vcvtq_u32_f32(B_f32x4);
    C_u32x4 = vcvtq_u32_f32(C_f32x4);
    D_u32x4 = vcvtq_u32_f32(D_f32x4);
    // Convert uint32 vectors to uint16 vectors (with saturation)
    vd_low_u16x8  = vcombine_u16(vqmovn_u32(A_u32x4), vqmovn_u32(B_u32x4));
    vd_high_u16x8 = vcombine_u16(vqmovn_u32(C_u32x4), vqmovn_u32(D_u32x4));
    // convert uint16 vectors to uint8 vectors (with saturation)
    return vcombine_u8(vqmovn_u16(vd_low_u16x8), vqmovn_u16(vd_high_u16x8));
}
} // namespace arm_compute
