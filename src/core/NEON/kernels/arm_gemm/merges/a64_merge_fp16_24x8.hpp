/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#pragma once

#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

template<>
void MergeResults<24, 8, false>(__fp16 *out, const __fp16 *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const __fp16 *bias, Activation act, bool append)
{
    const __fp16 *inptr = in;
    __fp16 nullbias[24];
    __fp16 minval = - static_cast<__fp16>(std::numeric_limits<float>::infinity());
    __fp16 maxval =   static_cast<__fp16>(std::numeric_limits<float>::infinity());

    switch(act.type)
    {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            maxval = static_cast<__fp16>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            minval = 0.0f;
            break;
    }

    if (!append && !bias)
    {
        memset(nullbias, 0, (24 * sizeof(__fp16)));
    }

    for (int y=y0; y<ymax; y+=8)
    {
        __fp16 *outptr0 = out + (y * ldout) + x0;
        __fp16 *outptr1 = outptr0 + ldout;
        __fp16 *outptr2 = outptr1 + ldout;
        __fp16 *outptr3 = outptr2 + ldout;
        __fp16 *outptr4 = outptr3 + ldout;
        __fp16 *outptr5 = outptr4 + ldout;
        __fp16 *outptr6 = outptr5 + ldout;
        __fp16 *outptr7 = outptr6 + ldout;

        const int height = ymax - y;

        for (int i=x0; i<xmax; i+=24)
        {
            if (append)
            {
                switch(height)
                {
                case 1:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 2:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + *outptr1)), maxval);
                                    outptr1++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q5, [%[outptr1]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q6, [%[outptr1], #0x10]\n"
                                "ldr q14, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q7, [%[outptr1], #0x20]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q15, [%[inptr], #0x50]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "str q13, [%[outptr1]]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr1], #0x10]\n"
                                "str q15, [%[outptr1], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 3:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + *outptr1)), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + *outptr2)), maxval);
                                    outptr2++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q5, [%[outptr1]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q6, [%[outptr1], #0x10]\n"
                                "ldr q14, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q7, [%[outptr1], #0x20]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q15, [%[inptr], #0x50]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "ldr q8, [%[outptr2]]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q16, [%[inptr], #0x60]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x70]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr2], #0x20]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x80]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr1]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "str q14, [%[outptr1], #0x10]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "str q15, [%[outptr1], #0x20]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr2]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q17, [%[outptr2], #0x10]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 4:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + *outptr1)), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + *outptr2)), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + *outptr3)), maxval);
                                    outptr3++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q5, [%[outptr1]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q6, [%[outptr1], #0x10]\n"
                                "ldr q14, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q7, [%[outptr1], #0x20]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q15, [%[inptr], #0x50]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "ldr q8, [%[outptr2]]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q16, [%[inptr], #0x60]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x70]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr2], #0x20]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x80]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr3]]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x90]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q4, [%[outptr3], #0x10]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr1]]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "ldr q12, [%[inptr], #0xa0]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q5, [%[outptr3], #0x20]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr1], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q13, [%[inptr], #0xb0]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q15, [%[outptr1], #0x20]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr2]]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "str q17, [%[outptr2], #0x10]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q11, [%[outptr3]]\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr3], #0x10]\n"
                                "str q13, [%[outptr3], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 5:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + *outptr1)), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + *outptr2)), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + *outptr3)), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + *outptr4)), maxval);
                                    outptr4++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q5, [%[outptr1]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q6, [%[outptr1], #0x10]\n"
                                "ldr q14, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q7, [%[outptr1], #0x20]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q15, [%[inptr], #0x50]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "ldr q8, [%[outptr2]]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q16, [%[inptr], #0x60]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x70]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr2], #0x20]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x80]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr3]]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x90]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q4, [%[outptr3], #0x10]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr1]]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "ldr q12, [%[inptr], #0xa0]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q5, [%[outptr3], #0x20]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr1], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q13, [%[inptr], #0xb0]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q6, [%[outptr4]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q15, [%[outptr1], #0x20]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q14, [%[inptr], #0xc0]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q7, [%[outptr4], #0x10]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr2]]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q15, [%[inptr], #0xd0]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q8, [%[outptr4], #0x20]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "str q17, [%[outptr2], #0x10]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "ldr q16, [%[inptr], #0xe0]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q11, [%[outptr3]]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "str q12, [%[outptr3], #0x10]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr3], #0x20]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr4]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "str q15, [%[outptr4], #0x10]\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "str q16, [%[outptr4], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 6:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + *outptr1)), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + *outptr2)), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + *outptr3)), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + *outptr4)), maxval);
                                    outptr4++;
                                    *outptr5 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 120] + *outptr5)), maxval);
                                    outptr5++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q5, [%[outptr1]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q6, [%[outptr1], #0x10]\n"
                                "ldr q14, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q7, [%[outptr1], #0x20]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q15, [%[inptr], #0x50]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "ldr q8, [%[outptr2]]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q16, [%[inptr], #0x60]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x70]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr2], #0x20]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x80]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr3]]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x90]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q4, [%[outptr3], #0x10]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr1]]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "ldr q12, [%[inptr], #0xa0]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q5, [%[outptr3], #0x20]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr1], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q13, [%[inptr], #0xb0]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q6, [%[outptr4]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q15, [%[outptr1], #0x20]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q14, [%[inptr], #0xc0]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q7, [%[outptr4], #0x10]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr2]]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q15, [%[inptr], #0xd0]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q8, [%[outptr4], #0x20]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "str q17, [%[outptr2], #0x10]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "ldr q16, [%[inptr], #0xe0]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q9, [%[outptr5]]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0xf0]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr5], #0x10]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr3]]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x100]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr5], #0x20]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr3], #0x10]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x110]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr3], #0x20]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr4]]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "str q15, [%[outptr4], #0x10]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "str q16, [%[outptr4], #0x20]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "str q17, [%[outptr5]]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "str q10, [%[outptr5], #0x10]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "prfm PLDL1KEEP, [%[outptr5], #0x60]\n"
                                "str q11, [%[outptr5], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 7:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + *outptr1)), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + *outptr2)), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + *outptr3)), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + *outptr4)), maxval);
                                    outptr4++;
                                    *outptr5 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 120] + *outptr5)), maxval);
                                    outptr5++;
                                    *outptr6 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 144] + *outptr6)), maxval);
                                    outptr6++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q5, [%[outptr1]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q6, [%[outptr1], #0x10]\n"
                                "ldr q14, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q7, [%[outptr1], #0x20]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q15, [%[inptr], #0x50]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "ldr q8, [%[outptr2]]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q16, [%[inptr], #0x60]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x70]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr2], #0x20]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x80]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr3]]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x90]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q4, [%[outptr3], #0x10]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr1]]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "ldr q12, [%[inptr], #0xa0]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q5, [%[outptr3], #0x20]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr1], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q13, [%[inptr], #0xb0]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q6, [%[outptr4]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q15, [%[outptr1], #0x20]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q14, [%[inptr], #0xc0]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q7, [%[outptr4], #0x10]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr2]]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q15, [%[inptr], #0xd0]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q8, [%[outptr4], #0x20]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "str q17, [%[outptr2], #0x10]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "ldr q16, [%[inptr], #0xe0]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q9, [%[outptr5]]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0xf0]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr5], #0x10]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr3]]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x100]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr5], #0x20]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr3], #0x10]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x110]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q4, [%[outptr6]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr3], #0x20]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "ldr q12, [%[inptr], #0x120]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q5, [%[outptr6], #0x10]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr4]]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q13, [%[inptr], #0x130]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q6, [%[outptr6], #0x20]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q15, [%[outptr4], #0x10]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q14, [%[inptr], #0x140]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr4], #0x20]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "str q17, [%[outptr5]]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr5], #0x10]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q11, [%[outptr5], #0x20]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr6]]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "str q13, [%[outptr6], #0x10]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "prfm PLDL1KEEP, [%[outptr5], #0x60]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "str q14, [%[outptr6], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "prfm PLDL1KEEP, [%[outptr6], #0x60]\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                default:
                case 8:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + *outptr0)), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + *outptr1)), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + *outptr2)), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + *outptr3)), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + *outptr4)), maxval);
                                    outptr4++;
                                    *outptr5 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 120] + *outptr5)), maxval);
                                    outptr5++;
                                    *outptr6 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 144] + *outptr6)), maxval);
                                    outptr6++;
                                    *outptr7 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 168] + *outptr7)), maxval);
                                    outptr7++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[outptr0]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q10, [%[inptr]]\n"
                                "ldr q3, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q4, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q5, [%[outptr1]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q6, [%[outptr1], #0x10]\n"
                                "ldr q14, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q7, [%[outptr1], #0x20]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q15, [%[inptr], #0x50]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "ldr q8, [%[outptr2]]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "str q10, [%[outptr0]]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q16, [%[inptr], #0x60]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x70]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr2], #0x20]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x80]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr3]]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr0], #0x20]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x90]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q4, [%[outptr3], #0x10]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr1]]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "ldr q12, [%[inptr], #0xa0]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q5, [%[outptr3], #0x20]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr1], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q13, [%[inptr], #0xb0]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q6, [%[outptr4]]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q15, [%[outptr1], #0x20]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q14, [%[inptr], #0xc0]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q7, [%[outptr4], #0x10]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr2]]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q15, [%[inptr], #0xd0]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q8, [%[outptr4], #0x20]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "str q17, [%[outptr2], #0x10]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "ldr q16, [%[inptr], #0xe0]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q9, [%[outptr5]]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0xf0]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q2, [%[outptr5], #0x10]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr3]]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "ldr q10, [%[inptr], #0x100]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "ldr q3, [%[outptr5], #0x20]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr3], #0x10]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q11, [%[inptr], #0x110]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q4, [%[outptr6]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr3], #0x20]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "ldr q12, [%[inptr], #0x120]\n"
                                "fadd v10.8h, v10.8h, v2.8h\n"
                                "ldr q5, [%[outptr6], #0x10]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr4]]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q13, [%[inptr], #0x130]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q6, [%[outptr6], #0x20]\n"
                                "fmin v10.8h, v10.8h, v0.8h\n"
                                "str q15, [%[outptr4], #0x10]\n"
                                "fadd v11.8h, v11.8h, v3.8h\n"
                                "ldr q14, [%[inptr], #0x140]\n"
                                "fadd v12.8h, v12.8h, v4.8h\n"
                                "ldr q7, [%[outptr7]]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr4], #0x20]\n"
                                "fmax v10.8h, v10.8h, v1.8h\n"
                                "ldr q15, [%[inptr], #0x150]\n"
                                "fmin v11.8h, v11.8h, v0.8h\n"
                                "ldr q8, [%[outptr7], #0x10]\n"
                                "fmin v12.8h, v12.8h, v0.8h\n"
                                "str q17, [%[outptr5]]\n"
                                "fadd v13.8h, v13.8h, v5.8h\n"
                                "ldr q16, [%[inptr], #0x160]\n"
                                "fadd v14.8h, v14.8h, v6.8h\n"
                                "ldr q9, [%[outptr7], #0x20]\n"
                                "fmax v11.8h, v11.8h, v1.8h\n"
                                "str q10, [%[outptr5], #0x10]\n"
                                "fmax v12.8h, v12.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x170]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q11, [%[outptr5], #0x20]\n"
                                "fadd v15.8h, v15.8h, v7.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q12, [%[outptr6]]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr6], #0x10]\n"
                                "fadd v16.8h, v16.8h, v8.8h\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fadd v17.8h, v17.8h, v9.8h\n"
                                "str q14, [%[outptr6], #0x20]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "str q15, [%[outptr7]]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "str q16, [%[outptr7], #0x10]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "prfm PLDL1KEEP, [%[outptr5], #0x60]\n"
                                "str q17, [%[outptr7], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "prfm PLDL1KEEP, [%[outptr6], #0x60]\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                                "prfm PLDL1KEEP, [%[outptr7], #0x60]\n"
                                "add %[outptr7], %[outptr7], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;


                }
            }
            else
            {
                const __fp16 *biasptr = bias ? bias + i : nullbias;

                switch(height)
                {
                case 1:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 2:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + biasptr[xi])), maxval);
                                    outptr1++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "ldr q16, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "ldr q17, [%[inptr], #0x40]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q18, [%[inptr], #0x50]\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "fadd v16.8h, v16.8h, v2.8h\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "fadd v17.8h, v17.8h, v3.8h\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "fadd v18.8h, v18.8h, v4.8h\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr1]]\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "str q17, [%[outptr1], #0x10]\n"
                                "str q18, [%[outptr1], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 3:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + biasptr[xi])), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + biasptr[xi])), maxval);
                                    outptr2++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "ldr q16, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "ldr q17, [%[inptr], #0x40]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q18, [%[inptr], #0x50]\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "ldr q19, [%[inptr], #0x60]\n"
                                "fadd v16.8h, v16.8h, v2.8h\n"
                                "ldr q20, [%[inptr], #0x70]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x80]\n"
                                "fadd v17.8h, v17.8h, v3.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "fadd v18.8h, v18.8h, v4.8h\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "fadd v19.8h, v19.8h, v2.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr1]]\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "fadd v20.8h, v20.8h, v3.8h\n"
                                "str q17, [%[outptr1], #0x10]\n"
                                "fadd v13.8h, v13.8h, v4.8h\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q18, [%[outptr1], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "str q19, [%[outptr2]]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q20, [%[outptr2], #0x10]\n"
                                "str q13, [%[outptr2], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 4:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + biasptr[xi])), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + biasptr[xi])), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + biasptr[xi])), maxval);
                                    outptr3++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "ldr q16, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "ldr q17, [%[inptr], #0x40]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q18, [%[inptr], #0x50]\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "ldr q19, [%[inptr], #0x60]\n"
                                "fadd v16.8h, v16.8h, v2.8h\n"
                                "ldr q20, [%[inptr], #0x70]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x80]\n"
                                "fadd v17.8h, v17.8h, v3.8h\n"
                                "fadd v18.8h, v18.8h, v4.8h\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "ldr q14, [%[inptr], #0x90]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "fadd v19.8h, v19.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0xa0]\n"
                                "fadd v20.8h, v20.8h, v3.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr1]]\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "ldr q16, [%[inptr], #0xb0]\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "str q17, [%[outptr1], #0x10]\n"
                                "fadd v13.8h, v13.8h, v4.8h\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "fadd v14.8h, v14.8h, v2.8h\n"
                                "str q18, [%[outptr1], #0x20]\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q19, [%[outptr2]]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "fadd v15.8h, v15.8h, v3.8h\n"
                                "fadd v16.8h, v16.8h, v4.8h\n"
                                "str q20, [%[outptr2], #0x10]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr2], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr3]]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "str q15, [%[outptr3], #0x10]\n"
                                "str q16, [%[outptr3], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 5:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + biasptr[xi])), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + biasptr[xi])), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + biasptr[xi])), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + biasptr[xi])), maxval);
                                    outptr4++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "ldr q16, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "ldr q17, [%[inptr], #0x40]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q18, [%[inptr], #0x50]\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "ldr q19, [%[inptr], #0x60]\n"
                                "fadd v16.8h, v16.8h, v2.8h\n"
                                "ldr q20, [%[inptr], #0x70]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x80]\n"
                                "fadd v17.8h, v17.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q14, [%[inptr], #0x90]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fadd v18.8h, v18.8h, v4.8h\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "fadd v19.8h, v19.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0xa0]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "str q16, [%[outptr1]]\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "ldr q16, [%[inptr], #0xb0]\n"
                                "fadd v20.8h, v20.8h, v3.8h\n"
                                "fadd v13.8h, v13.8h, v4.8h\n"
                                "str q17, [%[outptr1], #0x10]\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0xc0]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q18, [%[outptr1], #0x20]\n"
                                "fadd v14.8h, v14.8h, v2.8h\n"
                                "ldr q18, [%[inptr], #0xd0]\n"
                                "fadd v15.8h, v15.8h, v3.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "str q19, [%[outptr2]]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "ldr q19, [%[inptr], #0xe0]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q20, [%[outptr2], #0x10]\n"
                                "fadd v16.8h, v16.8h, v4.8h\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "fadd v17.8h, v17.8h, v2.8h\n"
                                "str q13, [%[outptr2], #0x20]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q14, [%[outptr3]]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "fadd v18.8h, v18.8h, v3.8h\n"
                                "fadd v19.8h, v19.8h, v4.8h\n"
                                "str q15, [%[outptr3], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "str q16, [%[outptr3], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "str q17, [%[outptr4]]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "str q18, [%[outptr4], #0x10]\n"
                                "str q19, [%[outptr4], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 6:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + biasptr[xi])), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + biasptr[xi])), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + biasptr[xi])), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + biasptr[xi])), maxval);
                                    outptr4++;
                                    *outptr5 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 120] + biasptr[xi])), maxval);
                                    outptr5++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "ldr q16, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "ldr q17, [%[inptr], #0x40]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q18, [%[inptr], #0x50]\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "ldr q19, [%[inptr], #0x60]\n"
                                "fadd v16.8h, v16.8h, v2.8h\n"
                                "ldr q20, [%[inptr], #0x70]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x80]\n"
                                "fadd v17.8h, v17.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q14, [%[inptr], #0x90]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fadd v18.8h, v18.8h, v4.8h\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "fadd v19.8h, v19.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0xa0]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "str q16, [%[outptr1]]\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "ldr q16, [%[inptr], #0xb0]\n"
                                "fadd v20.8h, v20.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "str q17, [%[outptr1], #0x10]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0xc0]\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr5], #0x60]\n"
                                "fadd v13.8h, v13.8h, v4.8h\n"
                                "str q18, [%[outptr1], #0x20]\n"
                                "fadd v14.8h, v14.8h, v2.8h\n"
                                "ldr q18, [%[inptr], #0xd0]\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q19, [%[outptr2]]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "ldr q19, [%[inptr], #0xe0]\n"
                                "fadd v15.8h, v15.8h, v3.8h\n"
                                "fadd v16.8h, v16.8h, v4.8h\n"
                                "str q20, [%[outptr2], #0x10]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "ldr q20, [%[inptr], #0xf0]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q13, [%[outptr2], #0x20]\n"
                                "fadd v17.8h, v17.8h, v2.8h\n"
                                "ldr q13, [%[inptr], #0x100]\n"
                                "fadd v18.8h, v18.8h, v3.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr3]]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q14, [%[inptr], #0x110]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "str q15, [%[outptr3], #0x10]\n"
                                "fadd v19.8h, v19.8h, v4.8h\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "fadd v20.8h, v20.8h, v2.8h\n"
                                "str q16, [%[outptr3], #0x20]\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "str q17, [%[outptr4]]\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "fadd v13.8h, v13.8h, v3.8h\n"
                                "fadd v14.8h, v14.8h, v4.8h\n"
                                "str q18, [%[outptr4], #0x10]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q19, [%[outptr4], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q20, [%[outptr5]]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "str q13, [%[outptr5], #0x10]\n"
                                "str q14, [%[outptr5], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                case 7:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + biasptr[xi])), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + biasptr[xi])), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + biasptr[xi])), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + biasptr[xi])), maxval);
                                    outptr4++;
                                    *outptr5 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 120] + biasptr[xi])), maxval);
                                    outptr5++;
                                    *outptr6 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 144] + biasptr[xi])), maxval);
                                    outptr6++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "ldr q16, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "ldr q17, [%[inptr], #0x40]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q18, [%[inptr], #0x50]\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "ldr q19, [%[inptr], #0x60]\n"
                                "fadd v16.8h, v16.8h, v2.8h\n"
                                "ldr q20, [%[inptr], #0x70]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x80]\n"
                                "fadd v17.8h, v17.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q14, [%[inptr], #0x90]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fadd v18.8h, v18.8h, v4.8h\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "fadd v19.8h, v19.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0xa0]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "str q16, [%[outptr1]]\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "ldr q16, [%[inptr], #0xb0]\n"
                                "fadd v20.8h, v20.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "str q17, [%[outptr1], #0x10]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0xc0]\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr5], #0x60]\n"
                                "fadd v13.8h, v13.8h, v4.8h\n"
                                "str q18, [%[outptr1], #0x20]\n"
                                "fadd v14.8h, v14.8h, v2.8h\n"
                                "ldr q18, [%[inptr], #0xd0]\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q19, [%[outptr2]]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "ldr q19, [%[inptr], #0xe0]\n"
                                "fadd v15.8h, v15.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q20, [%[outptr2], #0x10]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q20, [%[inptr], #0xf0]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr6], #0x60]\n"
                                "fadd v16.8h, v16.8h, v4.8h\n"
                                "str q13, [%[outptr2], #0x20]\n"
                                "fadd v17.8h, v17.8h, v2.8h\n"
                                "ldr q13, [%[inptr], #0x100]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q14, [%[outptr3]]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q14, [%[inptr], #0x110]\n"
                                "fadd v18.8h, v18.8h, v3.8h\n"
                                "fadd v19.8h, v19.8h, v4.8h\n"
                                "str q15, [%[outptr3], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q15, [%[inptr], #0x120]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "str q16, [%[outptr3], #0x20]\n"
                                "fadd v20.8h, v20.8h, v2.8h\n"
                                "ldr q16, [%[inptr], #0x130]\n"
                                "fadd v13.8h, v13.8h, v3.8h\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "str q17, [%[outptr4]]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0x140]\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q18, [%[outptr4], #0x10]\n"
                                "fadd v14.8h, v14.8h, v4.8h\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "fadd v15.8h, v15.8h, v2.8h\n"
                                "str q19, [%[outptr4], #0x20]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q20, [%[outptr5]]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "fadd v16.8h, v16.8h, v3.8h\n"
                                "fadd v17.8h, v17.8h, v4.8h\n"
                                "str q13, [%[outptr5], #0x10]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "str q14, [%[outptr5], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "str q15, [%[outptr6]]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "str q16, [%[outptr6], #0x10]\n"
                                "str q17, [%[outptr6], #0x20]\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;

                default:
                case 8:
                    {
                        if ((i+23) >= xmax)
                        {
                            for (int xi=0; xi<23; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi] + biasptr[xi])), maxval);
                                    outptr0++;
                                    *outptr1 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 24] + biasptr[xi])), maxval);
                                    outptr1++;
                                    *outptr2 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 48] + biasptr[xi])), maxval);
                                    outptr2++;
                                    *outptr3 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 72] + biasptr[xi])), maxval);
                                    outptr3++;
                                    *outptr4 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 96] + biasptr[xi])), maxval);
                                    outptr4++;
                                    *outptr5 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 120] + biasptr[xi])), maxval);
                                    outptr5++;
                                    *outptr6 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 144] + biasptr[xi])), maxval);
                                    outptr6++;
                                    *outptr7 = std::min(std::max(minval, static_cast<__fp16>(inptr[xi + 168] + biasptr[xi])), maxval);
                                    outptr7++;
                                }
                            }
                            inptr += 192;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                ".arch  armv8.2-a+fp16\n"
#endif
                                "dup v0.8h, %[maxval].h[0]\n"
                                "ldr q2, [%[biasptr]]\n"
                                "dup v1.8h, %[minval].h[0]\n"
                                "ldr q3, [%[biasptr], #0x10]\n"
                                "ldr q4, [%[biasptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "ldr q13, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "ldr q14, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fadd v13.8h, v13.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0x20]\n"
                                "ldr q16, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fadd v14.8h, v14.8h, v3.8h\n"
                                "ldr q17, [%[inptr], #0x40]\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "ldr q18, [%[inptr], #0x50]\n"
                                "fadd v15.8h, v15.8h, v4.8h\n"
                                "ldr q19, [%[inptr], #0x60]\n"
                                "fadd v16.8h, v16.8h, v2.8h\n"
                                "ldr q20, [%[inptr], #0x70]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "str q13, [%[outptr0]]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "ldr q13, [%[inptr], #0x80]\n"
                                "fadd v17.8h, v17.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "str q14, [%[outptr0], #0x10]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "ldr q14, [%[inptr], #0x90]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fadd v18.8h, v18.8h, v4.8h\n"
                                "str q15, [%[outptr0], #0x20]\n"
                                "fadd v19.8h, v19.8h, v2.8h\n"
                                "ldr q15, [%[inptr], #0xa0]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "str q16, [%[outptr1]]\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "ldr q16, [%[inptr], #0xb0]\n"
                                "fadd v20.8h, v20.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "str q17, [%[outptr1], #0x10]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "ldr q17, [%[inptr], #0xc0]\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr5], #0x60]\n"
                                "fadd v13.8h, v13.8h, v4.8h\n"
                                "str q18, [%[outptr1], #0x20]\n"
                                "fadd v14.8h, v14.8h, v2.8h\n"
                                "ldr q18, [%[inptr], #0xd0]\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q19, [%[outptr2]]\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "ldr q19, [%[inptr], #0xe0]\n"
                                "fadd v15.8h, v15.8h, v3.8h\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "str q20, [%[outptr2], #0x10]\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "ldr q20, [%[inptr], #0xf0]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "prfm PSTL1KEEP, [%[outptr6], #0x60]\n"
                                "fadd v16.8h, v16.8h, v4.8h\n"
                                "str q13, [%[outptr2], #0x20]\n"
                                "fadd v17.8h, v17.8h, v2.8h\n"
                                "ldr q13, [%[inptr], #0x100]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "str q14, [%[outptr3]]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "ldr q14, [%[inptr], #0x110]\n"
                                "fadd v18.8h, v18.8h, v3.8h\n"
                                "prfm PSTL1KEEP, [%[outptr7], #0x60]\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "str q15, [%[outptr3], #0x10]\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "ldr q15, [%[inptr], #0x120]\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "fadd v19.8h, v19.8h, v4.8h\n"
                                "str q16, [%[outptr3], #0x20]\n"
                                "fadd v20.8h, v20.8h, v2.8h\n"
                                "ldr q16, [%[inptr], #0x130]\n"
                                "fadd v13.8h, v13.8h, v3.8h\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "str q17, [%[outptr4]]\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "ldr q17, [%[inptr], #0x140]\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "fmin v13.8h, v13.8h, v0.8h\n"
                                "str q18, [%[outptr4], #0x10]\n"
                                "fadd v14.8h, v14.8h, v4.8h\n"
                                "ldr q18, [%[inptr], #0x150]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "fmax v13.8h, v13.8h, v1.8h\n"
                                "fmin v14.8h, v14.8h, v0.8h\n"
                                "str q19, [%[outptr4], #0x20]\n"
                                "fadd v15.8h, v15.8h, v2.8h\n"
                                "ldr q19, [%[inptr], #0x160]\n"
                                "fadd v16.8h, v16.8h, v3.8h\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmax v14.8h, v14.8h, v1.8h\n"
                                "str q20, [%[outptr5]]\n"
                                "fmin v15.8h, v15.8h, v0.8h\n"
                                "ldr q20, [%[inptr], #0x170]\n"
                                "fmin v16.8h, v16.8h, v0.8h\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fadd v17.8h, v17.8h, v4.8h\n"
                                "str q13, [%[outptr5], #0x10]\n"
                                "fmax v15.8h, v15.8h, v1.8h\n"
                                "fmax v16.8h, v16.8h, v1.8h\n"
                                "fadd v18.8h, v18.8h, v2.8h\n"
                                "str q14, [%[outptr5], #0x20]\n"
                                "fmin v17.8h, v17.8h, v0.8h\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "fadd v19.8h, v19.8h, v3.8h\n"
                                "str q15, [%[outptr6]]\n"
                                "fmin v18.8h, v18.8h, v0.8h\n"
                                "fmax v17.8h, v17.8h, v1.8h\n"
                                "fadd v20.8h, v20.8h, v4.8h\n"
                                "str q16, [%[outptr6], #0x10]\n"
                                "fmin v19.8h, v19.8h, v0.8h\n"
                                "fmax v18.8h, v18.8h, v1.8h\n"
                                "fmin v20.8h, v20.8h, v0.8h\n"
                                "str q17, [%[outptr6], #0x20]\n"
                                "fmax v19.8h, v19.8h, v1.8h\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                                "fmax v20.8h, v20.8h, v1.8h\n"
                                "str q18, [%[outptr7]]\n"
                                "str q19, [%[outptr7], #0x10]\n"
                                "str q20, [%[outptr7], #0x20]\n"
                                "add %[outptr7], %[outptr7], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr), [minval] "w" (minval), [maxval] "w" (maxval)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "memory"
                            );
                        }
                    }
                    break;


                }
            }
        }
    }
}

#endif // __aarch64__ && (FP16_KERNELS || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
