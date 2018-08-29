/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifdef __aarch64__

#include <algorithm>

#include <arm_neon.h>

#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void a64_sgemv_pretransposed(const float *A, int lda, const float *X, float *Y, float beta, int M, int N) {
    const bool beta0 = (beta==0.0f);
    const bool beta1 = (beta==1.0f);

    for (int x=0; x<N; x+=32) {
        float *y_ptr = Y + x;

        // How many elements are we processing in this loop?
        int l = std::min(N - x, 32);

        register float32x4_t r0 asm("v24");
        register float32x4_t r1 asm("v25");
        register float32x4_t r2 asm("v26");
        register float32x4_t r3 asm("v27");
        register float32x4_t r4 asm("v28");
        register float32x4_t r5 asm("v29");
        register float32x4_t r6 asm("v30");
        register float32x4_t r7 asm("v31");

        register float32x4_t x0  asm("v0");
        register float32x4_t x0a asm("v1");

        const float *x_ptr = X;
        const float *a_ptr = A + ((x/32) * lda);

        if (beta0) {
            r0=r1=r2=r3=r4=r5=r6=r7=vdupq_n_f32(0.0f);
        } else {
            if (l==32) {
                // Fastest path - load all 8 vectors
                r0 = vld1q_f32(y_ptr);
                r1 = vld1q_f32(y_ptr + 4);
                r2 = vld1q_f32(y_ptr + 8);
                r3 = vld1q_f32(y_ptr + 12);
                r4 = vld1q_f32(y_ptr + 16);
                r5 = vld1q_f32(y_ptr + 20);
                r6 = vld1q_f32(y_ptr + 24);
                r7 = vld1q_f32(y_ptr + 28);
            } else {
                // Slow case - leftovers.  Note that we don't care about
                // out-of-range vectors and lanes as we will throw them away at
                // the end.
                int vecs=l/4; // How many leftover vectors?
                int oddbits=l%4; // And how many odd single values?

                if (oddbits) {
                    // Load the outstanding odd values into a vector first
                    float32x4_t oddvec = vdupq_n_f32(0.0f); // This does not really need to be initialized, but the compiler has a hard time with that.
                    float *oddbase = y_ptr + l - oddbits;

                    switch (oddbits) {
                        case 3:
                            oddvec = vld1q_lane_f32(oddbase + 2, oddvec, 2);
                            // fall through
                        case 2:
                            oddvec = vld1q_lane_f32(oddbase + 1, oddvec, 1);
                            // fall through
                        case 1:
                            oddvec = vld1q_lane_f32(oddbase, oddvec, 0);
                            break;

                        default:
                            UNREACHABLE("Impossible case in switch.");
                    }

                    // Now load the whole vectors, putting the oddments in when we run out.
                    do {
                        if (vecs==0) { r0 = oddvec; break; }

                        r0 = vld1q_f32(y_ptr);
                        if (--vecs==0) { r1 = oddvec; break; }

                        r1 = vld1q_f32(y_ptr + 4);
                        if (--vecs==0) { r2 = oddvec; break; }

                        r2 = vld1q_f32(y_ptr + 8);
                        if (--vecs==0) { r3 = oddvec; break; }

                        r3 = vld1q_f32(y_ptr + 12);
                        if (--vecs==0) { r4 = oddvec; break; }

                        r4 = vld1q_f32(y_ptr + 16);
                        if (--vecs==0) { r5 = oddvec; break; }

                        r5 = vld1q_f32(y_ptr + 20);
                        if (--vecs==0) { r6 = oddvec; break; }

                        r6 = vld1q_f32(y_ptr + 24);
                        r7 = oddvec;
                    } while (0);
                } else {
                    // Slightly less slow path - just load the whole vectors
                    do {
                        // It can't be the case that oddbits==0 AND vecs==0 or we wouldn't be here.
                        if (vecs==0) { UNREACHABLE("Impossible lack of work to do"); }

                        r0 = vld1q_f32(y_ptr);
                        if (--vecs==0) { break; }

                        r1 = vld1q_f32(y_ptr + 4);
                        if (--vecs==0) { break; }

                        r2 = vld1q_f32(y_ptr + 8);
                        if (--vecs==0) { break; }

                        r3 = vld1q_f32(y_ptr + 12);
                        if (--vecs==0) { break; }

                        r4 = vld1q_f32(y_ptr + 16);
                        if (--vecs==0) { break; }

                        r5 = vld1q_f32(y_ptr + 20);
                        if (--vecs==0) { break; }

                        r6 = vld1q_f32(y_ptr + 24);
                    } while (0);
                }
            }

            if (!beta1) {
                const float32x4_t vb = vdupq_n_f32(beta);

                r0 = vmulq_f32(r0, vb);
                r1 = vmulq_f32(r1, vb);
                r2 = vmulq_f32(r2, vb);
                r3 = vmulq_f32(r3, vb);
                r4 = vmulq_f32(r4, vb);
                r5 = vmulq_f32(r5, vb);
                r6 = vmulq_f32(r6, vb);
                r7 = vmulq_f32(r7, vb);
            }
        }

        if (M>=8) {
            int k = (M/8)-1;
            x0 = vld1q_f32(x_ptr);

            __asm __volatile (
                "ldr	q2, [%[a_ptr], #0]\n"
                "ldr	q3, [%[a_ptr], #16]\n"
                "ldr	q4, [%[a_ptr], #32]\n"
                "ldr	q5, [%[a_ptr], #48]\n"
                "ldr	q6, [%[a_ptr], #64]\n"
                "ldr	q7, [%[a_ptr], #80]\n"
                "ldr	q8, [%[a_ptr], #96]\n"
                "ldr	q9, [%[a_ptr], #112]\n"
                "ldr	q10, [%[a_ptr], #128]\n"
                "ldr	q11, [%[a_ptr], #144]\n"
                "ldr	q12, [%[a_ptr], #160]\n"
                "ldr	q13, [%[a_ptr], #176]\n"
                "ldr	q14, [%[a_ptr], #192]\n"
                "ldr	q15, [%[a_ptr], #208]\n"
                "ldr	q16, [%[a_ptr], #224]\n"
                "ldr	q17, [%[a_ptr], #240]\n"
                "ldr	q18, [%[a_ptr], #256]\n"
                "ldr	q19, [%[a_ptr], #272]\n"
                "ldr	q20, [%[a_ptr], #288]\n"
                "ldr	q21, [%[a_ptr], #304]\n"
                "ldr	q22, [%[a_ptr], #320]\n"
                "ldr	q23, [%[a_ptr], #336]\n"
                ASM_PREFETCH("[%[a_ptr], #384]")
                ASM_PREFETCH("[%[a_ptr], #448]")
                ASM_PREFETCH("[%[a_ptr], #512]")
                ASM_PREFETCH("[%[a_ptr], #576]")
                ASM_PREFETCH("[%[a_ptr], #640]")
                ASM_PREFETCH("[%[a_ptr], #704]")
                ASM_PREFETCH("[%[a_ptr], #768]")
                ASM_PREFETCH("[%[a_ptr], #832]")
                ASM_PREFETCH("[%[a_ptr], #896]")
                ASM_PREFETCH("[%[a_ptr], #960]")
                ASM_PREFETCH("[%[a_ptr], #1024]")
                ASM_PREFETCH("[%[a_ptr], #1088]")
                ASM_PREFETCH("[%[a_ptr], #1152]")
                ASM_PREFETCH("[%[a_ptr], #1216]")
                ASM_PREFETCH("[%[a_ptr], #1280]")
                ASM_PREFETCH("[%[a_ptr], #1344]")
                ASM_PREFETCH("[%[a_ptr], #1408]")
                ASM_PREFETCH("[%[a_ptr], #1472]")
                ASM_PREFETCH("[%[a_ptr], #1536]")
                ASM_PREFETCH("[%[a_ptr], #1600]")
                ASM_PREFETCH("[%[a_ptr], #1664]")
                ASM_PREFETCH("[%[a_ptr], #1728]")
                ASM_PREFETCH("[%[a_ptr], #1792]")
                ASM_PREFETCH("[%[a_ptr], #1856]")
                ASM_PREFETCH("[%[a_ptr], #1920]")
                ASM_PREFETCH("[%[a_ptr], #1984]")
                "add	%[a_ptr], %[a_ptr], #352\n"

                "cbz	%w[k], 2f\n"

                "1:\n"
                // Unroll 0
                "fmla	%[r0].4s, v2.4s, %[x0].s[0]\n"
                "ldr	%q[x0a], [%[x_ptr], #16]\n"
                "fmla	%[r1].4s, v3.4s, %[x0].s[0]\n"
                "ldr	q3, [%[a_ptr], #0]\n"
                "subs	%w[k], %w[k], #1\n"
                "fmla	%[r2].4s, v4.4s, %[x0].s[0]\n"
                "ldr	q4, [%[a_ptr], #16]\n"
                "fmla	%[r3].4s, v5.4s, %[x0].s[0]\n"
                "ldr	q5, [%[a_ptr], #32]\n"
                "add	%[x_ptr], %[x_ptr], #32\n"
                ASM_PREFETCH("[%[a_ptr], #1664]")
                "fmla	%[r4].4s, v6.4s, %[x0].s[0]\n"
                "ldr	q6, [%[a_ptr], #48]\n"
                "fmla	%[r5].4s, v7.4s, %[x0].s[0]\n"
                "ldr	q7, [%[a_ptr], #64]\n"
                "fmla	%[r6].4s, v8.4s, %[x0].s[0]\n"
                "ldr	q8, [%[a_ptr], #80]\n"
                "fmla	%[r7].4s, v9.4s, %[x0].s[0]\n"
                "ldr	q9, [%[a_ptr], #96]\n"
                ASM_PREFETCH("[%[a_ptr], #1728]")

                // Unroll 1
                "fmla	%[r0].4s, v10.4s, %[x0].s[1]\n"
                "ldr	q10, [%[a_ptr], #112]\n"
                "fmla	%[r1].4s, v11.4s, %[x0].s[1]\n"
                "ldr	q11, [%[a_ptr], #128]\n"
                "fmla	%[r2].4s, v12.4s, %[x0].s[1]\n"
                "ldr	q12, [%[a_ptr], #144]\n"
                "fmla	%[r3].4s, v13.4s, %[x0].s[1]\n"
                "ldr	q13, [%[a_ptr], #160]\n"
                ASM_PREFETCH("[%[a_ptr], #1792]")
                "fmla	%[r4].4s, v14.4s, %[x0].s[1]\n"
                "ldr	q14, [%[a_ptr], #176]\n"
                "fmla	%[r5].4s, v15.4s, %[x0].s[1]\n"
                "ldr	q15, [%[a_ptr], #192]\n"
                "fmla	%[r6].4s, v16.4s, %[x0].s[1]\n"
                "ldr	q16, [%[a_ptr], #208]\n"
                "fmla	%[r7].4s, v17.4s, %[x0].s[1]\n"
                "ldr	q17, [%[a_ptr], #224]\n"
                ASM_PREFETCH("[%[a_ptr], #1856]")

                // Unroll 2
                "fmla	%[r0].4s, v18.4s, %[x0].s[2]\n"
                "ldr	q18, [%[a_ptr], #240]\n"
                "fmla	%[r1].4s, v19.4s, %[x0].s[2]\n"
                "ldr	q19, [%[a_ptr], #256]\n"
                "fmla	%[r2].4s, v20.4s, %[x0].s[2]\n"
                "ldr	q20, [%[a_ptr], #272]\n"
                "fmla	%[r3].4s, v21.4s, %[x0].s[2]\n"
                "ldr	q21, [%[a_ptr], #288]\n"
                ASM_PREFETCH("[%[a_ptr], #1920]")
                "fmla	%[r4].4s, v22.4s, %[x0].s[2]\n"
                "ldr	q22, [%[a_ptr], #304]\n"
                "fmla	%[r5].4s, v23.4s, %[x0].s[2]\n"
                "ldr	q23, [%[a_ptr], #320]\n"
                "fmla	%[r6].4s, v3.4s, %[x0].s[2]\n"
                "ldr	q2, [%[a_ptr], #336]\n"
                "ldr	q3, [%[a_ptr], #352]\n"
                "fmla	%[r7].4s, v4.4s, %[x0].s[2]\n"
                "ldr	q4, [%[a_ptr], #368]\n"
                ASM_PREFETCH("[%[a_ptr], #1984]")

                // Unroll 3
                "fmla	%[r0].4s, v5.4s, %[x0].s[3]\n"
                "ldr	q5, [%[a_ptr], #384]\n"
                "fmla	%[r1].4s, v6.4s, %[x0].s[3]\n"
                "ldr	q6, [%[a_ptr], #400]\n"
                "fmla	%[r2].4s, v7.4s, %[x0].s[3]\n"
                "ldr	q7, [%[a_ptr], #416]\n"
                "fmla	%[r3].4s, v8.4s, %[x0].s[3]\n"
                ASM_PREFETCH("[%[a_ptr], #2048]")
                "ldr	q8, [%[a_ptr], #432]\n"
                "fmla	%[r4].4s, v9.4s, %[x0].s[3]\n"
                "ldr	q9, [%[a_ptr], #448]\n"
                "fmla	%[r5].4s, v10.4s, %[x0].s[3]\n"
                "ldr	q10, [%[a_ptr], #464]\n"
                "fmla	%[r6].4s, v11.4s, %[x0].s[3]\n"
                "ldr	q11, [%[a_ptr], #480]\n"
                "fmla	%[r7].4s, v12.4s, %[x0].s[3]\n"
                "ldr	q12, [%[a_ptr], #496]\n"
                ASM_PREFETCH("[%[a_ptr], #2112]")

                // Unroll 4
                "fmla	%[r0].4s, v13.4s, %[x0a].s[0]\n"
                "ldr	%q[x0], [%[x_ptr]]\n"
                "fmla	%[r1].4s, v14.4s, %[x0a].s[0]\n"
                "ldr	q14, [%[a_ptr], #512]\n"
                "fmla	%[r2].4s, v15.4s, %[x0a].s[0]\n"
                "ldr	q15, [%[a_ptr], #528]\n"
                "fmla	%[r3].4s, v16.4s, %[x0a].s[0]\n"
                ASM_PREFETCH("[%[a_ptr], #2176]")
                "ldr	q16, [%[a_ptr], #544]\n"
                "fmla	%[r4].4s, v17.4s, %[x0a].s[0]\n"
                "ldr	q17, [%[a_ptr], #560]\n"
                "fmla	%[r5].4s, v18.4s, %[x0a].s[0]\n"
                "ldr	q18, [%[a_ptr], #576]\n"
                "fmla	%[r6].4s, v19.4s, %[x0a].s[0]\n"
                "ldr	q19, [%[a_ptr], #592]\n"
                "fmla	%[r7].4s, v20.4s, %[x0a].s[0]\n"
                "ldr	q20, [%[a_ptr], #608]\n"
                ASM_PREFETCH("[%[a_ptr], #2240]")

                // Unroll 5
                "fmla	%[r0].4s, v21.4s, %[x0a].s[1]\n"
                "ldr	q21, [%[a_ptr], #624]\n"
                "fmla	%[r1].4s, v22.4s, %[x0a].s[1]\n"
                "ldr	q22, [%[a_ptr], #640]\n"
                "fmla	%[r2].4s, v23.4s, %[x0a].s[1]\n"
                "ldr	q23, [%[a_ptr], #656]\n"
                "fmla	%[r3].4s, v2.4s, %[x0a].s[1]\n"
                "ldr	q2, [%[a_ptr], #672]\n"
                ASM_PREFETCH("[%[a_ptr], #2304]")
                "fmla	%[r4].4s, v3.4s, %[x0a].s[1]\n"
                "ldr	q3, [%[a_ptr], #688]\n"
                "fmla	%[r5].4s, v4.4s, %[x0a].s[1]\n"
                "ldr	q4, [%[a_ptr], #704]\n"
                "fmla	%[r6].4s, v5.4s, %[x0a].s[1]\n"
                "ldr	q5, [%[a_ptr], #720]\n"
                "fmla	%[r7].4s, v6.4s, %[x0a].s[1]\n"
                "ldr	q6, [%[a_ptr], #736]\n"
                ASM_PREFETCH("[%[a_ptr], #2368]")

                // Unroll 6
                "fmla	%[r0].4s, v7.4s, %[x0a].s[2]\n"
                "ldr	q7, [%[a_ptr], #752]\n"
                "fmla	%[r1].4s, v8.4s, %[x0a].s[2]\n"
                "ldr	q8, [%[a_ptr], #768]\n"
                "fmla	%[r2].4s, v9.4s, %[x0a].s[2]\n"
                "ldr	q9, [%[a_ptr], #784]\n"
                "fmla	%[r3].4s, v10.4s, %[x0a].s[2]\n"
                "ldr	q10, [%[a_ptr], #800]\n"
                ASM_PREFETCH("[%[a_ptr], #2432]")
                "fmla	%[r4].4s, v11.4s, %[x0a].s[2]\n"
                "ldr	q11, [%[a_ptr], #816]\n"
                "fmla	%[r5].4s, v12.4s, %[x0a].s[2]\n"
                "ldr	q12, [%[a_ptr], #832]\n"
                "fmla	%[r6].4s, v14.4s, %[x0a].s[2]\n"
                "ldr	q13, [%[a_ptr], #848]\n"
                "ldr	q14, [%[a_ptr], #864]\n"
                "fmla	%[r7].4s, v15.4s, %[x0a].s[2]\n"
                "ldr	q15, [%[a_ptr], #880]\n"
                ASM_PREFETCH("[%[a_ptr], #2496]")

                // Unroll 7
                "fmla	%[r0].4s, v16.4s, %[x0a].s[3]\n"
                "ldr	q16, [%[a_ptr], #896]\n"
                "fmla	%[r1].4s, v17.4s, %[x0a].s[3]\n"
                "ldr	q17, [%[a_ptr], #912]\n"
                "fmla	%[r2].4s, v18.4s, %[x0a].s[3]\n"
                "ldr	q18, [%[a_ptr], #928]\n"
                "fmla	%[r3].4s, v19.4s, %[x0a].s[3]\n"
                ASM_PREFETCH("[%[a_ptr], #2560]")
                "ldr	q19, [%[a_ptr], #944]\n"
                "fmla	%[r4].4s, v20.4s, %[x0a].s[3]\n"
                "ldr	q20, [%[a_ptr], #960]\n"
                "fmla	%[r5].4s, v21.4s, %[x0a].s[3]\n"
                "ldr	q21, [%[a_ptr], #976]\n"
                "add	%[a_ptr], %[a_ptr], #1024\n"
                "fmla	%[r6].4s, v22.4s, %[x0a].s[3]\n"
                "ldr	q22, [%[a_ptr], #-32]\n"
                "fmla	%[r7].4s, v23.4s, %[x0a].s[3]\n"
                "ldr	q23, [%[a_ptr], #-16]\n"
                ASM_PREFETCH("[%[a_ptr], #1600]")
                "bne	1b\n"

                // Detached final iteration
                "2:\n"

                // Unroll 0
                "fmla	%[r0].4s, v2.4s, %[x0].s[0]\n"
                "ldr	%q[x0a], [%[x_ptr], #16]\n"
                "fmla	%[r1].4s, v3.4s, %[x0].s[0]\n"
                "ldr	q3, [%[a_ptr], #0]\n"
                "subs	%w[k], %w[k], #1\n"
                "fmla	%[r2].4s, v4.4s, %[x0].s[0]\n"
                "ldr	q4, [%[a_ptr], #16]\n"
                "fmla	%[r3].4s, v5.4s, %[x0].s[0]\n"
                "ldr	q5, [%[a_ptr], #32]\n"
                "add	%[x_ptr], %[x_ptr], #32\n"
                "fmla	%[r4].4s, v6.4s, %[x0].s[0]\n"
                "ldr	q6, [%[a_ptr], #48]\n"
                "fmla	%[r5].4s, v7.4s, %[x0].s[0]\n"
                "ldr	q7, [%[a_ptr], #64]\n"
                "fmla	%[r6].4s, v8.4s, %[x0].s[0]\n"
                "ldr	q8, [%[a_ptr], #80]\n"
                "fmla	%[r7].4s, v9.4s, %[x0].s[0]\n"
                "ldr	q9, [%[a_ptr], #96]\n"

                // Unroll 1
                "fmla	%[r0].4s, v10.4s, %[x0].s[1]\n"
                "ldr	q10, [%[a_ptr], #112]\n"
                "fmla	%[r1].4s, v11.4s, %[x0].s[1]\n"
                "ldr	q11, [%[a_ptr], #128]\n"
                "fmla	%[r2].4s, v12.4s, %[x0].s[1]\n"
                "ldr	q12, [%[a_ptr], #144]\n"
                "fmla	%[r3].4s, v13.4s, %[x0].s[1]\n"
                "ldr	q13, [%[a_ptr], #160]\n"
                "fmla	%[r4].4s, v14.4s, %[x0].s[1]\n"
                "ldr	q14, [%[a_ptr], #176]\n"
                "fmla	%[r5].4s, v15.4s, %[x0].s[1]\n"
                "ldr	q15, [%[a_ptr], #192]\n"
                "fmla	%[r6].4s, v16.4s, %[x0].s[1]\n"
                "ldr	q16, [%[a_ptr], #208]\n"
                "fmla	%[r7].4s, v17.4s, %[x0].s[1]\n"
                "ldr	q17, [%[a_ptr], #224]\n"

                // Unroll 2
                "fmla	%[r0].4s, v18.4s, %[x0].s[2]\n"
                "ldr	q18, [%[a_ptr], #240]\n"
                "fmla	%[r1].4s, v19.4s, %[x0].s[2]\n"
                "ldr	q19, [%[a_ptr], #256]\n"
                "fmla	%[r2].4s, v20.4s, %[x0].s[2]\n"
                "ldr	q20, [%[a_ptr], #272]\n"
                "fmla	%[r3].4s, v21.4s, %[x0].s[2]\n"
                "ldr	q21, [%[a_ptr], #288]\n"
                "fmla	%[r4].4s, v22.4s, %[x0].s[2]\n"
                "ldr	q22, [%[a_ptr], #304]\n"
                "fmla	%[r5].4s, v23.4s, %[x0].s[2]\n"
                "ldr	q23, [%[a_ptr], #320]\n"
                "fmla	%[r6].4s, v3.4s, %[x0].s[2]\n"
                "ldr	q2, [%[a_ptr], #336]\n"
                "ldr	q3, [%[a_ptr], #352]\n"
                "fmla	%[r7].4s, v4.4s, %[x0].s[2]\n"
                "ldr	q4, [%[a_ptr], #368]\n"

                // Unroll 3
                "fmla	%[r0].4s, v5.4s, %[x0].s[3]\n"
                "ldr	q5, [%[a_ptr], #384]\n"
                "fmla	%[r1].4s, v6.4s, %[x0].s[3]\n"
                "ldr	q6, [%[a_ptr], #400]\n"
                "fmla	%[r2].4s, v7.4s, %[x0].s[3]\n"
                "ldr	q7, [%[a_ptr], #416]\n"
                "fmla	%[r3].4s, v8.4s, %[x0].s[3]\n"
                "ldr	q8, [%[a_ptr], #432]\n"
                "fmla	%[r4].4s, v9.4s, %[x0].s[3]\n"
                "ldr	q9, [%[a_ptr], #448]\n"
                "fmla	%[r5].4s, v10.4s, %[x0].s[3]\n"
                "ldr	q10, [%[a_ptr], #464]\n"
                "fmla	%[r6].4s, v11.4s, %[x0].s[3]\n"
                "ldr	q11, [%[a_ptr], #480]\n"
                "fmla	%[r7].4s, v12.4s, %[x0].s[3]\n"
                "ldr	q12, [%[a_ptr], #496]\n"

                // Unroll 4
                "fmla	%[r0].4s, v13.4s, %[x0a].s[0]\n"
                "fmla	%[r1].4s, v14.4s, %[x0a].s[0]\n"
                "ldr	q14, [%[a_ptr], #512]\n"
                "fmla	%[r2].4s, v15.4s, %[x0a].s[0]\n"
                "ldr	q15, [%[a_ptr], #528]\n"
                "fmla	%[r3].4s, v16.4s, %[x0a].s[0]\n"
                "ldr	q16, [%[a_ptr], #544]\n"
                "fmla	%[r4].4s, v17.4s, %[x0a].s[0]\n"
                "ldr	q17, [%[a_ptr], #560]\n"
                "fmla	%[r5].4s, v18.4s, %[x0a].s[0]\n"
                "ldr	q18, [%[a_ptr], #576]\n"
                "fmla	%[r6].4s, v19.4s, %[x0a].s[0]\n"
                "ldr	q19, [%[a_ptr], #592]\n"
                "fmla	%[r7].4s, v20.4s, %[x0a].s[0]\n"
                "ldr	q20, [%[a_ptr], #608]\n"

                // Unroll 5
                "fmla	%[r0].4s, v21.4s, %[x0a].s[1]\n"
                "ldr	q21, [%[a_ptr], #624]\n"
                "fmla	%[r1].4s, v22.4s, %[x0a].s[1]\n"
                "ldr	q22, [%[a_ptr], #640]\n"
                "fmla	%[r2].4s, v23.4s, %[x0a].s[1]\n"
                "ldr	q23, [%[a_ptr], #656]\n"
                "fmla	%[r3].4s, v2.4s, %[x0a].s[1]\n"
                "add	%[a_ptr], %[a_ptr], #672\n"
                "fmla	%[r4].4s, v3.4s, %[x0a].s[1]\n"
                "fmla	%[r5].4s, v4.4s, %[x0a].s[1]\n"
                "fmla	%[r6].4s, v5.4s, %[x0a].s[1]\n"
                "fmla	%[r7].4s, v6.4s, %[x0a].s[1]\n"

                // Unroll 6
                "fmla	%[r0].4s, v7.4s, %[x0a].s[2]\n"
                "fmla	%[r1].4s, v8.4s, %[x0a].s[2]\n"
                "fmla	%[r2].4s, v9.4s, %[x0a].s[2]\n"
                "fmla	%[r3].4s, v10.4s, %[x0a].s[2]\n"
                "fmla	%[r4].4s, v11.4s, %[x0a].s[2]\n"
                "fmla	%[r5].4s, v12.4s, %[x0a].s[2]\n"
                "fmla	%[r6].4s, v14.4s, %[x0a].s[2]\n"
                "fmla	%[r7].4s, v15.4s, %[x0a].s[2]\n"

                // Unroll 7
                "fmla	%[r0].4s, v16.4s, %[x0a].s[3]\n"
                "fmla	%[r1].4s, v17.4s, %[x0a].s[3]\n"
                "fmla	%[r2].4s, v18.4s, %[x0a].s[3]\n"
                "fmla	%[r3].4s, v19.4s, %[x0a].s[3]\n"
                "fmla	%[r4].4s, v20.4s, %[x0a].s[3]\n"
                "fmla	%[r5].4s, v21.4s, %[x0a].s[3]\n"
                "fmla	%[r6].4s, v22.4s, %[x0a].s[3]\n"
                "fmla	%[r7].4s, v23.4s, %[x0a].s[3]\n"
            :
              [a_ptr] "+r" (a_ptr), [x_ptr] "+r" (x_ptr),
              [x0] "+w" (x0), [x0a] "+w" (x0a), [k] "+r" (k),
              [r0] "+w" (r0), [r1] "+w" (r1), [r2] "+w" (r2), [r3] "+w" (r3),
              [r4] "+w" (r4), [r5] "+w" (r5), [r6] "+w" (r6), [r7] "+w" (r7)
            :
            : "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
              "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "x20", "x21", "cc", "memory");
        }

        // Deal with ragged M
        if (M % 8) {
            int l=(M%8)-1;

            __asm __volatile (
                "ldr	q2, [%[a_ptr], #0]\n"
                "ldr	q3, [%[a_ptr], #16]\n"
                "ldr	q4, [%[a_ptr], #32]\n"
                "ldr	q5, [%[a_ptr], #48]\n"
                "ldr	q6, [%[a_ptr], #64]\n"
                "ldr	q7, [%[a_ptr], #80]\n"
                "ldr	q8, [%[a_ptr], #96]\n"
                "ldr	q9, [%[a_ptr], #112]\n"
                "ldr	%s[x0], [%[x_ptr]]\n"
                "add	%[a_ptr], %[a_ptr], #128\n"
                "add	%[x_ptr], %[x_ptr], #4\n"

                "cbz	%w[l], 2f\n"

                "1:\n"
                "fmla	%[r0].4s, v2.4s, %[x0].s[0]\n"
                "ldr	q2, [%[a_ptr], #0]\n"
                "subs	%w[l], %w[l], #1\n"
                "fmla	%[r1].4s, v3.4s, %[x0].s[0]\n"
                "ldr	q3, [%[a_ptr], #16]\n"
                "fmla	%[r2].4s, v4.4s, %[x0].s[0]\n"
                "ldr	q4, [%[a_ptr], #32]\n"
                "fmla	%[r3].4s, v5.4s, %[x0].s[0]\n"
                "ldr	q5, [%[a_ptr], #48]\n"
                "fmla	%[r4].4s, v6.4s, %[x0].s[0]\n"
                "ldr	q6, [%[a_ptr], #64]\n"
                "fmla	%[r5].4s, v7.4s, %[x0].s[0]\n"
                "ldr	q7, [%[a_ptr], #80]\n"
                "fmla	%[r6].4s, v8.4s, %[x0].s[0]\n"
                "ldr	q8, [%[a_ptr], #96]\n"
                "fmla	%[r7].4s, v9.4s, %[x0].s[0]\n"
                "ldr	q9, [%[a_ptr], #112]\n"
                "ldr	%s[x0], [%[x_ptr]]\n"
                "add	%[a_ptr], %[a_ptr], #128\n"
                "add	%[x_ptr], %[x_ptr], #4\n"
                "bne	1b\n"

                "2:\n"

                "fmla	%[r0].4s, v2.4s, %[x0].s[0]\n"
                "fmla	%[r1].4s, v3.4s, %[x0].s[0]\n"
                "fmla	%[r2].4s, v4.4s, %[x0].s[0]\n"
                "fmla	%[r3].4s, v5.4s, %[x0].s[0]\n"
                "fmla	%[r4].4s, v6.4s, %[x0].s[0]\n"
                "fmla	%[r5].4s, v7.4s, %[x0].s[0]\n"
                "fmla	%[r6].4s, v8.4s, %[x0].s[0]\n"
                "fmla	%[r7].4s, v9.4s, %[x0].s[0]\n"
            :
              [a_ptr] "+r" (a_ptr), [x_ptr] "+r" (x_ptr),
              [x0] "+w" (x0), [l] "+r" (l),
              [r0] "+w" (r0), [r1] "+w" (r1), [r2] "+w" (r2), [r3] "+w" (r3),
              [r4] "+w" (r4), [r5] "+w" (r5), [r6] "+w" (r6), [r7] "+w" (r7)
            :
            : "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc", "memory");
        }

        if (l==32) {
            // Fast path
            vst1q_f32(y_ptr, r0);
            vst1q_f32(y_ptr + 4, r1);
            vst1q_f32(y_ptr + 8, r2);
            vst1q_f32(y_ptr + 12, r3);
            vst1q_f32(y_ptr + 16, r4);
            vst1q_f32(y_ptr + 20, r5);
            vst1q_f32(y_ptr + 24, r6);
            vst1q_f32(y_ptr + 28, r7);
        } else {
            int vecs=l/4;
            int oddbits=l%4;

            if (oddbits) {
                // As above - slowest path deals with vectors plus odd bits
                float32x4_t oddvec;

                do {
                    if (vecs==0) { oddvec=r0; break; }

                    vst1q_f32(y_ptr, r0);
                    if (--vecs==0) { oddvec=r1; break; }

                    vst1q_f32(y_ptr + 4, r1);
                    if (--vecs==0) { oddvec=r2; break; }

                    vst1q_f32(y_ptr + 8, r2);
                    if (--vecs==0) { oddvec=r3; break; }

                    vst1q_f32(y_ptr + 12, r3);
                    if (--vecs==0) { oddvec=r4; break; }

                    vst1q_f32(y_ptr + 16, r4);
                    if (--vecs==0) { oddvec=r5; break; }

                    vst1q_f32(y_ptr + 20, r5);
                    if (--vecs==0) { oddvec=r6; break; }

                    vst1q_f32(y_ptr + 24, r6);
                    oddvec=r7;
                } while (0);

                float *oddbase = y_ptr + l - oddbits;

                switch(oddbits) {
                    case 3:
                        vst1q_lane_f32(oddbase + 2, oddvec, 2);
                        // fall through
                    case 2:
                        vst1q_lane_f32(oddbase + 1, oddvec, 1);
                        // fall through
                    case 1:
                        vst1q_lane_f32(oddbase, oddvec, 0);
                        break;

                    default:
                        // oddbits must be 1, 2 or 3.
                        UNREACHABLE("Impossible case in switch.");
                }
            } else {
                // As above - medium path deals with vectors only
                do {
                    if (vecs==0) { UNREACHABLE("vecs and oddbits can't both be 0"); }

                    vst1q_f32(y_ptr, r0);
                    if (--vecs==0) { break; }

                    vst1q_f32(y_ptr + 4, r1);
                    if (--vecs==0) { break; }

                    vst1q_f32(y_ptr + 8, r2);
                    if (--vecs==0) { break; }

                    vst1q_f32(y_ptr + 12, r3);
                    if (--vecs==0) { break; }

                    vst1q_f32(y_ptr + 16, r4);
                    if (--vecs==0) { break; }

                    vst1q_f32(y_ptr + 20, r5);
                    if (--vecs==0) { break; }

                    vst1q_f32(y_ptr + 24, r6);
                } while (0);
            }
        }
    }
}

} // namespace arm_gemm

#endif // aarch64
