//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot)

    // -----------------------------------------------------------------------------
    // Args (x0 = args_ptr)
    //   0   (0x00)
    //   8   (0x08)
    //  16   (0x10)
    //  24   (0x18)
    //  32   (0x20)
    //  40   (0x28)
    //  48   (0x30)
    //  56   (0x38)
    //  64   (0x40)
    //  68   (0x44)
    //  72   (0x48)
    //  80   (0x50)
    //  88   (0x58)
    //  96   (0x60)
    // 104   (0x68)
    // 112   (0x70)
    // 120   (0x78)
    // -----------------------------------------------------------------------------
    // GPR usage
    //  x0/x16  args_ptr
    //  x19     n
    //  x20     lut_ptr (→ zt0)
    //  x21     lhs_ptr
    //  x22     row index
    //  x23     k offset
    //  x24     rhs_ptr
    //  x25     rhs_ptr_end
    //  x26     rhs_tile_base
    //  x27     dst_ptr
    //  x28     tmp
    //  x5/x7/x8  stride, bl, k params
    // -----------------------------------------------------------------------------
    // Z / ZA usage
    //  zt0     lut descriptor
    //  z0      lhs broadcast
    //  z16–z23 rhs nybble inputs
    //  z24–z27 rhs lut outputs / scales
    //  z4–z7   za extract
    //  z12–z15 final outputs
    //  z28–z31 temp accum
    // -----------------------------------------------------------------------------
    // Predicates
    //  p0.b    all true
    //  pn8.b   all true (bytes)
    //  pn9.s   n active (vlx4)
    //  pn10.b  n sub-tile 0
    //  pn11.b  n sub-tile 1
    //  pn12.s  m active (vlx4)
    // -----------------------------------------------------------------------------
    // Loop structure
    //  m_loop: M rows
    //    n_loop: N tiles (vlx4)
    //      k_loop: K blocks (size bl)
    //        zero ZA
    //        ld1rqb LHS
    //        ld1b + luti4 RHS
    //        sdot accumulate
    //      load RHS scales (lsl#16)
    //      extract ZA → f32
    //      scale, bias, clamp, store
    // -----------------------------------------------------------------------------

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot)
    stp     x19, x20, [sp, #-144]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #112]
    stp     x27, x28, [sp, #128]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]

    mov     x16, x0                       // args_ptr

    KAI_ASM_INST(0xd503477f)              // smstart
    mov     x11, #0                       // zero index for predicate formation (pn9)
    ldr     x7,  [x16, #120]
    ldr     x8,  [x16, #72]
    ldr     x15, [x16, #32]
    ldr     x19, [x16, #96]
    ldr     x21, [x16, #48]
    ldr     x20, [x16, #16]
    KAI_ASM_INST(0xe11f8280)              // ldr     zt0, [x20]
    ptrue   p0.b
    KAI_ASM_INST(0x25207810)              // ptrue   pn8.b
    KAI_ASM_INST(0x25b36571)              // whilelt pn9.s, x11, x19, vlx4
    KAI_ASM_INST(0x8551c201)              // ld1rw   z1.s, p0/z, [x16, #68]
    KAI_ASM_INST(0x8550c202)              // ld1rw   z2.s, p0/z, [x16, #64]
KAI_ASM_LABEL(m_loop)                     // LHS matrix row loop
    ldr     x26, [x16, #56]
    ldr     x27, [x16, #0]
    mov     x22, #0
    KAI_ASM_INST(0x25af66d4)              // whilelt pn12.s, x22, x15, vlx4
KAI_ASM_LABEL(n_loop)                     // RHS matrix row loop (N-tile iteration)
    mov     x24, x26
    ldr     x5,  [x16, #104]
    add     x25, x26, x5
    KAI_ASM_INST(0x25396712)              // whilelt pn10.b, x24, x25, vlx4
    addvl   x28, x24, #4
    KAI_ASM_INST(0x25396793)              // whilelt pn11.b, x28, x25, vlx4
    mov     x23, #0
    whilelt p1.b, x23, x8
    dup z28.s, #0
    dup z29.s, #0
    dup z30.s, #0
    dup z31.s, #0
KAI_ASM_LABEL(k_loop)
    mov     x13, x7
    KAI_ASM_INST(0xc00800ff)              // zero    {za}

    // Each k_block processes 16 bytes of RHS (4 LUT-nybble groups) and 16 LHS elements.
    // Accumulates into za.s[w11,0,vgx4] using sdot.
    // Loop unrolled for 4 LUT lookups and 4 sdot operations per iteration.
KAI_ASM_LABEL(k_block_loop)
    ld1rqb  { z0.b }, p1/z , [x21, x23]
    add     x23, x23, #16
    KAI_ASM_INST(0xa0408b10)              // ld1b    { z16.b - z19.b }, pn10/z, [x24]
    KAI_ASM_INST(0xa0418f14)              // ld1b {z20.b-z23.b}, pn11/z, [x24,#0x4, mul vl]
    KAI_ASM_INST(0xc08a4218)              // luti4   { z24.b, z25.b }, zt0, z16[0]
    KAI_ASM_INST(0xc08a423a)              // luti4   { z26.b, z27.b }, zt0, z17[0]
    KAI_ASM_INST(0xc150f320)              // sdot za.s[w11,0, vgx4], {z24.b-z27.b}, z0.b[0]
    KAI_ASM_INST(0xc08a4244)              // luti4   { z4.b, z5.b }, zt0, z18[0]
    KAI_ASM_INST(0xc08a4266)              // luti4   { z6.b, z7.b }, zt0, z19[0]
    KAI_ASM_INST(0xc150f4a0)              // sdot za.s[w11,0, vgx4], {z4.b-z7.b}, z0.b[1]
    KAI_ASM_INST(0xc08a4288)              // luti4   { z8.b, z9.b }, zt0, z20[0]
    KAI_ASM_INST(0xc08a42aa)              // luti4   { z10.b, z11.b }, zt0, z21[0]
    KAI_ASM_INST(0xc150f920)              // sdot za.s[w11,0, vgx4], {z8.b-z11.b}, z0.b[2]
    KAI_ASM_INST(0xc08a42cc)              // luti4   { z12.b, z13.b }, zt0, z22[0]
    KAI_ASM_INST(0xc08a42ee)              // luti4   { z14.b, z15.b }, zt0, z23[0]
    KAI_ASM_INST(0xc150fda0)              // sdot za.s[w11,0, vgx4], {z12.b-z15.b}, z0.b[3]
    addvl   x24, x24, #8
    KAI_ASM_INST(0x25396712)              // whilelt pn10.b, x24, x25, vlx4
    addvl   x28, x24, #4
    KAI_ASM_INST(0x25396793)              // whilelt pn11.b, x28, x25, vlx4
    subs x13, x13, #16
    b.gt k_block_loop                     // To block loop start

    // Load RHS scales (s16), shift left to Q15 in s32 lanes
    ld1h z24.s , p0/z, [x24]
    ld1h z25.s , p0/z, [x24,  #1, MUL VL]
    ld1h z26.s , p0/z, [x24,  #2, MUL VL]
    ld1h z27.s , p0/z, [x24,  #3, MUL VL]
    addvl   x24, x24, #2
    KAI_ASM_INST(0x25396712)              // whilelt pn10.b, x24, x25, vlx4
    addvl   x28, x24, #4
    KAI_ASM_INST(0x25396793)              // whilelt pn11.b, x28, x25, vlx4
    // Q15 conversion step
    lsl z24.s, z24.s, #16
    lsl z25.s, z25.s, #16
    lsl z26.s, z26.s, #16
    lsl z27.s, z27.s, #16
    // Extract 4 partial accumulators from ZA row w11
    KAI_ASM_INST(0xc0066c04)              // mov     { z4.d - z7.d }, za.d[w11, 0, vgx4]
    KAI_ASM_INST(0xc132e084)              // scvtf   { z4.s - z7.s }, { z4.s - z7.s }
    fmla z28.s, p0/m, z4.s, z24.s
    fmla z29.s, p0/m, z5.s, z25.s
    fmla z30.s, p0/m, z6.s, z26.s
    fmla z31.s, p0/m, z7.s, z27.s
    whilelt p1.b, x23, x8
    b.mi k_loop                           // b.first k_loop - next K tile (if any)
    add     x28, x21, x8
    KAI_ASM_INST(0x8540c380)              // ld1rw   { z0.s }, p0/z , [x28]
    KAI_ASM_INST(0x8541c383)              // ld1rw   { z3.s }, p0/z , [x28, #4]
    ldr     x5, [x16, #104]
    add     x28, x26, x5
    KAI_ASM_INST(0xa040c794)              // ld1w    { z20.s - z23.s }, pn9/z, [x28]
    KAI_ASM_INST(0xa041c78c)              // ld1w {z12.s-z15.s}, pn9/z, [x28, #0x4, mul vl]
    scvtf z0.s, p0/m, z0.s
    fmla     z28.s, p0/m, z20.s, z0.s
    fmla     z29.s, p0/m, z21.s, z0.s
    fmla     z30.s, p0/m, z22.s, z0.s
    fmla     z31.s, p0/m, z23.s, z0.s
    fmla    z12.s, p0/m,  z28.s, z3.s
    fmla    z13.s, p0/m,  z29.s, z3.s
    fmla    z14.s, p0/m,  z30.s, z3.s
    fmla    z15.s, p0/m,  z31.s, z3.s
    KAI_ASM_INST(0xc1a2c82c)              // fclamp  { z12.s - z15.s }, z1.s, z2.s
    KAI_ASM_INST(0xa036d36c)              // st1w {z12.s-z15.s}, pn12, [x27, x22, lsl #2]
    ldr x5, [x16, #88]
    add     x26, x26, x5
    addvl   x22, x22, #1
    KAI_ASM_INST(0x25af66d4)              // whilelt pn12.s, x22, x15, vlx4
    b.lt    n_loop                        // To RHS matrix row loop start
    ldr x5, [x16, #8]
    add     x27, x27, x5
    ldr x5, [x16, #80]
    add     x21, x21, x5
    ldr x5, [x16, #112]
    cmp     x21, x5
    b.lt    m_loop                        // To LHS matrix row loop start
    KAI_ASM_INST(0xd503467f)	          // smstop

    ldp     d14, d15, [sp, #96]
    ldp     d12, d13, [sp, #80]
    ldp     d10, d11, [sp, #64]
    ldp     d8,  d9,  [sp, #48]
    ldp     x27, x28, [sp, #128]
    ldp     x25, x26, [sp, #112]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme2_dot)

    KAI_ASM_END
