//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    mov x5, #0x80
    mov x21, #0x20
    sub SP, SP, #0x100
    ldr x20, [x0, #0x28]
    ldr x6, [x0, #0x40]
    ldr x7, [x0, #0x38]
    ldr x8, [x0, #0x8]
    ldr x17, [x0, #0x10]
    ldr x16, [x0, #0x30]
    mov x15, x20
    mul x5, x6, x5
    ldr x14, [x0, #0x0]
    ldr x13, [x0, #0x20]
    ldr x12, [x0, #0x18]
    cmp x15, #0x10
    madd x5, x7, x5, x21
    blt label_15
KAI_ASM_LABEL(label_1)  // Row loop
    mov x11, x17
    mov x10, x16
    add x9, x14, x13, LSL #4
KAI_ASM_LABEL(label_2)  // Column loop
    mov x27, x8
    movi v6.4s, #0x0
    mov x24, x7
    str q6, [SP, #0x0]
    str q6, [SP, #0x10]
    str q6, [SP, #0x20]
    add x23, x27, x5
    add x22, x23, x5
    str q6, [SP, #0x30]
    add x21, x22, x5
    str q6, [SP, #0x40]
    str q6, [SP, #0x50]
    str q6, [SP, #0x60]
    str q6, [SP, #0x70]
    str q6, [SP, #0x80]
    str q6, [SP, #0x90]
    str q6, [SP, #0xa0]
    str q6, [SP, #0xb0]
    str q6, [SP, #0xc0]
    str q6, [SP, #0xd0]
    str q6, [SP, #0xe0]
    str q6, [SP, #0xf0]
KAI_ASM_LABEL(label_3)  // Block loop
    movi v2.4s, #0x0
    movi v17.4s, #0x0
    mov x20, x6
    movi v12.4s, #0x0
    movi v9.4s, #0x0
    movi v14.4s, #0x0
    movi v11.4s, #0x0
    movi v13.4s, #0x0
    movi v15.4s, #0x0
    movi v23.4s, #0x0
    movi v29.4s, #0x0
    movi v0.4s, #0x0
    movi v4.4s, #0x0
    movi v16.4s, #0x0
    movi v21.4s, #0x0
    movi v10.4s, #0x0
    movi v3.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q6, [x11, #0x0]
    ldr q1, [x27, #0x0]
    movi v25.16b, #0xf0
    subs x20, x20, #0x1
    ldr q5, [x23, #0x0]
    ldr q30, [x22, #0x0]
    ldr q24, [x21, #0x0]
    ldr q18, [x11, #0x10]
    ldr q27, [x27, #0x10]
    ldr q20, [x23, #0x10]
    shl v31.16b, v6.16b, #0x4
    and v6.16b, v6.16b, v25.16b
    ldr q19, [x22, #0x10]
    ldr q26, [x21, #0x10]
    ldr q7, [x11, #0x20]
    ldr q8, [x27, #0x20]
    shl v22.16b, v18.16b, #0x4
    and v18.16b, v18.16b, v25.16b
    ldr q28, [x23, #0x20]
    KAI_ASM_INST(0x4f81e3e2)  // sdot v2.4s, v31.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e3f1)  // sdot v17.4s, v31.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81ebec)  // sdot v12.4s, v31.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1ebe9)  // sdot v9.4s, v31.16b, v1.4b[3]
    ldr q1, [x22, #0x20]
    KAI_ASM_INST(0x4f85e3ee)  // sdot v14.4s, v31.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e3eb)  // sdot v11.4s, v31.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85ebed)  // sdot v13.4s, v31.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5ebef)  // sdot v15.4s, v31.16b, v5.4b[3]
    ldr q5, [x21, #0x20]
    KAI_ASM_INST(0x4f9ee3f7)  // sdot v23.4s, v31.16b, v30.4b[0]
    KAI_ASM_INST(0x4fbee3fd)  // sdot v29.4s, v31.16b, v30.4b[1]
    KAI_ASM_INST(0x4f9eebe0)  // sdot v0.4s, v31.16b, v30.4b[2]
    KAI_ASM_INST(0x4fbeebe4)  // sdot v4.4s, v31.16b, v30.4b[3]
    ldr q30, [x11, #0x30]
    add x11, x11, #0x40
    KAI_ASM_INST(0x4f98e3f0)  // sdot v16.4s, v31.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e3f5)  // sdot v21.4s, v31.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ebea)  // sdot v10.4s, v31.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ebe3)  // sdot v3.4s, v31.16b, v24.4b[3]
    ldr q24, [x27, #0x30]
    ldr q31, [x23, #0x30]
    KAI_ASM_INST(0x4f9be2c2)  // sdot v2.4s, v22.16b, v27.4b[0]
    KAI_ASM_INST(0x4fbbe2d1)  // sdot v17.4s, v22.16b, v27.4b[1]
    KAI_ASM_INST(0x4f9beacc)  // sdot v12.4s, v22.16b, v27.4b[2]
    KAI_ASM_INST(0x4fbbeac9)  // sdot v9.4s, v22.16b, v27.4b[3]
    ldr q27, [x22, #0x30]
    KAI_ASM_INST(0x4f94e2ce)  // sdot v14.4s, v22.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e2cb)  // sdot v11.4s, v22.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94eacd)  // sdot v13.4s, v22.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4eacf)  // sdot v15.4s, v22.16b, v20.4b[3]
    ldr q20, [x21, #0x30]
    KAI_ASM_INST(0x4f93e2d7)  // sdot v23.4s, v22.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e2dd)  // sdot v29.4s, v22.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93eac0)  // sdot v0.4s, v22.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eac4)  // sdot v4.4s, v22.16b, v19.4b[3]
    ldr q19, [x27, #0x40]
    KAI_ASM_INST(0x4f9ae2d0)  // sdot v16.4s, v22.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae2d5)  // sdot v21.4s, v22.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aeaca)  // sdot v10.4s, v22.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaeac3)  // sdot v3.4s, v22.16b, v26.4b[3]
    ldr q22, [x23, #0x40]
    shl v26.16b, v7.16b, #0x4
    and v7.16b, v7.16b, v25.16b
    KAI_ASM_INST(0x4f88e342)  // sdot v2.4s, v26.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e351)  // sdot v17.4s, v26.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88eb4c)  // sdot v12.4s, v26.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8eb49)  // sdot v9.4s, v26.16b, v8.4b[3]
    ldr q8, [x22, #0x40]
    KAI_ASM_INST(0x4f9ce34e)  // sdot v14.4s, v26.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce34b)  // sdot v11.4s, v26.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9ceb4d)  // sdot v13.4s, v26.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbceb4f)  // sdot v15.4s, v26.16b, v28.4b[3]
    ldr q28, [x21, #0x40]
    KAI_ASM_INST(0x4f81e357)  // sdot v23.4s, v26.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e35d)  // sdot v29.4s, v26.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81eb40)  // sdot v0.4s, v26.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1eb44)  // sdot v4.4s, v26.16b, v1.4b[3]
    ldr q1, [x27, #0x50]
    KAI_ASM_INST(0x4f85e350)  // sdot v16.4s, v26.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e355)  // sdot v21.4s, v26.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85eb4a)  // sdot v10.4s, v26.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5eb43)  // sdot v3.4s, v26.16b, v5.4b[3]
    ldr q5, [x23, #0x50]
    shl v26.16b, v30.16b, #0x4
    and v30.16b, v30.16b, v25.16b
    ldr q25, [x22, #0x50]
    KAI_ASM_INST(0x4f98e342)  // sdot v2.4s, v26.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e351)  // sdot v17.4s, v26.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98eb4c)  // sdot v12.4s, v26.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8eb49)  // sdot v9.4s, v26.16b, v24.4b[3]
    ldr q24, [x21, #0x50]
    KAI_ASM_INST(0x4f9fe34e)  // sdot v14.4s, v26.16b, v31.4b[0]
    KAI_ASM_INST(0x4fbfe34b)  // sdot v11.4s, v26.16b, v31.4b[1]
    KAI_ASM_INST(0x4f9feb4d)  // sdot v13.4s, v26.16b, v31.4b[2]
    KAI_ASM_INST(0x4fbfeb4f)  // sdot v15.4s, v26.16b, v31.4b[3]
    ldr q31, [x27, #0x60]
    KAI_ASM_INST(0x4f9be357)  // sdot v23.4s, v26.16b, v27.4b[0]
    KAI_ASM_INST(0x4fbbe35d)  // sdot v29.4s, v26.16b, v27.4b[1]
    KAI_ASM_INST(0x4f9beb40)  // sdot v0.4s, v26.16b, v27.4b[2]
    KAI_ASM_INST(0x4fbbeb44)  // sdot v4.4s, v26.16b, v27.4b[3]
    ldr q27, [x23, #0x60]
    KAI_ASM_INST(0x4f94e350)  // sdot v16.4s, v26.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e355)  // sdot v21.4s, v26.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94eb4a)  // sdot v10.4s, v26.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4eb43)  // sdot v3.4s, v26.16b, v20.4b[3]
    ldr q26, [x22, #0x60]
    ldr q20, [x21, #0x60]
    KAI_ASM_INST(0x4f93e0c2)  // sdot v2.4s, v6.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e0d1)  // sdot v17.4s, v6.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93e8cc)  // sdot v12.4s, v6.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3e8c9)  // sdot v9.4s, v6.16b, v19.4b[3]
    ldr q19, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f96e0ce)  // sdot v14.4s, v6.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e0cb)  // sdot v11.4s, v6.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96e8cd)  // sdot v13.4s, v6.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6e8cf)  // sdot v15.4s, v6.16b, v22.4b[3]
    ldr q22, [x23, #0x70]
    add x23, x23, #0x80
    KAI_ASM_INST(0x4f88e0d7)  // sdot v23.4s, v6.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e0dd)  // sdot v29.4s, v6.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e8c0)  // sdot v0.4s, v6.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e8c4)  // sdot v4.4s, v6.16b, v8.4b[3]
    ldr q8, [x22, #0x70]
    add x22, x22, #0x80
    KAI_ASM_INST(0x4f9ce0d0)  // sdot v16.4s, v6.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce0d5)  // sdot v21.4s, v6.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9ce8ca)  // sdot v10.4s, v6.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbce8c3)  // sdot v3.4s, v6.16b, v28.4b[3]
    ldr q28, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4f81e242)  // sdot v2.4s, v18.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e251)  // sdot v17.4s, v18.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81ea4c)  // sdot v12.4s, v18.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1ea49)  // sdot v9.4s, v18.16b, v1.4b[3]
    KAI_ASM_INST(0x4f85e24e)  // sdot v14.4s, v18.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e24b)  // sdot v11.4s, v18.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85ea4d)  // sdot v13.4s, v18.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5ea4f)  // sdot v15.4s, v18.16b, v5.4b[3]
    KAI_ASM_INST(0x4f99e257)  // sdot v23.4s, v18.16b, v25.4b[0]
    KAI_ASM_INST(0x4fb9e25d)  // sdot v29.4s, v18.16b, v25.4b[1]
    KAI_ASM_INST(0x4f99ea40)  // sdot v0.4s, v18.16b, v25.4b[2]
    KAI_ASM_INST(0x4fb9ea44)  // sdot v4.4s, v18.16b, v25.4b[3]
    KAI_ASM_INST(0x4f98e250)  // sdot v16.4s, v18.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e255)  // sdot v21.4s, v18.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ea4a)  // sdot v10.4s, v18.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ea43)  // sdot v3.4s, v18.16b, v24.4b[3]
    KAI_ASM_INST(0x4f9fe0e2)  // sdot v2.4s, v7.16b, v31.4b[0]
    KAI_ASM_INST(0x4fbfe0f1)  // sdot v17.4s, v7.16b, v31.4b[1]
    KAI_ASM_INST(0x4f9fe8ec)  // sdot v12.4s, v7.16b, v31.4b[2]
    KAI_ASM_INST(0x4fbfe8e9)  // sdot v9.4s, v7.16b, v31.4b[3]
    KAI_ASM_INST(0x4f9be0ee)  // sdot v14.4s, v7.16b, v27.4b[0]
    KAI_ASM_INST(0x4fbbe0eb)  // sdot v11.4s, v7.16b, v27.4b[1]
    KAI_ASM_INST(0x4f9be8ed)  // sdot v13.4s, v7.16b, v27.4b[2]
    KAI_ASM_INST(0x4fbbe8ef)  // sdot v15.4s, v7.16b, v27.4b[3]
    KAI_ASM_INST(0x4f9ae0f7)  // sdot v23.4s, v7.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae0fd)  // sdot v29.4s, v7.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9ae8e0)  // sdot v0.4s, v7.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbae8e4)  // sdot v4.4s, v7.16b, v26.4b[3]
    KAI_ASM_INST(0x4f94e0f0)  // sdot v16.4s, v7.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e0f5)  // sdot v21.4s, v7.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94e8ea)  // sdot v10.4s, v7.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4e8e3)  // sdot v3.4s, v7.16b, v20.4b[3]
    KAI_ASM_INST(0x4f93e3c2)  // sdot v2.4s, v30.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e3d1)  // sdot v17.4s, v30.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93ebcc)  // sdot v12.4s, v30.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3ebc9)  // sdot v9.4s, v30.16b, v19.4b[3]
    KAI_ASM_INST(0x4f96e3ce)  // sdot v14.4s, v30.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e3cb)  // sdot v11.4s, v30.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96ebcd)  // sdot v13.4s, v30.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6ebcf)  // sdot v15.4s, v30.16b, v22.4b[3]
    KAI_ASM_INST(0x4f88e3d7)  // sdot v23.4s, v30.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e3dd)  // sdot v29.4s, v30.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88ebc0)  // sdot v0.4s, v30.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8ebc4)  // sdot v4.4s, v30.16b, v8.4b[3]
    KAI_ASM_INST(0x4f9ce3d0)  // sdot v16.4s, v30.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce3d5)  // sdot v21.4s, v30.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9cebca)  // sdot v10.4s, v30.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbcebc3)  // sdot v3.4s, v30.16b, v28.4b[3]
    bgt label_4
    ldr d7, [x11, #0x0]
    ldr q31, [SP, #0x0]
    scvtf v2.4s, v2.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    scvtf v12.4s, v12.4s, #0x4
    scvtf v9.4s, v9.4s, #0x4
    add x11, x11, #0x8
    shll v7.4s, v7.4h, #0x10
    fmla v31.4s, v2.4s, v7.4s
    str q31, [SP, #0x0]
    ldr q2, [SP, #0x10]
    fmla v2.4s, v17.4s, v7.4s
    str q2, [SP, #0x10]
    ldr q2, [SP, #0x20]
    fmla v2.4s, v12.4s, v7.4s
    str q2, [SP, #0x20]
    ldr q2, [SP, #0x30]
    fmla v2.4s, v9.4s, v7.4s
    str q2, [SP, #0x30]
    ldr q28, [SP, #0x40]
    scvtf v14.4s, v14.4s, #0x4
    scvtf v11.4s, v11.4s, #0x4
    scvtf v13.4s, v13.4s, #0x4
    scvtf v15.4s, v15.4s, #0x4
    fmla v28.4s, v14.4s, v7.4s
    str q28, [SP, #0x40]
    ldr q1, [SP, #0x50]
    fmla v1.4s, v11.4s, v7.4s
    str q1, [SP, #0x50]
    ldr q11, [SP, #0x60]
    fmla v11.4s, v13.4s, v7.4s
    str q11, [SP, #0x60]
    ldr q14, [SP, #0x70]
    fmla v14.4s, v15.4s, v7.4s
    str q14, [SP, #0x70]
    ldr q19, [SP, #0x80]
    scvtf v23.4s, v23.4s, #0x4
    scvtf v29.4s, v29.4s, #0x4
    scvtf v0.4s, v0.4s, #0x4
    scvtf v4.4s, v4.4s, #0x4
    fmla v19.4s, v23.4s, v7.4s
    str q19, [SP, #0x80]
    ldr q15, [SP, #0x90]
    fmla v15.4s, v29.4s, v7.4s
    str q15, [SP, #0x90]
    ldr q25, [SP, #0xa0]
    fmla v25.4s, v0.4s, v7.4s
    str q25, [SP, #0xa0]
    ldr q12, [SP, #0xb0]
    fmla v12.4s, v4.4s, v7.4s
    str q12, [SP, #0xb0]
    ldr q2, [SP, #0xc0]
    scvtf v16.4s, v16.4s, #0x4
    scvtf v21.4s, v21.4s, #0x4
    scvtf v10.4s, v10.4s, #0x4
    scvtf v3.4s, v3.4s, #0x4
    fmla v2.4s, v16.4s, v7.4s
    str q2, [SP, #0xc0]
    ldr q16, [SP, #0xd0]
    fmla v16.4s, v21.4s, v7.4s
    str q16, [SP, #0xd0]
    ldr q16, [SP, #0xe0]
    fmla v16.4s, v10.4s, v7.4s
    str q16, [SP, #0xe0]
    ldr q16, [SP, #0xf0]
    fmla v16.4s, v3.4s, v7.4s
    str q16, [SP, #0xf0]
    subs x24, x24, #0x1
    bgt label_3
    ld1 { v11.4s }, [x27]
    ld1 { v10.4s }, [x23]
    add x27, x27, #0x10
    add x23, x23, #0x10
    ld1 { v9.4s }, [x22]
    ld1 { v8.4s }, [x21]
    add x22, x22, #0x10
    add x21, x21, #0x10
    ldr q31, [SP, #0x0]
    ldr q30, [SP, #0x10]
    add x20, x12, #0x4
    cmp x10, #0x4
    ldr q29, [SP, #0x20]
    ldr q28, [SP, #0x30]
    scvtf v11.4s, v11.4s
    scvtf v10.4s, v10.4s
    ldr q27, [SP, #0x40]
    ldr q26, [SP, #0x50]
    scvtf v9.4s, v9.4s
    scvtf v8.4s, v8.4s
    ldr q25, [SP, #0x60]
    ldr q24, [SP, #0x70]
    ldr q23, [SP, #0x80]
    ldr q22, [SP, #0x90]
    ldr q21, [SP, #0xa0]
    ldr q20, [SP, #0xb0]
    ldr q19, [SP, #0xc0]
    ldr q18, [SP, #0xd0]
    ldr q17, [SP, #0xe0]
    ldr q16, [SP, #0xf0]
    ldr q7, [x11, #0x0]
    ldr q6, [x27, #0x0]
    ldr q5, [x23, #0x0]
    ldr q4, [x22, #0x0]
    ldr q3, [x21, #0x0]
    ldr q2, [x11, #0x10]
    add x11, x11, #0x20
    ld1r { v1.4s }, [x12]
    ld1r { v0.4s }, [x20]
    fmla v31.4s, v7.4s, v11.s[0]
    fmla v30.4s, v7.4s, v11.s[1]
    fmla v29.4s, v7.4s, v11.s[2]
    fmla v28.4s, v7.4s, v11.s[3]
    fmla v27.4s, v7.4s, v10.s[0]
    fmla v26.4s, v7.4s, v10.s[1]
    fmla v25.4s, v7.4s, v10.s[2]
    fmla v24.4s, v7.4s, v10.s[3]
    fmla v23.4s, v7.4s, v9.s[0]
    fmla v22.4s, v7.4s, v9.s[1]
    fmul v31.4s, v31.4s, v6.s[0]
    fmla v21.4s, v7.4s, v9.s[2]
    fmla v20.4s, v7.4s, v9.s[3]
    fmul v30.4s, v30.4s, v6.s[1]
    fmla v19.4s, v7.4s, v8.s[0]
    fmla v18.4s, v7.4s, v8.s[1]
    fmul v29.4s, v29.4s, v6.s[2]
    fmla v17.4s, v7.4s, v8.s[2]
    fmla v16.4s, v7.4s, v8.s[3]
    fmul v28.4s, v28.4s, v6.s[3]
    fmul v27.4s, v27.4s, v5.s[0]
    fmul v26.4s, v26.4s, v5.s[1]
    fmul v25.4s, v25.4s, v5.s[2]
    fmul v24.4s, v24.4s, v5.s[3]
    fmul v23.4s, v23.4s, v4.s[0]
    fmul v22.4s, v22.4s, v4.s[1]
    fmul v21.4s, v21.4s, v4.s[2]
    fmul v20.4s, v20.4s, v4.s[3]
    fmul v19.4s, v19.4s, v3.s[0]
    fmul v18.4s, v18.4s, v3.s[1]
    fmul v17.4s, v17.4s, v3.s[2]
    fmul v16.4s, v16.4s, v3.s[3]
    fadd v31.4s, v31.4s, v2.4s
    fadd v30.4s, v30.4s, v2.4s
    fadd v29.4s, v29.4s, v2.4s
    fadd v28.4s, v28.4s, v2.4s
    fadd v27.4s, v27.4s, v2.4s
    fadd v26.4s, v26.4s, v2.4s
    fadd v25.4s, v25.4s, v2.4s
    fadd v24.4s, v24.4s, v2.4s
    fadd v23.4s, v23.4s, v2.4s
    fadd v22.4s, v22.4s, v2.4s
    fadd v21.4s, v21.4s, v2.4s
    fadd v20.4s, v20.4s, v2.4s
    fadd v19.4s, v19.4s, v2.4s
    fadd v18.4s, v18.4s, v2.4s
    fadd v17.4s, v17.4s, v2.4s
    fadd v16.4s, v16.4s, v2.4s
    fmax v31.4s, v31.4s, v1.4s
    fmax v30.4s, v30.4s, v1.4s
    fmax v29.4s, v29.4s, v1.4s
    fmax v28.4s, v28.4s, v1.4s
    fmax v27.4s, v27.4s, v1.4s
    fmax v26.4s, v26.4s, v1.4s
    fmax v25.4s, v25.4s, v1.4s
    fmax v24.4s, v24.4s, v1.4s
    fmax v23.4s, v23.4s, v1.4s
    fmax v22.4s, v22.4s, v1.4s
    fmax v21.4s, v21.4s, v1.4s
    fmax v20.4s, v20.4s, v1.4s
    fmax v19.4s, v19.4s, v1.4s
    fmax v18.4s, v18.4s, v1.4s
    fmax v17.4s, v17.4s, v1.4s
    fmax v16.4s, v16.4s, v1.4s
    fmin v31.4s, v31.4s, v0.4s
    fmin v30.4s, v30.4s, v0.4s
    fmin v29.4s, v29.4s, v0.4s
    fmin v28.4s, v28.4s, v0.4s
    fmin v27.4s, v27.4s, v0.4s
    fmin v26.4s, v26.4s, v0.4s
    fmin v25.4s, v25.4s, v0.4s
    fmin v24.4s, v24.4s, v0.4s
    fmin v23.4s, v23.4s, v0.4s
    fmin v22.4s, v22.4s, v0.4s
    fmin v21.4s, v21.4s, v0.4s
    fmin v20.4s, v20.4s, v0.4s
    fmin v19.4s, v19.4s, v0.4s
    fmin v18.4s, v18.4s, v0.4s
    fmin v17.4s, v17.4s, v0.4s
    fmin v16.4s, v16.4s, v0.4s
    blt label_9
    mov x20, x14
    str q31, [x20, #0x0]
    add x20, x20, x13
    str q30, [x20, #0x0]
    add x20, x20, x13
    str q29, [x20, #0x0]
    add x20, x20, x13
    str q28, [x20, #0x0]
    add x20, x20, x13
    str q27, [x20, #0x0]
    add x20, x20, x13
    str q26, [x20, #0x0]
    add x20, x20, x13
    str q25, [x20, #0x0]
    add x20, x20, x13
    str q24, [x20, #0x0]
    add x20, x20, x13
    str q23, [x20, #0x0]
    add x20, x20, x13
    str q22, [x20, #0x0]
    add x20, x20, x13
    str q21, [x20, #0x0]
    add x20, x20, x13
    str q20, [x20, #0x0]
    add x20, x20, x13
    str q19, [x20, #0x0]
    add x20, x20, x13
    str q18, [x20, #0x0]
    add x20, x20, x13
    str q17, [x20, #0x0]
    add x20, x20, x13
    str q16, [x20, #0x0]
    b label_14
KAI_ASM_LABEL(label_9)  // Partial output
    mov x28, x14
    add x26, x28, x13, LSL #2
    add x25, x26, x13, LSL #1
    add x24, x26, x13
    add x23, x25, x13
    add x22, x28, x13, LSL #1
    add x21, x28, x13
    add x20, x22, x13
    add x27, x23, x13
    tbz x10, #1, label_10
    st1 { v24.d }[0], [x23], #0x8
    st1 { v25.d }[0], [x25], #0x8
    st1 { v26.d }[0], [x24], #0x8
    st1 { v27.d }[0], [x26], #0x8
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x22], #0x8
    st1 { v30.d }[0], [x21], #0x8
    st1 { v31.d }[0], [x28], #0x8
    tbz x10, #0, label_11
    st1 { v24.s }[2], [x23]
    st1 { v25.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v27.s }[2], [x26]
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x22]
    st1 { v30.s }[2], [x21]
    st1 { v31.s }[2], [x28]
    b label_11
KAI_ASM_LABEL(label_10)  // Output block 0: partial_1_0
    st1 { v24.s }[0], [x23]
    st1 { v25.s }[0], [x25]
    st1 { v26.s }[0], [x24]
    st1 { v27.s }[0], [x26]
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x22]
    st1 { v30.s }[0], [x21]
    st1 { v31.s }[0], [x28]
KAI_ASM_LABEL(label_11)  // Output block 0: Done
    add x26, x27, x13, LSL #2
    add x25, x27, x13, LSL #1
    add x24, x26, x13, LSL #1
    add x23, x27, x13
    add x22, x25, x13
    add x21, x26, x13
    add x20, x24, x13
    tbz x10, #1, label_12
    st1 { v16.d }[0], [x20], #0x8
    st1 { v17.d }[0], [x24], #0x8
    st1 { v18.d }[0], [x21], #0x8
    st1 { v19.d }[0], [x26], #0x8
    st1 { v20.d }[0], [x22], #0x8
    st1 { v21.d }[0], [x25], #0x8
    st1 { v22.d }[0], [x23], #0x8
    st1 { v23.d }[0], [x27], #0x8
    tbz x10, #0, label_13
    st1 { v16.s }[2], [x20]
    st1 { v17.s }[2], [x24]
    st1 { v18.s }[2], [x21]
    st1 { v19.s }[2], [x26]
    st1 { v20.s }[2], [x22]
    st1 { v21.s }[2], [x25]
    st1 { v22.s }[2], [x23]
    st1 { v23.s }[2], [x27]
    b label_13
KAI_ASM_LABEL(label_12)  // Output block 1: partial_1_0
    st1 { v16.s }[0], [x20]
    st1 { v17.s }[0], [x24]
    st1 { v18.s }[0], [x21]
    st1 { v19.s }[0], [x26]
    st1 { v20.s }[0], [x22]
    st1 { v21.s }[0], [x25]
    st1 { v22.s }[0], [x23]
    st1 { v23.s }[0], [x27]
KAI_ASM_LABEL(label_13)  // Output block 1: Done
KAI_ASM_LABEL(label_14)  // Output stage exit
    subs x10, x10, #0x4
    add x14, x14, #0x10
    bgt label_2
    mov x20, #0x4
    sub x15, x15, #0x10
    cmp x15, #0x10
    mov x14, x9
    madd x8, x20, x5, x8
    bge label_1
KAI_ASM_LABEL(label_15)  // Row loop skip
    cbz x15, label_25
KAI_ASM_LABEL(label_16)  // Row tail: Row loop
    mov x26, x17
    mov x25, x16
    add x24, x14, x13, LSL #2
KAI_ASM_LABEL(label_17)  // Row tail: Column loop
    movi v16.4s, #0x0
    mov x27, x8
    mov x21, x7
    str q16, [SP, #0x0]
    str q16, [SP, #0x10]
    str q16, [SP, #0x20]
    str q16, [SP, #0x30]
KAI_ASM_LABEL(label_18)  // Row tail: Block loop
    movi v2.4s, #0x0
    movi v17.4s, #0x0
    mov x20, x6
    movi v12.4s, #0x0
    movi v9.4s, #0x0
KAI_ASM_LABEL(label_19)  // Row tail: Sub block loop
    ldr q0, [x26, #0x0]
    ldr q31, [x27, #0x0]
    movi v30.16b, #0xf0
    subs x20, x20, #0x1
    ldr q29, [x26, #0x10]
    ldr q28, [x27, #0x10]
    ldr q27, [x26, #0x20]
    ldr q26, [x27, #0x20]
    ldr q25, [x26, #0x30]
    ldr q24, [x27, #0x30]
    shl v23.16b, v0.16b, #0x4
    and v0.16b, v0.16b, v30.16b
    ldr q22, [x27, #0x40]
    ldr q21, [x27, #0x50]
    shl v20.16b, v29.16b, #0x4
    and v29.16b, v29.16b, v30.16b
    ldr q7, [x27, #0x60]
    ldr q18, [x27, #0x70]
    shl v19.16b, v27.16b, #0x4
    and v27.16b, v27.16b, v30.16b
    KAI_ASM_INST(0x4f9fe2e2)  // sdot v2.4s, v23.16b, v31.4b[0]
    KAI_ASM_INST(0x4fbfe2f1)  // sdot v17.4s, v23.16b, v31.4b[1]
    shl v16.16b, v25.16b, #0x4
    add x26, x26, #0x40
    KAI_ASM_INST(0x4f9feaec)  // sdot v12.4s, v23.16b, v31.4b[2]
    KAI_ASM_INST(0x4fbfeae9)  // sdot v9.4s, v23.16b, v31.4b[3]
    and v25.16b, v25.16b, v30.16b
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f9ce282)  // sdot v2.4s, v20.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce291)  // sdot v17.4s, v20.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9cea8c)  // sdot v12.4s, v20.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbcea89)  // sdot v9.4s, v20.16b, v28.4b[3]
    KAI_ASM_INST(0x4f9ae262)  // sdot v2.4s, v19.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae271)  // sdot v17.4s, v19.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aea6c)  // sdot v12.4s, v19.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaea69)  // sdot v9.4s, v19.16b, v26.4b[3]
    KAI_ASM_INST(0x4f98e202)  // sdot v2.4s, v16.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e211)  // sdot v17.4s, v16.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ea0c)  // sdot v12.4s, v16.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ea09)  // sdot v9.4s, v16.16b, v24.4b[3]
    KAI_ASM_INST(0x4f96e002)  // sdot v2.4s, v0.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e011)  // sdot v17.4s, v0.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96e80c)  // sdot v12.4s, v0.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6e809)  // sdot v9.4s, v0.16b, v22.4b[3]
    KAI_ASM_INST(0x4f95e3a2)  // sdot v2.4s, v29.16b, v21.4b[0]
    KAI_ASM_INST(0x4fb5e3b1)  // sdot v17.4s, v29.16b, v21.4b[1]
    KAI_ASM_INST(0x4f95ebac)  // sdot v12.4s, v29.16b, v21.4b[2]
    KAI_ASM_INST(0x4fb5eba9)  // sdot v9.4s, v29.16b, v21.4b[3]
    KAI_ASM_INST(0x4f87e362)  // sdot v2.4s, v27.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e371)  // sdot v17.4s, v27.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87eb6c)  // sdot v12.4s, v27.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7eb69)  // sdot v9.4s, v27.16b, v7.4b[3]
    KAI_ASM_INST(0x4f92e322)  // sdot v2.4s, v25.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e331)  // sdot v17.4s, v25.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92eb2c)  // sdot v12.4s, v25.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eb29)  // sdot v9.4s, v25.16b, v18.4b[3]
    bgt label_19
    ldr d7, [x26, #0x0]
    ldr q16, [SP, #0x0]
    scvtf v2.4s, v2.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    scvtf v12.4s, v12.4s, #0x4
    scvtf v9.4s, v9.4s, #0x4
    add x26, x26, #0x8
    shll v7.4s, v7.4h, #0x10
    fmla v16.4s, v2.4s, v7.4s
    str q16, [SP, #0x0]
    ldr q16, [SP, #0x10]
    fmla v16.4s, v17.4s, v7.4s
    str q16, [SP, #0x10]
    ldr q16, [SP, #0x20]
    fmla v16.4s, v12.4s, v7.4s
    str q16, [SP, #0x20]
    ldr q16, [SP, #0x30]
    fmla v16.4s, v9.4s, v7.4s
    str q16, [SP, #0x30]
    subs x21, x21, #0x1
    bgt label_18
    ld1 { v21.4s }, [x27]
    ldr q31, [SP, #0x0]
    add x27, x27, #0x10
    add x20, x12, #0x4
    ldr q30, [SP, #0x10]
    ldr q29, [SP, #0x20]
    cmp x25, #0x4
    ldr q28, [SP, #0x30]
    ldr q20, [x26, #0x0]
    ldr q19, [x27, #0x0]
    ldr q18, [x26, #0x10]
    scvtf v21.4s, v21.4s
    add x26, x26, #0x20
    ld1r { v17.4s }, [x12]
    ld1r { v16.4s }, [x20]
    fmla v31.4s, v20.4s, v21.s[0]
    fmla v30.4s, v20.4s, v21.s[1]
    fmla v29.4s, v20.4s, v21.s[2]
    fmla v28.4s, v20.4s, v21.s[3]
    fmul v31.4s, v31.4s, v19.s[0]
    fmul v30.4s, v30.4s, v19.s[1]
    fadd v31.4s, v31.4s, v18.4s
    fmul v29.4s, v29.4s, v19.s[2]
    fmul v28.4s, v28.4s, v19.s[3]
    fadd v30.4s, v30.4s, v18.4s
    fmax v31.4s, v31.4s, v17.4s
    fadd v29.4s, v29.4s, v18.4s
    fadd v28.4s, v28.4s, v18.4s
    fmax v30.4s, v30.4s, v17.4s
    fmin v31.4s, v31.4s, v16.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v28.4s, v28.4s, v17.4s
    fmin v30.4s, v30.4s, v16.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v28.4s, v28.4s, v16.4s
    blt label_21
    mov x20, x14
    cmp x15, #0x1
    str q31, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    cmp x15, #0x2
    str q30, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    cmp x15, #0x3
    str q29, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    str q28, [x20, #0x0]
    b label_24
KAI_ASM_LABEL(label_21)  // Row tail: Partial output
    mov x23, x14
    cmp x15, #0x1
    add x22, x23, x13
    csel x22, x22, x23, GT
    cmp x15, #0x2
    add x21, x23, x13, LSL #1
    csel x21, x21, x22, GT
    cmp x15, #0x3
    add x20, x21, x13
    csel x20, x20, x21, GT
    tbz x25, #1, label_22
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x21], #0x8
    st1 { v30.d }[0], [x22], #0x8
    st1 { v31.d }[0], [x23], #0x8
    tbz x25, #0, label_23
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x21]
    st1 { v30.s }[2], [x22]
    st1 { v31.s }[2], [x23]
    b label_23
KAI_ASM_LABEL(label_22)  // Row tail: Output block 0: partial_1_0
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x21]
    st1 { v30.s }[0], [x22]
    st1 { v31.s }[0], [x23]
KAI_ASM_LABEL(label_23)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_24)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x14, x14, #0x10
    bgt label_17
    subs x15, x15, #0x4
    add x8, x8, x5
    mov x14, x24
    bgt label_16
KAI_ASM_LABEL(label_25)  // Row tail: Row loop skip
    add SP, SP, #0x100
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp d8, d9, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod)

// Optimized kernel for bl = 32

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_opt32_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_opt32_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_opt32_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_opt32_neon_dotprod)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    mov x6, #0x80
    mov x21, #0x20
    ldr x20, [x0, #0x28]
    ldr x7, [x0, #0x38]
    ldr x8, [x0, #0x8]
    ldr x17, [x0, #0x10]
    ldr x16, [x0, #0x30]
    ldr x15, [x0, #0x0]
    mov x14, x20
    ldr x13, [x0, #0x20]
    madd x6, x7, x6, x21
    ldr x12, [x0, #0x18]
    cmp x14, #0x10
    blt label_opt_14
KAI_ASM_LABEL(label_opt_1)  // Row loop
    mov x11, x17
    mov x10, x16
    add x9, x15, x13, LSL #4
KAI_ASM_LABEL(label_opt_2)  // Column loop
    mov x27, x8
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x20, x7
    movi v29.16b, #0x0
    movi v28.16b, #0x0
    movi v27.16b, #0x0
    movi v26.16b, #0x0
    add x23, x27, x6
    add x22, x23, x6
    movi v25.16b, #0x0
    movi v24.16b, #0x0
    add x21, x22, x6
    movi v23.16b, #0x0
    movi v22.16b, #0x0
    movi v21.16b, #0x0
    movi v20.16b, #0x0
    movi v19.16b, #0x0
    movi v18.16b, #0x0
    movi v17.16b, #0x0
    movi v16.16b, #0x0
KAI_ASM_LABEL(label_opt_3)  // Block loop
    ldr q3, [x11, #0x0]
    ldr q2, [x27, #0x0]
    movi v5.4s, #0x0
    movi v4.4s, #0x0
    ldr q0, [x11, #0x10]
    ldr q1, [x27, #0x10]
    movi v6.4s, #0x0
    movi v11.4s, #0x0
    ldr q15, [x11, #0x20]
    ldr q14, [x27, #0x20]
    movi v7.16b, #0xf0
    ldr q13, [x11, #0x30]
    ldr q8, [x27, #0x30]
    shl v12.16b, v3.16b, #0x4
    add x11, x11, #0x40
    ldr q9, [x27, #0x40]
    ldr q10, [x27, #0x50]
    and v3.16b, v3.16b, v7.16b
    KAI_ASM_INST(0x4f82e185)  // sdot v5.4s, v12.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e184)  // sdot v4.4s, v12.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e986)  // sdot v6.4s, v12.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e98b)  // sdot v11.4s, v12.16b, v2.4b[3]
    shl v2.16b, v0.16b, #0x4
    and v0.16b, v0.16b, v7.16b
    KAI_ASM_INST(0x4f81e045)  // sdot v5.4s, v2.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e044)  // sdot v4.4s, v2.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e846)  // sdot v6.4s, v2.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e84b)  // sdot v11.4s, v2.16b, v1.4b[3]
    shl v1.16b, v15.16b, #0x4
    and v15.16b, v15.16b, v7.16b
    KAI_ASM_INST(0x4f8ee025)  // sdot v5.4s, v1.16b, v14.4b[0]
    KAI_ASM_INST(0x4faee024)  // sdot v4.4s, v1.16b, v14.4b[1]
    KAI_ASM_INST(0x4f8ee826)  // sdot v6.4s, v1.16b, v14.4b[2]
    KAI_ASM_INST(0x4faee82b)  // sdot v11.4s, v1.16b, v14.4b[3]
    shl v14.16b, v13.16b, #0x4
    and v13.16b, v13.16b, v7.16b
    ldr q7, [x27, #0x60]
    KAI_ASM_INST(0x4f88e1c5)  // sdot v5.4s, v14.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e1c4)  // sdot v4.4s, v14.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e9c6)  // sdot v6.4s, v14.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e9cb)  // sdot v11.4s, v14.16b, v8.4b[3]
    ldr q8, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f89e065)  // sdot v5.4s, v3.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e064)  // sdot v4.4s, v3.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e866)  // sdot v6.4s, v3.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e86b)  // sdot v11.4s, v3.16b, v9.4b[3]
    ldr d9, [x11, #0x0]
    add x11, x11, #0x8
    KAI_ASM_INST(0x4f8ae005)  // sdot v5.4s, v0.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae004)  // sdot v4.4s, v0.16b, v10.4b[1]
    shll v9.4s, v9.4h, #0x10
    KAI_ASM_INST(0x4f8ae806)  // sdot v6.4s, v0.16b, v10.4b[2]
    KAI_ASM_INST(0x4faae80b)  // sdot v11.4s, v0.16b, v10.4b[3]
    KAI_ASM_INST(0x4f87e1e5)  // sdot v5.4s, v15.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e1e4)  // sdot v4.4s, v15.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e9e6)  // sdot v6.4s, v15.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e9eb)  // sdot v11.4s, v15.16b, v7.4b[3]
    KAI_ASM_INST(0x4f88e1a5)  // sdot v5.4s, v13.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e1a4)  // sdot v4.4s, v13.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e9a6)  // sdot v6.4s, v13.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e9ab)  // sdot v11.4s, v13.16b, v8.4b[3]
    scvtf v5.4s, v5.4s, #0x4
    scvtf v4.4s, v4.4s, #0x4
    fmla v31.4s, v5.4s, v9.4s
    scvtf v6.4s, v6.4s, #0x4
    scvtf v11.4s, v11.4s, #0x4
    fmla v30.4s, v4.4s, v9.4s
    fmla v29.4s, v6.4s, v9.4s
    fmla v28.4s, v11.4s, v9.4s
    ldr q8, [x23, #0x0]
    ldr q5, [x23, #0x10]
    movi v11.4s, #0x0
    movi v10.4s, #0x0
    ldr q7, [x23, #0x20]
    movi v4.4s, #0x0
    movi v6.4s, #0x0
    KAI_ASM_INST(0x4f88e18b)  // sdot v11.4s, v12.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e18a)  // sdot v10.4s, v12.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e984)  // sdot v4.4s, v12.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e986)  // sdot v6.4s, v12.16b, v8.4b[3]
    ldr q8, [x23, #0x30]
    KAI_ASM_INST(0x4f85e04b)  // sdot v11.4s, v2.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e04a)  // sdot v10.4s, v2.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e844)  // sdot v4.4s, v2.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e846)  // sdot v6.4s, v2.16b, v5.4b[3]
    ldr q5, [x23, #0x40]
    KAI_ASM_INST(0x4f87e02b)  // sdot v11.4s, v1.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e02a)  // sdot v10.4s, v1.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e824)  // sdot v4.4s, v1.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e826)  // sdot v6.4s, v1.16b, v7.4b[3]
    ldr q7, [x23, #0x50]
    KAI_ASM_INST(0x4f88e1cb)  // sdot v11.4s, v14.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e1ca)  // sdot v10.4s, v14.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e9c4)  // sdot v4.4s, v14.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e9c6)  // sdot v6.4s, v14.16b, v8.4b[3]
    ldr q8, [x23, #0x60]
    KAI_ASM_INST(0x4f85e06b)  // sdot v11.4s, v3.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e06a)  // sdot v10.4s, v3.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e864)  // sdot v4.4s, v3.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e866)  // sdot v6.4s, v3.16b, v5.4b[3]
    ldr q5, [x23, #0x70]
    add x23, x23, #0x80
    KAI_ASM_INST(0x4f87e00b)  // sdot v11.4s, v0.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e00a)  // sdot v10.4s, v0.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e804)  // sdot v4.4s, v0.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e806)  // sdot v6.4s, v0.16b, v7.4b[3]
    KAI_ASM_INST(0x4f88e1eb)  // sdot v11.4s, v15.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e1ea)  // sdot v10.4s, v15.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e9e4)  // sdot v4.4s, v15.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e9e6)  // sdot v6.4s, v15.16b, v8.4b[3]
    KAI_ASM_INST(0x4f85e1ab)  // sdot v11.4s, v13.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e1aa)  // sdot v10.4s, v13.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e9a4)  // sdot v4.4s, v13.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e9a6)  // sdot v6.4s, v13.16b, v5.4b[3]
    scvtf v11.4s, v11.4s, #0x4
    scvtf v10.4s, v10.4s, #0x4
    scvtf v4.4s, v4.4s, #0x4
    fmla v27.4s, v11.4s, v9.4s
    scvtf v6.4s, v6.4s, #0x4
    fmla v26.4s, v10.4s, v9.4s
    fmla v25.4s, v4.4s, v9.4s
    fmla v24.4s, v6.4s, v9.4s
    ldr q5, [x22, #0x0]
    ldr q4, [x22, #0x10]
    movi v11.4s, #0x0
    movi v10.4s, #0x0
    ldr q6, [x22, #0x20]
    movi v8.4s, #0x0
    movi v7.4s, #0x0
    KAI_ASM_INST(0x4f85e18b)  // sdot v11.4s, v12.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e18a)  // sdot v10.4s, v12.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e988)  // sdot v8.4s, v12.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e987)  // sdot v7.4s, v12.16b, v5.4b[3]
    ldr q5, [x22, #0x30]
    KAI_ASM_INST(0x4f84e04b)  // sdot v11.4s, v2.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e04a)  // sdot v10.4s, v2.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e848)  // sdot v8.4s, v2.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e847)  // sdot v7.4s, v2.16b, v4.4b[3]
    ldr q4, [x22, #0x40]
    KAI_ASM_INST(0x4f86e02b)  // sdot v11.4s, v1.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e02a)  // sdot v10.4s, v1.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e828)  // sdot v8.4s, v1.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e827)  // sdot v7.4s, v1.16b, v6.4b[3]
    ldr q6, [x22, #0x50]
    KAI_ASM_INST(0x4f85e1cb)  // sdot v11.4s, v14.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e1ca)  // sdot v10.4s, v14.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e9c8)  // sdot v8.4s, v14.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e9c7)  // sdot v7.4s, v14.16b, v5.4b[3]
    ldr q5, [x22, #0x60]
    KAI_ASM_INST(0x4f84e06b)  // sdot v11.4s, v3.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e06a)  // sdot v10.4s, v3.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e868)  // sdot v8.4s, v3.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e867)  // sdot v7.4s, v3.16b, v4.4b[3]
    ldr q4, [x22, #0x70]
    add x22, x22, #0x80
    KAI_ASM_INST(0x4f86e00b)  // sdot v11.4s, v0.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e00a)  // sdot v10.4s, v0.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e808)  // sdot v8.4s, v0.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e807)  // sdot v7.4s, v0.16b, v6.4b[3]
    KAI_ASM_INST(0x4f85e1eb)  // sdot v11.4s, v15.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e1ea)  // sdot v10.4s, v15.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e9e8)  // sdot v8.4s, v15.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e9e7)  // sdot v7.4s, v15.16b, v5.4b[3]
    KAI_ASM_INST(0x4f84e1ab)  // sdot v11.4s, v13.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e1aa)  // sdot v10.4s, v13.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e9a8)  // sdot v8.4s, v13.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e9a7)  // sdot v7.4s, v13.16b, v4.4b[3]
    scvtf v11.4s, v11.4s, #0x4
    scvtf v10.4s, v10.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    fmla v23.4s, v11.4s, v9.4s
    scvtf v7.4s, v7.4s, #0x4
    fmla v22.4s, v10.4s, v9.4s
    fmla v21.4s, v8.4s, v9.4s
    fmla v20.4s, v7.4s, v9.4s
    ldr q5, [x21, #0x0]
    ldr q4, [x21, #0x10]
    movi v8.4s, #0x0
    movi v11.4s, #0x0
    ldr q7, [x21, #0x20]
    movi v10.4s, #0x0
    movi v6.4s, #0x0
    KAI_ASM_INST(0x4f85e188)  // sdot v8.4s, v12.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e18b)  // sdot v11.4s, v12.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e98a)  // sdot v10.4s, v12.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e986)  // sdot v6.4s, v12.16b, v5.4b[3]
    ldr q5, [x21, #0x30]
    ldr q12, [x21, #0x40]
    KAI_ASM_INST(0x4f84e048)  // sdot v8.4s, v2.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e04b)  // sdot v11.4s, v2.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e84a)  // sdot v10.4s, v2.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e846)  // sdot v6.4s, v2.16b, v4.4b[3]
    ldr q4, [x21, #0x50]
    ldr q2, [x21, #0x60]
    KAI_ASM_INST(0x4f87e028)  // sdot v8.4s, v1.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e02b)  // sdot v11.4s, v1.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e82a)  // sdot v10.4s, v1.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e826)  // sdot v6.4s, v1.16b, v7.4b[3]
    ldr q1, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4f85e1c8)  // sdot v8.4s, v14.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e1cb)  // sdot v11.4s, v14.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e9ca)  // sdot v10.4s, v14.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e9c6)  // sdot v6.4s, v14.16b, v5.4b[3]
    KAI_ASM_INST(0x4f8ce068)  // sdot v8.4s, v3.16b, v12.4b[0]
    KAI_ASM_INST(0x4face06b)  // sdot v11.4s, v3.16b, v12.4b[1]
    KAI_ASM_INST(0x4f8ce86a)  // sdot v10.4s, v3.16b, v12.4b[2]
    KAI_ASM_INST(0x4face866)  // sdot v6.4s, v3.16b, v12.4b[3]
    KAI_ASM_INST(0x4f84e008)  // sdot v8.4s, v0.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e00b)  // sdot v11.4s, v0.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e80a)  // sdot v10.4s, v0.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e806)  // sdot v6.4s, v0.16b, v4.4b[3]
    KAI_ASM_INST(0x4f82e1e8)  // sdot v8.4s, v15.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e1eb)  // sdot v11.4s, v15.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e9ea)  // sdot v10.4s, v15.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e9e6)  // sdot v6.4s, v15.16b, v2.4b[3]
    KAI_ASM_INST(0x4f81e1a8)  // sdot v8.4s, v13.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e1ab)  // sdot v11.4s, v13.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e9aa)  // sdot v10.4s, v13.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e9a6)  // sdot v6.4s, v13.16b, v1.4b[3]
    scvtf v8.4s, v8.4s, #0x4
    scvtf v11.4s, v11.4s, #0x4
    scvtf v10.4s, v10.4s, #0x4
    fmla v19.4s, v8.4s, v9.4s
    scvtf v6.4s, v6.4s, #0x4
    fmla v18.4s, v11.4s, v9.4s
    fmla v17.4s, v10.4s, v9.4s
    fmla v16.4s, v6.4s, v9.4s
    subs x20, x20, #0x1
    bgt label_opt_3
    ld1 { v11.4s }, [x27]
    ld1 { v10.4s }, [x23]
    add x27, x27, #0x10
    add x23, x23, #0x10
    ld1 { v9.4s }, [x22]
    ld1 { v8.4s }, [x21]
    add x22, x22, #0x10
    add x21, x21, #0x10
    ldr q7, [x11, #0x0]
    ldr q6, [x27, #0x0]
    add x20, x12, #0x4
    cmp x10, #0x4
    ldr q5, [x23, #0x0]
    ldr q4, [x22, #0x0]
    scvtf v11.4s, v11.4s
    scvtf v10.4s, v10.4s
    ldr q3, [x21, #0x0]
    ldr q2, [x11, #0x10]
    scvtf v9.4s, v9.4s
    scvtf v8.4s, v8.4s
    ld1r { v1.4s }, [x12]
    ld1r { v0.4s }, [x20]
    add x11, x11, #0x20
    fmla v31.4s, v7.4s, v11.s[0]
    fmla v30.4s, v7.4s, v11.s[1]
    fmla v29.4s, v7.4s, v11.s[2]
    fmla v28.4s, v7.4s, v11.s[3]
    fmla v27.4s, v7.4s, v10.s[0]
    fmla v26.4s, v7.4s, v10.s[1]
    fmla v25.4s, v7.4s, v10.s[2]
    fmla v24.4s, v7.4s, v10.s[3]
    fmla v23.4s, v7.4s, v9.s[0]
    fmul v31.4s, v31.4s, v6.s[0]
    fmla v22.4s, v7.4s, v9.s[1]
    fmla v21.4s, v7.4s, v9.s[2]
    fmul v30.4s, v30.4s, v6.s[1]
    fmla v20.4s, v7.4s, v9.s[3]
    fmla v19.4s, v7.4s, v8.s[0]
    fmul v29.4s, v29.4s, v6.s[2]
    fmla v18.4s, v7.4s, v8.s[1]
    fmla v17.4s, v7.4s, v8.s[2]
    fmul v28.4s, v28.4s, v6.s[3]
    fmla v16.4s, v7.4s, v8.s[3]
    fmul v27.4s, v27.4s, v5.s[0]
    fmul v26.4s, v26.4s, v5.s[1]
    fmul v25.4s, v25.4s, v5.s[2]
    fmul v24.4s, v24.4s, v5.s[3]
    fmul v23.4s, v23.4s, v4.s[0]
    fmul v22.4s, v22.4s, v4.s[1]
    fmul v21.4s, v21.4s, v4.s[2]
    fmul v20.4s, v20.4s, v4.s[3]
    fmul v19.4s, v19.4s, v3.s[0]
    fmul v18.4s, v18.4s, v3.s[1]
    fmul v17.4s, v17.4s, v3.s[2]
    fmul v16.4s, v16.4s, v3.s[3]
    fadd v31.4s, v31.4s, v2.4s
    fadd v30.4s, v30.4s, v2.4s
    fadd v29.4s, v29.4s, v2.4s
    fadd v28.4s, v28.4s, v2.4s
    fadd v27.4s, v27.4s, v2.4s
    fadd v26.4s, v26.4s, v2.4s
    fadd v25.4s, v25.4s, v2.4s
    fadd v24.4s, v24.4s, v2.4s
    fadd v23.4s, v23.4s, v2.4s
    fadd v22.4s, v22.4s, v2.4s
    fadd v21.4s, v21.4s, v2.4s
    fadd v20.4s, v20.4s, v2.4s
    fadd v19.4s, v19.4s, v2.4s
    fadd v18.4s, v18.4s, v2.4s
    fadd v17.4s, v17.4s, v2.4s
    fadd v16.4s, v16.4s, v2.4s
    fmax v31.4s, v31.4s, v1.4s
    fmax v30.4s, v30.4s, v1.4s
    fmax v29.4s, v29.4s, v1.4s
    fmax v28.4s, v28.4s, v1.4s
    fmax v27.4s, v27.4s, v1.4s
    fmax v26.4s, v26.4s, v1.4s
    fmax v25.4s, v25.4s, v1.4s
    fmax v24.4s, v24.4s, v1.4s
    fmax v23.4s, v23.4s, v1.4s
    fmax v22.4s, v22.4s, v1.4s
    fmax v21.4s, v21.4s, v1.4s
    fmax v20.4s, v20.4s, v1.4s
    fmax v19.4s, v19.4s, v1.4s
    fmax v18.4s, v18.4s, v1.4s
    fmax v17.4s, v17.4s, v1.4s
    fmax v16.4s, v16.4s, v1.4s
    fmin v31.4s, v31.4s, v0.4s
    fmin v30.4s, v30.4s, v0.4s
    fmin v29.4s, v29.4s, v0.4s
    fmin v28.4s, v28.4s, v0.4s
    fmin v27.4s, v27.4s, v0.4s
    fmin v26.4s, v26.4s, v0.4s
    fmin v25.4s, v25.4s, v0.4s
    fmin v24.4s, v24.4s, v0.4s
    fmin v23.4s, v23.4s, v0.4s
    fmin v22.4s, v22.4s, v0.4s
    fmin v21.4s, v21.4s, v0.4s
    fmin v20.4s, v20.4s, v0.4s
    fmin v19.4s, v19.4s, v0.4s
    fmin v18.4s, v18.4s, v0.4s
    fmin v17.4s, v17.4s, v0.4s
    fmin v16.4s, v16.4s, v0.4s
    blt label_opt_8
    mov x20, x15
    str q31, [x20, #0x0]
    add x20, x20, x13
    str q30, [x20, #0x0]
    add x20, x20, x13
    str q29, [x20, #0x0]
    add x20, x20, x13
    str q28, [x20, #0x0]
    add x20, x20, x13
    str q27, [x20, #0x0]
    add x20, x20, x13
    str q26, [x20, #0x0]
    add x20, x20, x13
    str q25, [x20, #0x0]
    add x20, x20, x13
    str q24, [x20, #0x0]
    add x20, x20, x13
    str q23, [x20, #0x0]
    add x20, x20, x13
    str q22, [x20, #0x0]
    add x20, x20, x13
    str q21, [x20, #0x0]
    add x20, x20, x13
    str q20, [x20, #0x0]
    add x20, x20, x13
    str q19, [x20, #0x0]
    add x20, x20, x13
    str q18, [x20, #0x0]
    add x20, x20, x13
    str q17, [x20, #0x0]
    add x20, x20, x13
    str q16, [x20, #0x0]
    b label_opt_13
KAI_ASM_LABEL(label_opt_8)  // Partial output
    mov x28, x15
    add x26, x28, x13, LSL #2
    add x25, x26, x13, LSL #1
    add x24, x26, x13
    add x23, x25, x13
    add x22, x28, x13, LSL #1
    add x21, x28, x13
    add x20, x22, x13
    add x27, x23, x13
    tbz x10, #1, label_opt_9
    st1 { v24.d }[0], [x23], #0x8
    st1 { v25.d }[0], [x25], #0x8
    st1 { v26.d }[0], [x24], #0x8
    st1 { v27.d }[0], [x26], #0x8
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x22], #0x8
    st1 { v30.d }[0], [x21], #0x8
    st1 { v31.d }[0], [x28], #0x8
    tbz x10, #0, label_opt_10
    st1 { v24.s }[2], [x23]
    st1 { v25.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v27.s }[2], [x26]
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x22]
    st1 { v30.s }[2], [x21]
    st1 { v31.s }[2], [x28]
    b label_opt_10
KAI_ASM_LABEL(label_opt_9)  // Output block 0: partial_1_0
    st1 { v24.s }[0], [x23]
    st1 { v25.s }[0], [x25]
    st1 { v26.s }[0], [x24]
    st1 { v27.s }[0], [x26]
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x22]
    st1 { v30.s }[0], [x21]
    st1 { v31.s }[0], [x28]
KAI_ASM_LABEL(label_opt_10)  // Output block 0: Done
    add x26, x27, x13, LSL #2
    add x25, x27, x13, LSL #1
    add x24, x26, x13, LSL #1
    add x23, x27, x13
    add x22, x25, x13
    add x21, x26, x13
    add x20, x24, x13
    tbz x10, #1, label_opt_11
    st1 { v16.d }[0], [x20], #0x8
    st1 { v17.d }[0], [x24], #0x8
    st1 { v18.d }[0], [x21], #0x8
    st1 { v19.d }[0], [x26], #0x8
    st1 { v20.d }[0], [x22], #0x8
    st1 { v21.d }[0], [x25], #0x8
    st1 { v22.d }[0], [x23], #0x8
    st1 { v23.d }[0], [x27], #0x8
    tbz x10, #0, label_opt_12
    st1 { v16.s }[2], [x20]
    st1 { v17.s }[2], [x24]
    st1 { v18.s }[2], [x21]
    st1 { v19.s }[2], [x26]
    st1 { v20.s }[2], [x22]
    st1 { v21.s }[2], [x25]
    st1 { v22.s }[2], [x23]
    st1 { v23.s }[2], [x27]
    b label_opt_12
KAI_ASM_LABEL(label_opt_11)  // Output block 1: partial_1_0
    st1 { v16.s }[0], [x20]
    st1 { v17.s }[0], [x24]
    st1 { v18.s }[0], [x21]
    st1 { v19.s }[0], [x26]
    st1 { v20.s }[0], [x22]
    st1 { v21.s }[0], [x25]
    st1 { v22.s }[0], [x23]
    st1 { v23.s }[0], [x27]
KAI_ASM_LABEL(label_opt_12)  // Output block 1: Done
KAI_ASM_LABEL(label_opt_13)  // Output stage exit
    subs x10, x10, #0x4
    add x15, x15, #0x10
    bgt label_opt_2
    mov x20, #0x4
    sub x14, x14, #0x10
    cmp x14, #0x10
    mov x15, x9
    madd x8, x20, x6, x8
    bge label_opt_1
KAI_ASM_LABEL(label_opt_14)  // Row loop skip
    cbz x14, label_opt_23
KAI_ASM_LABEL(label_opt_15)  // Row tail: Row loop
    mov x26, x17
    mov x25, x16
    add x24, x15, x13, LSL #2
KAI_ASM_LABEL(label_opt_16)  // Row tail: Column loop
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x27, x8
    mov x20, x7
    movi v29.16b, #0x0
    movi v28.16b, #0x0
KAI_ASM_LABEL(label_opt_17)  // Row tail: Block loop
    ldr q9, [x26, #0x0]
    ldr q8, [x27, #0x0]
    movi v7.4s, #0x0
    movi v6.4s, #0x0
    ldr q5, [x26, #0x10]
    ldr q4, [x27, #0x10]
    movi v3.4s, #0x0
    movi v2.4s, #0x0
    ldr q1, [x26, #0x20]
    ldr q0, [x27, #0x20]
    movi v27.16b, #0xf0
    ldr q26, [x26, #0x30]
    ldr q25, [x27, #0x30]
    shl v24.16b, v9.16b, #0x4
    add x26, x26, #0x40
    ldr q23, [x27, #0x40]
    ldr q22, [x27, #0x50]
    shl v21.16b, v5.16b, #0x4
    and v9.16b, v9.16b, v27.16b
    ldr q20, [x27, #0x60]
    ldr q19, [x27, #0x70]
    shl v18.16b, v1.16b, #0x4
    and v5.16b, v5.16b, v27.16b
    ldr d16, [x26, #0x0]
    KAI_ASM_INST(0x4f88e307)  // sdot v7.4s, v24.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e306)  // sdot v6.4s, v24.16b, v8.4b[1]
    shl v17.16b, v26.16b, #0x4
    KAI_ASM_INST(0x4f88eb03)  // sdot v3.4s, v24.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8eb02)  // sdot v2.4s, v24.16b, v8.4b[3]
    and v1.16b, v1.16b, v27.16b
    add x26, x26, #0x8
    and v26.16b, v26.16b, v27.16b
    add x27, x27, #0x80
    shll v16.4s, v16.4h, #0x10
    KAI_ASM_INST(0x4f84e2a7)  // sdot v7.4s, v21.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e2a6)  // sdot v6.4s, v21.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84eaa3)  // sdot v3.4s, v21.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4eaa2)  // sdot v2.4s, v21.16b, v4.4b[3]
    KAI_ASM_INST(0x4f80e247)  // sdot v7.4s, v18.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e246)  // sdot v6.4s, v18.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80ea43)  // sdot v3.4s, v18.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0ea42)  // sdot v2.4s, v18.16b, v0.4b[3]
    KAI_ASM_INST(0x4f99e227)  // sdot v7.4s, v17.16b, v25.4b[0]
    KAI_ASM_INST(0x4fb9e226)  // sdot v6.4s, v17.16b, v25.4b[1]
    KAI_ASM_INST(0x4f99ea23)  // sdot v3.4s, v17.16b, v25.4b[2]
    KAI_ASM_INST(0x4fb9ea22)  // sdot v2.4s, v17.16b, v25.4b[3]
    KAI_ASM_INST(0x4f97e127)  // sdot v7.4s, v9.16b, v23.4b[0]
    KAI_ASM_INST(0x4fb7e126)  // sdot v6.4s, v9.16b, v23.4b[1]
    KAI_ASM_INST(0x4f97e923)  // sdot v3.4s, v9.16b, v23.4b[2]
    KAI_ASM_INST(0x4fb7e922)  // sdot v2.4s, v9.16b, v23.4b[3]
    KAI_ASM_INST(0x4f96e0a7)  // sdot v7.4s, v5.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e0a6)  // sdot v6.4s, v5.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96e8a3)  // sdot v3.4s, v5.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6e8a2)  // sdot v2.4s, v5.16b, v22.4b[3]
    KAI_ASM_INST(0x4f94e027)  // sdot v7.4s, v1.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e026)  // sdot v6.4s, v1.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94e823)  // sdot v3.4s, v1.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4e822)  // sdot v2.4s, v1.16b, v20.4b[3]
    KAI_ASM_INST(0x4f93e347)  // sdot v7.4s, v26.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e346)  // sdot v6.4s, v26.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93eb43)  // sdot v3.4s, v26.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eb42)  // sdot v2.4s, v26.16b, v19.4b[3]
    scvtf v7.4s, v7.4s, #0x4
    scvtf v6.4s, v6.4s, #0x4
    scvtf v3.4s, v3.4s, #0x4
    fmla v31.4s, v7.4s, v16.4s
    scvtf v2.4s, v2.4s, #0x4
    fmla v30.4s, v6.4s, v16.4s
    fmla v29.4s, v3.4s, v16.4s
    fmla v28.4s, v2.4s, v16.4s
    subs x20, x20, #0x1
    bgt label_opt_17
    ld1 { v21.4s }, [x27]
    ldr q20, [x26, #0x0]
    add x27, x27, #0x10
    add x20, x12, #0x4
    ldr q19, [x27, #0x0]
    ldr q18, [x26, #0x10]
    cmp x25, #0x4
    add x26, x26, #0x20
    ld1r { v17.4s }, [x12]
    ld1r { v16.4s }, [x20]
    scvtf v21.4s, v21.4s
    fmla v31.4s, v20.4s, v21.s[0]
    fmla v30.4s, v20.4s, v21.s[1]
    fmla v29.4s, v20.4s, v21.s[2]
    fmla v28.4s, v20.4s, v21.s[3]
    fmul v31.4s, v31.4s, v19.s[0]
    fmul v30.4s, v30.4s, v19.s[1]
    fmul v29.4s, v29.4s, v19.s[2]
    fadd v31.4s, v31.4s, v18.4s
    fmul v28.4s, v28.4s, v19.s[3]
    fadd v30.4s, v30.4s, v18.4s
    fadd v29.4s, v29.4s, v18.4s
    fadd v28.4s, v28.4s, v18.4s
    fmax v31.4s, v31.4s, v17.4s
    fmax v30.4s, v30.4s, v17.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v28.4s, v28.4s, v17.4s
    fmin v31.4s, v31.4s, v16.4s
    fmin v30.4s, v30.4s, v16.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v28.4s, v28.4s, v16.4s
    blt label_opt_19
    mov x20, x15
    cmp x14, #0x1
    str q31, [x20, #0x0]
    add x20, x20, x13
    ble label_opt_22
    cmp x14, #0x2
    str q30, [x20, #0x0]
    add x20, x20, x13
    ble label_opt_22
    cmp x14, #0x3
    str q29, [x20, #0x0]
    add x20, x20, x13
    ble label_opt_22
    str q28, [x20, #0x0]
    b label_opt_22
KAI_ASM_LABEL(label_opt_19)  // Row tail: Partial output
    mov x23, x15
    cmp x14, #0x1
    add x22, x23, x13
    csel x22, x22, x23, GT
    cmp x14, #0x2
    add x21, x23, x13, LSL #1
    csel x21, x21, x22, GT
    cmp x14, #0x3
    add x20, x21, x13
    csel x20, x20, x21, GT
    tbz x25, #1, label_opt_20
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x21], #0x8
    st1 { v30.d }[0], [x22], #0x8
    st1 { v31.d }[0], [x23], #0x8
    tbz x25, #0, label_opt_21
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x21]
    st1 { v30.s }[2], [x22]
    st1 { v31.s }[2], [x23]
    b label_opt_21
KAI_ASM_LABEL(label_opt_20)  // Row tail: Output block 0: partial_1_0
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x21]
    st1 { v30.s }[0], [x22]
    st1 { v31.s }[0], [x23]
KAI_ASM_LABEL(label_opt_21)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_opt_22)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x15, x15, #0x10
    bgt label_opt_16
    subs x14, x14, #0x4
    add x8, x8, x6
    mov x15, x24
    bgt label_opt_15
KAI_ASM_LABEL(label_opt_23)  // Row tail: Row loop skip
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp d8, d9, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_opt32_neon_dotprod)

    KAI_ASM_END
