//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KAI_TEST_COMMON_ASSEMBLY_H
#define KAI_TEST_COMMON_ASSEMBLY_H

// clang-format off

#ifdef _MSC_VER

#define KAI_ASM_HEADER AREA |.text|, CODE, READONLY, ALIGN=4
#define KAI_ASM_LABEL(label) |label|
#define KAI_ASM_TARGET(label, direction) |label|
#define KAI_ASM_FUNCTION(label) |label|
#define KAI_ASM_EXPORT(label) global label
#define KAI_ASM_FOOTER end
#define KAI_ASM_INST(num) dcd num

#else  // _MSC_VER

#define KAI_ASM_HEADER .text
#define KAI_ASM_LABEL(label) label:
#define KAI_ASM_TARGET(label, direction) label##direction

#ifdef __APPLE__
#define KAI_ASM_FUNCTION(label) _##label:
#define KAI_ASM_EXPORT(label) \
    .global _##label;         \
    .type _##label, %function
#else  // __APPLE__
#define KAI_ASM_FUNCTION(label) label:
#define KAI_ASM_EXPORT(label) \
    .global label;            \
    .type label, %function
#endif  // __APPLE__

#define KAI_ASM_FOOTER
#define KAI_ASM_INST(num) .inst num

#endif  // _MSC_VER

// clang-format on

#endif  // KAI_TEST_COMMON_ASSEMBLY_H
