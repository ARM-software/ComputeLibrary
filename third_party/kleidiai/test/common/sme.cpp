//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE)
#error This file must be compiled for AArch64, FEAT_SVE.
#else  // Architectural features check.

#include "test/common/sme.hpp"

#include "test/common/cpu_info.hpp"

namespace kai::test {

template <>
uint64_t get_sme_vector_length<1>() {
    static uint64_t res = 0;

    if (res == 0) {
        if (cpu_has_sme()) {
            __asm__ __volatile__(
                ".inst 0xd503477f  // SMSTART ZA\n"
                "cntb %0\n"
                ".inst 0xd503467f  // SMSTOP\n"
                : "=r"(res)
                :
                : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
                  "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
                  "z30", "z31");
        } else {
            res = 1;
        }
    }

    return res;
}

template <>
uint64_t get_sme_vector_length<2>() {
    static uint64_t res = 0;

    if (res == 0) {
        if (cpu_has_sme()) {
            __asm__ __volatile__(
                ".inst 0xd503477f  // SMSTART ZA\n"
                "cnth %0\n"
                ".inst 0xd503467f  // SMSTOP\n"
                : "=r"(res)
                :
                : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
                  "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
                  "z30", "z31");
        } else {
            res = 1;
        }
    }

    return res;
}

template <>
uint64_t get_sme_vector_length<4>() {
    static uint64_t res = 0;

    if (res == 0) {
        if (cpu_has_sme()) {
            __asm__ __volatile__(
                ".inst 0xd503477f  // SMSTART ZA\n"
                "cntw %0\n"
                ".inst 0xd503467f  // SMSTOP\n"
                : "=r"(res)
                :
                : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
                  "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
                  "z30", "z31");
        } else {
            res = 1;
        }
    }

    return res;
}

}  // namespace kai::test

#endif  // Architectural features check.
