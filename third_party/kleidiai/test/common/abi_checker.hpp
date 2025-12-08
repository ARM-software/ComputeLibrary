//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include "kai/kai_common.h"

namespace kai::test {

#if defined(__ARM_FEATURE_SME) && !_MSC_VER

/// Checker for FP ABI compliance
template <typename Func, typename... Args>
inline auto abi_check_fp(Func&& func, Args&&... args)
    -> decltype(std::invoke(std::forward<Func>(func), std::forward<Args>(args)...)) {
    using ResultType = std::invoke_result_t<Func, Args...>;
    using StorageType = std::conditional_t<std::is_void_v<ResultType>, int, ResultType>;

    std::optional<StorageType> result;
    static constexpr const uint64_t canary = 0xAAAABBBBCCCCDDDDULL;

    /* The block below will attempt to verify that FP registers are preserved
     * as expected. GP registers are not really easily possible to verify
     * using this method, as this function itself might change them */
    __asm__ __volatile__(
        // Fill callee saved registers with canaries
        "ldr x9, [%x[canary]]\n\t"

        // FP registers
        "fmov d8, x9\n\t"
        "fmov d9, x9\n\t"
        "fmov d10, x9\n\t"
        "fmov d11, x9\n\t"
        "fmov d12, x9\n\t"
        "fmov d13, x9\n\t"
        "fmov d14, x9\n\t"
        "fmov d15, x9\n\t"
        :
        : [canary] "r"(&canary)
        : "x9",  // Canary storage
          "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");

    if constexpr (std::is_void_v<ResultType>) {
        std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    } else {
        result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    }

    uint64_t first_mismatch = canary;
    __asm__ __volatile__(
        // Check that canary is still present in all callee saved registers
        "ldr x9, [%x[canary]]\n\t"

        // Check FP registers
        "11: fmov x10, d8\n\t"
        "cmp x10, x9\n\t"
        "b.eq 12f\n\t"
        "mov %x[first_mismatch], #8\n\t"
        "b 20f\n\t"

        "12: fmov x10, d9\n\t"
        "cmp x10, x9\n\t"
        "b.eq 13f\n\t"
        "mov %x[first_mismatch], #9\n\t"
        "b 20f\n\t"

        "13: fmov x10, d10\n\t"
        "cmp x10, x9\n\t"
        "b.eq 14f\n\t"
        "mov %x[first_mismatch], #10\n\t"
        "b 20f\n\t"

        "14: fmov x10, d11\n\t"
        "cmp x10, x9\n\t"
        "b.eq 15f\n\t"
        "mov %x[first_mismatch], #11\n\t"
        "b 20f\n\t"

        "15: fmov x10, d12\n\t"
        "cmp x10, x9\n\t"
        "b.eq 16f\n\t"
        "mov %x[first_mismatch], #12\n\t"
        "b 20f\n\t"

        "16: fmov x10, d13\n\t"
        "cmp x10, x9\n\t"
        "b.eq 17f\n\t"
        "mov %x[first_mismatch], #13\n\t"
        "b 20f\n\t"

        "17: fmov x10, d14\n\t"
        "cmp x10, x9\n\t"
        "b.eq 18f\n\t"
        "mov %x[first_mismatch], #14\n\t"
        "b 20f\n\t"

        "18: fmov x10, d15\n\t"
        "cmp x10, x9\n\t"
        "b.eq 20f\n\t"
        "mov %x[first_mismatch], #15\n\t"

        "20:\n\t"
        : [first_mismatch] "+r"(first_mismatch)
        : [canary] "r"(&canary)
        : "cc", "x9", "x10");

    KAI_ASSERT_MSG(first_mismatch == canary, "FP register corruption detected");
    if constexpr (!std::is_void_v<ResultType>) {
        return *result;
    }
}

/// Checker for SME ABI compliance
template <typename Func, typename... Args>
__arm_new("za") __arm_locally_streaming inline auto abi_check_za(Func&& func, Args&&... args)
    -> decltype(std::invoke(std::forward<Func>(func), std::forward<Args>(args)...)) {
    using ResultType = std::invoke_result_t<Func, Args...>;
    using StorageType = std::conditional_t<std::is_void_v<ResultType>, int, ResultType>;

    std::optional<StorageType> result;
    static constexpr const uint64_t canary = 0xAAAABBBBCCCCDDDDULL;

    /* This block attempts to check if ZA register is correctly preserved
     * by filling with a known pattern, and then checking pattern after
     * returning from function call */
    __asm__ __volatile__(
        "ldr x9, [%x[canary]]\n\t"

        // Fill ZA with canary pattern
        "dup z16.d, x9\n\t"                         // Broadcast canary to vector
        "rdsvl x9, #1\n\t"                          // Read number of ZA rows
        "ptrue p0.b\n\t"                            // Make p0.b fully enabled
        "mov w12, wzr\n\t"                          // Set row index to 0
        "1: mova za0h.b[w12, #0], p0/m, z16.b\n\t"  // copy vector tor row
        "add w12, w12, #1\n\t"                      // Increment row index
        "cmp x12, x9\n\t"                           // Repeat until all rows are filled
        "blt 1b\n\t"
        :
        : [canary] "r"(&canary)
        : "cc",
          "x9",   // Canary storage, and then ZA row count
          "x12",  // ZA row index
          "z16",  // canary vector
          "p0",   // predicate
          "za");

    if constexpr (std::is_void_v<ResultType>) {
        abi_check_fp(std::forward<Func>(func), std::forward<Args>(args)...);
    } else {
        result = abi_check_fp<Func>(std::forward<Func>(func), std::forward<Args>(args)...);
    }

    uint64_t first_mismatch = canary;
    __asm__ __volatile__(
        // Check that canary is still present in ZA
        "ldr x9, [%x[canary]]\n\t"

        "dup z16.d, x9\n\t"                          // Broadcast canary to vector
        "rdsvl x9, #1\n\t"                           // get rows of ZA
        "ptrue p0.b\n\t"                             // Make p0.b fully enabled
        "mov w12, wzr\n\t"                           // Clear w12
        "20: mova z17.b, p0/m, za0h.b[w12, #0]\n\t"  // Read row w12 from ZA
        "cmpne p1.b, p0/z, z16.b, z17.b\n\t"         // p1 = true for any mismatch
        "cntp x10, p0, p1.b\n\t"                     // x10 = number of mismatches
        "cmp x10, xzr\n\t"                           // if (mismatches == 0)
        "b.eq 21f\n\t"                               //   proceed
        "mov %x[first_mismatch], x12\n\t"            // else, store mismatching row
        "b 30f\n\t"                                  //   and leave checker
        "21: add w12, w12, #1\n\t"                   // w12 += 1
        "cmp x12, x9\n\t"                            // if (w12 < SVL_b)
        "blt 20b\n\t"                                //   check next row
        "30:\n\t"
        : [first_mismatch] "+r"(first_mismatch)
        : [canary] "r"(&canary)
        : "cc",
          "x9",   // Canary storage, then row ZA row count
          "x10",  // Row mismatch counter
          "x12",  // Row index
          "z16",  // Canary vector
          "z17",  // Current ZA row
          "p0");
    KAI_ASSERT_MSG(first_mismatch == canary, "ZA register corruption detected");

    if constexpr (!std::is_void_v<ResultType>) {
        return *result;
    }
}

/// Wrapper for checking ABI compliance
template <typename Func, typename... Args>
inline auto abi_check(Func&& func, Args&&... args)
    -> decltype(std::invoke(std::forward<Func>(func), std::forward<Args>(args)...)) {
    using ResultType = std::invoke_result_t<Func, Args...>;
    using StorageType = std::conditional_t<std::is_void_v<ResultType>, int, ResultType>;

    std::optional<StorageType> result;

    if constexpr (std::is_void_v<ResultType>) {
        abi_check_za<Func>(std::forward<Func>(func), std::forward<Args>(args)...);
    } else {
        result = abi_check_za<Func>(std::forward<Func>(func), std::forward<Args>(args)...);
    }

    if constexpr (!std::is_void_v<ResultType>) {
        return *result;
    }
}

#else

/// Call wrapped function, without any checking
template <typename Func, typename... Args>
inline auto abi_check(Func&& func, Args&&... args)
    -> decltype(std::invoke(std::forward<Func>(func), std::forward<Args>(args)...)) {
    using ResultType = std::invoke_result_t<Func, Args...>;
    using StorageType = std::conditional_t<std::is_void_v<ResultType>, int, ResultType>;

    std::optional<StorageType> result;

    if constexpr (std::is_void_v<ResultType>) {
        std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    } else {
        result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    }

    if constexpr (!std::is_void_v<ResultType>) {
        return *result;
    }
}

#endif

}  // namespace kai::test
