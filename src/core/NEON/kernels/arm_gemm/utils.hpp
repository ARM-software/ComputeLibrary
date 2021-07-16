/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#include "arm_gemm.hpp"

#include <cstddef>

// Macro for unreachable code (e.g. impossible default cases on switch)
#define UNREACHABLE(why)  __builtin_unreachable()

// Paranoid option for the above with assert
// #define UNREACHABLE(why)   assert(0 && why)

namespace arm_gemm {

template<typename T>
std::string get_type_name() {
#ifdef __GNUC__
    std::string s = __PRETTY_FUNCTION__;

    auto start = s.find("cls_");

    if (start==std::string::npos) {
        return "(unknown)";
    }

    for(size_t x = start+4; x<s.size(); x++) {
        if (s[x] == ';' || s[x] == ']') {
            return s.substr(start+4, x-(start+4));
        }
    }

    return "(unknown)";
#else
    return "(unsupported)";
#endif
}

template<typename T>
inline T iceildiv(const T a, const T b) {
    return (a + b - 1) / b;
}

template <typename T>
inline T roundup(const T a, const T b) {
    T rem = a % b;

    if (rem) {
        return a + b - rem;
    } else {
        return a;
    }
}

enum class VLType {
    None,
    SVE,
};

template<typename T>
struct IndirectOutputArg {
    struct {
        T       *base;
        size_t   stride;
    } direct = {};
    struct {
        T * const *ptr;
        size_t     offset;
    } indirect = {};
    bool is_indirect;

    // Direct
    IndirectOutputArg(T *base, size_t stride) : is_indirect(false) {
        direct.base = base;
        direct.stride = stride;
    }

    // Indirect
    IndirectOutputArg(T * const * ptr, size_t offset) : is_indirect(true) {
        indirect.ptr = ptr;
        indirect.offset = offset;
    }

    IndirectOutputArg() : is_indirect(false) {
        direct.base = nullptr;
        direct.stride = 0;
    }
};

// Check that the provided Requantize32 doesn't have a left shift.
inline bool quant_no_left_shift(const Requantize32 &qp) {
    if (qp.per_channel_requant) {
        return (qp.per_channel_left_shifts == nullptr);
    } else {
        return (qp.per_layer_left_shift == 0);
    }
}

// Check that the provided Requantize32 is compatible with the "symmetric" hybrid kernels.  These don't include row
// sums, so the 'b_offset' has to be zero.
inline bool quant_hybrid_symmetric(const Requantize32 &qp) {
    return quant_no_left_shift(qp) && qp.b_offset == 0;
}

// Check that the provided Requantize32 is compatible with the "asymmetric" hybrid kernels.  These don't support per
// channel quantization.  Technically b_offset==0 cases would work, but it is a waste to sum and then multiply by 0...
inline bool quant_hybrid_asymmetric(const Requantize32 &qp) {
    return quant_no_left_shift(qp) /*  && qp.b_offset != 0 */ && qp.per_channel_requant==false;
}

template<typename T>
struct IndirectInputArg {
    struct {
        const T *base;
        size_t   stride;
    } direct = {};
    struct {
        const T * const * const * ptr;
        unsigned int start_row;
        unsigned int start_col;
    } indirect = {};
    bool is_indirect;

    // Direct
    IndirectInputArg(const T *base, size_t stride) : is_indirect(false) {
        direct.base = base;
        direct.stride = stride;
    }

    // Indirect
    IndirectInputArg(const T * const * const *ptr, unsigned int start_row, unsigned int start_col) : is_indirect(true) {
        indirect.ptr = ptr;
        indirect.start_row = start_row;
        indirect.start_col = start_col;
    }

    IndirectInputArg() : is_indirect(false) {
        direct.base = nullptr;
        direct.stride = 0;
    }
};

namespace utils {
template <typename T>
inline unsigned long get_vector_length() {
#if defined(ARM_COMPUTE_ENABLE_SVE)
    uint64_t vl;

    __asm __volatile (
        ".inst 0x0420e3e0\n" // CNTB X0, ALL, MUL #1
        "mov %0, X0\n"
        : "=r" (vl)
        :
        : "x0"
    );

    return vl / sizeof(T);
#else // !defined(ARM_COMPUTE_ENABLE_SVE)
    return 16 / sizeof(T);
#endif // defined(ARM_COMPUTE_ENABLE_SVE)
}

template <typename T>
inline unsigned long get_vector_length(VLType vl_type) {
  switch (vl_type) {
#if defined(ARM_COMPUTE_ENABLE_SVE)
    case VLType::SVE:
      return get_vector_length<T>();
#endif // defined(ARM_COMPUTE_ENABLE_SVE)
    default:
      return 16 / sizeof(T);
  }
}
} // utils namespace
} // arm_gemm namespace

using namespace arm_gemm::utils;
