/*
 * Copyright (c) 2020 Arm Limited.
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
#include "a32_transpose_interleave_8way_32bit.hpp"
#include "a64_transpose_interleave_12_1x4.hpp"
#include "a64_transpose_interleave_12_1x8.hpp"
#include "a64_transpose_interleave_12_2x2.hpp"
#include "a64_transpose_interleave_12_2x4_fp32bf16.hpp"
#include "a64_transpose_interleave_12_2x4.hpp"
#include "a64_transpose_interleave_128.hpp"
#include "a64_transpose_interleave_12_s8s16.hpp"
#include "a64_transpose_interleave_12_u8u16.hpp"
#include "a64_transpose_interleave_16_1x4.hpp"
#include "a64_transpose_interleave_16_1x8.hpp"
#include "a64_transpose_interleave_16_2x2.hpp"
#include "a64_transpose_interleave_16_2x4.hpp"
#include "a64_transpose_interleave_16_2x4_fp32bf16.hpp"
#include "a64_transpose_interleave_16.hpp"
#include "a64_transpose_interleave_24_bf16fp32.hpp"
#include "a64_transpose_interleave_24_fp16fp32.hpp"
#include "a64_transpose_interleave_24_2x4_fp32bf16.hpp"
#include "a64_transpose_interleave_24.hpp"
#include "a64_transpose_interleave_32_1x4.hpp"
#include "a64_transpose_interleave_32_2x2.hpp"
#include "a64_transpose_interleave_4_1x16.hpp"
#include "a64_transpose_interleave_4_1x4.hpp"
#include "a64_transpose_interleave_48.hpp"
#include "a64_transpose_interleave_64.hpp"
#include "a64_transpose_interleave_96.hpp"
