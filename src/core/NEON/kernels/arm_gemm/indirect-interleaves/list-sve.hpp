/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "sme_interleave1VL_bf16_bf16.hpp"
#include "sme_interleave1VL_block2_bf16_bf16.hpp"
#include "sme_interleave1VL_block2_fp16_fp16.hpp"
#include "sme_interleave1VL_block4_s8_s8.hpp"
#include "sme_interleave1VL_block4_u8_u8.hpp"
#include "sme_interleave1VL_block4_s8_s8_summing.hpp"
#include "sme_interleave1VL_block4_u8_u8_summing.hpp"
#include "sme_interleave1VL_fp16_fp16.hpp"
#include "sme_interleave1VL_fp32_fp32.hpp"
#include "sme_interleave2VL_block2_bf16_bf16.hpp"
#include "sme_interleave2VL_block2_fp16_fp16.hpp"
#include "sme_interleave2VL_block4_s8_s8.hpp"
#include "sme_interleave2VL_block4_s8_s8_summing.hpp"
#include "sme_interleave2VL_block4_u8_u8.hpp"
#include "sme_interleave2VL_block4_u8_u8_summing.hpp"
#include "sme_interleave2VL_fp16_fp16.hpp"
#include "sme_interleave2VL_bf16_bf16.hpp"
#include "sme_interleave2VL_fp32_fp32.hpp"
#include "sme_interleave4VL_block2_bf16_bf16.hpp"
#include "sme_interleave4VL_block2_fp16_fp16.hpp"
#include "sme_interleave4VL_block4_s8_s8.hpp"
#include "sme_interleave4VL_block4_u8_u8.hpp"
#include "sme_interleave4VL_block4_s8_s8_summing.hpp"
#include "sme_interleave4VL_block4_u8_u8_summing.hpp"
#include "sme_interleave4VL_fp32_fp32.hpp"

#include "sme2_interleave1VL_block2_fp32_bf16.hpp"
#include "sme2_interleave2VL_block2_fp32_bf16.hpp"
#include "sme2_interleave4VL_block2_fp32_bf16.hpp"
