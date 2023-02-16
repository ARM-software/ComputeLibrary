/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifdef ARM_COMPUTE_ENABLE_SME
#include "sme_transpose_interleave_16VL_1x4.hpp"
#include "sme_transpose_interleave_16VL_2x2_fp32bf16.hpp"
#include "sme_transpose_interleave_16VL_2x2.hpp"
#include "sme_transpose_interleave_16VL.hpp"
#include "sme_transpose_interleave_1VL_1x4.hpp"
#include "sme_transpose_interleave_1VL_2x2_fp32bf16.hpp"
#include "sme_transpose_interleave_1VL_2x2.hpp"
#include "sme_transpose_interleave_1VL.hpp"
#include "sme_transpose_interleave_2VL_1x4.hpp"
#include "sme_transpose_interleave_2VL_2x2.hpp"
#include "sme_transpose_interleave_2VL_2x2_fp32bf16.hpp"
#include "sme_transpose_interleave_2VL.hpp"
#include "sme_transpose_interleave_4VL_1x4.hpp"
#include "sme_transpose_interleave_4VL_2x2.hpp"
#include "sme_transpose_interleave_4VL_2x2_fp32bf16.hpp"
#include "sme_transpose_interleave_4VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SME
#include "sve_transpose_interleave_12VL_2x4_fp32bf16.hpp"
#include "sve_transpose_interleave_1VL_1x4.hpp"
#include "sve_transpose_interleave_1VL.hpp"
#include "sve_transpose_interleave_3VL_1x4.hpp"
#include "sve_transpose_interleave_3VL_2x2.hpp"
#include "sve_transpose_interleave_3VL.hpp"
#include "sve_transpose_interleave_4VL_1x4.hpp"
#include "sve_transpose_interleave_4VL_2x2.hpp"
#include "sve_transpose_interleave_4VL.hpp"
#include "sve_transpose_interleave_6VL_1x8.hpp"
#include "sve_transpose_interleave_6VL_2x4_fp32bf16.hpp"
#include "sve_transpose_interleave_6VL_2x4.hpp"
#include "sve_transpose_interleave_6VL_4x2.hpp"
#include "sve_transpose_interleave_8VL_1x4.hpp"
#include "sve_transpose_interleave_8VL_1x8.hpp"
#include "sve_transpose_interleave_8VL_2x2.hpp"
#include "sve_transpose_interleave_8VL_2x4.hpp"
#include "sve_transpose_interleave_8VL_2x4_fp32bf16.hpp"
#include "sve_transpose_interleave_8VL.hpp"
