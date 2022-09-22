/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_INTRINSICS_H
#define ARM_COMPUTE_WRAPPER_INTRINSICS_H

#include "src/core/NEON/wrapper/intrinsics/abs.h"
#include "src/core/NEON/wrapper/intrinsics/add.h"
#include "src/core/NEON/wrapper/intrinsics/and.h"
#include "src/core/NEON/wrapper/intrinsics/bsl.h"
#include "src/core/NEON/wrapper/intrinsics/ceq.h"
#include "src/core/NEON/wrapper/intrinsics/cge.h"
#include "src/core/NEON/wrapper/intrinsics/cgt.h"
#include "src/core/NEON/wrapper/intrinsics/cgtz.h"
#include "src/core/NEON/wrapper/intrinsics/cle.h"
#include "src/core/NEON/wrapper/intrinsics/clt.h"
#include "src/core/NEON/wrapper/intrinsics/combine.h"
#include "src/core/NEON/wrapper/intrinsics/cvt.h"
#include "src/core/NEON/wrapper/intrinsics/div.h"
#include "src/core/NEON/wrapper/intrinsics/dup_n.h"
#include "src/core/NEON/wrapper/intrinsics/eor.h"
#include "src/core/NEON/wrapper/intrinsics/erf.h"
#include "src/core/NEON/wrapper/intrinsics/exp.h"
#include "src/core/NEON/wrapper/intrinsics/ext.h"
#include "src/core/NEON/wrapper/intrinsics/gethigh.h"
#include "src/core/NEON/wrapper/intrinsics/getlane.h"
#include "src/core/NEON/wrapper/intrinsics/getlow.h"
#include "src/core/NEON/wrapper/intrinsics/inv.h"
#include "src/core/NEON/wrapper/intrinsics/invsqrt.h"
#include "src/core/NEON/wrapper/intrinsics/load.h"
#include "src/core/NEON/wrapper/intrinsics/log.h"
#include "src/core/NEON/wrapper/intrinsics/max.h"
#include "src/core/NEON/wrapper/intrinsics/min.h"
#include "src/core/NEON/wrapper/intrinsics/mla.h"
#include "src/core/NEON/wrapper/intrinsics/movl.h"
#include "src/core/NEON/wrapper/intrinsics/movn.h"
#include "src/core/NEON/wrapper/intrinsics/mul.h"
#include "src/core/NEON/wrapper/intrinsics/neg.h"
#include "src/core/NEON/wrapper/intrinsics/not.h"
#include "src/core/NEON/wrapper/intrinsics/orr.h"
#include "src/core/NEON/wrapper/intrinsics/pmax.h"
#include "src/core/NEON/wrapper/intrinsics/pmin.h"
#include "src/core/NEON/wrapper/intrinsics/pow.h"
#include "src/core/NEON/wrapper/intrinsics/qmov.h"
#include "src/core/NEON/wrapper/intrinsics/qmovun.h"
#include "src/core/NEON/wrapper/intrinsics/reinterpret.h"
#include "src/core/NEON/wrapper/intrinsics/rev64.h"
#include "src/core/NEON/wrapper/intrinsics/round.h"
#include "src/core/NEON/wrapper/intrinsics/setlane.h"
#include "src/core/NEON/wrapper/intrinsics/shr.h"
#include "src/core/NEON/wrapper/intrinsics/sin.h"
#include "src/core/NEON/wrapper/intrinsics/sqrt.h"
#include "src/core/NEON/wrapper/intrinsics/store.h"
#include "src/core/NEON/wrapper/intrinsics/sub.h"
#include "src/core/NEON/wrapper/intrinsics/tanh.h"
#include "src/core/NEON/wrapper/intrinsics/tbl.h"

#if defined(__ARM_FEATURE_SVE)
#include "src/core/NEON/wrapper/intrinsics/svcnt.h"
#include "src/core/NEON/wrapper/intrinsics/svcvt.h"
#include "src/core/NEON/wrapper/intrinsics/svdup_n.h"
#include "src/core/NEON/wrapper/intrinsics/svexp.h"
#include "src/core/NEON/wrapper/intrinsics/svlog.h"
#include "src/core/NEON/wrapper/intrinsics/svpow.h"
#include "src/core/NEON/wrapper/intrinsics/svptrue.h"
#include "src/core/NEON/wrapper/intrinsics/svqadd.h"
#include "src/core/NEON/wrapper/intrinsics/svsin.h"
#include "src/core/NEON/wrapper/intrinsics/svwhilelt.h"
#endif /* defined(__ARM_FEATURE_SVE) */

#endif /* ARM_COMPUTE_WRAPPER_INTRINSICS_H */
