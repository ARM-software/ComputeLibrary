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
#ifndef SRC_CORE_SVE_KERNELS_ELEMENTWISE_LIST_H
#define SRC_CORE_SVE_KERNELS_ELEMENTWISE_LIST_H

#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/core/NEON/wrapper/svtraits.h"

namespace arm_compute
{
namespace cpu
{
using namespace arm_compute::wrapper;

template <typename VectorType>
VectorType elementwise_pow(svbool_t &pg, const VectorType &a, const VectorType &b)
{
    return svpow_z(pg, a, b);
}

template <typename VectorType>
VectorType elementwise_div(svbool_t &pg, const VectorType &a, const VectorType &b)
{
    return svdiv_z(pg, a, b);
}

template <uint32_t bytewidth>
svbool_t narrow_to_byte_predicate(svbool_t pg)
{
    const auto all_false = svpfalse();

    switch(bytewidth)
    {
        case 8:
            pg = svuzp1_b32(pg, all_false);
        /* fall through */
        case 4:
            pg = svuzp1_b16(pg, all_false);
        /* fall through */
        case 2:
            pg = svuzp1_b8(pg, all_false);
        /* fall through */
        default:
            break;
    }
    return pg;
}

template <typename VectorType>
VectorType elementwise_arithmetic_op(svbool_t &pg, const VectorType &a, const VectorType &b, ArithmeticOperation op)
{
    using ScalarType = typename wrapper::sve_scalar<VectorType>::type;
    VectorType res{};

    switch(op)
    {
        case ArithmeticOperation::MAX:
            res = svmax_z(pg, a, b);
            break;
        case ArithmeticOperation::MIN:
            res = svmin_z(pg, a, b);
            break;
        case ArithmeticOperation::SQUARED_DIFF:
        {
            const auto tmp = svsub_z(pg, a, b);
            res            = svmul_z(pg, tmp, tmp);
            break;
        }
        case ArithmeticOperation::PRELU:
        {
            const auto zero = svdup_n(ScalarType(0));
            const auto tmp  = svmul_z(pg, a, b);
            const auto gt   = svcmpgt(pg, a, zero);
            res             = svsel(gt, a, tmp);
            break;
        }
        case ArithmeticOperation::DIV:
        {
            res = elementwise_div(pg, a, b);
            break;
        }
        case ArithmeticOperation::POWER:
        {
            res = elementwise_pow(pg, a, b);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return res;
}

template <typename InputVectorType, typename OutputVectorType>
OutputVectorType elementwise_comparison_op(svbool_t &pg, const InputVectorType &a, const InputVectorType &b, ComparisonOperation op)
{
    svbool_t selection_vector{};

    switch(op)
    {
        case ComparisonOperation::Equal:
            selection_vector = svcmpeq(pg, a, b);
            break;
        case ComparisonOperation::NotEqual:
            selection_vector = svcmpne(pg, a, b);
            break;
        case ComparisonOperation::Greater:
            selection_vector = svcmpgt(pg, a, b);
            break;
        case ComparisonOperation::GreaterEqual:
            selection_vector = svcmpge(pg, a, b);
            break;
        case ComparisonOperation::Less:
            selection_vector = svcmplt(pg, a, b);
            break;
        case ComparisonOperation::LessEqual:
            selection_vector = svcmple(pg, a, b);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    using InputScalarType = typename wrapper::sve_scalar<InputVectorType>::type;
    selection_vector      = narrow_to_byte_predicate<sizeof(InputScalarType)>(selection_vector);

    using OutputScalarType  = typename wrapper::sve_scalar<OutputVectorType>::type;
    const auto false_vector = svdup_n(static_cast<OutputScalarType>((uint32_t)0));
    const auto true_vector  = svdup_n(static_cast<OutputScalarType>(~(uint32_t)0));
    auto       ret          = svsel(selection_vector, true_vector, false_vector);

    return ret;
}

template <typename ScalarType>
void elementwise_arithmetic_op(const ITensor *in1, const ITensor *in2, ITensor *out, ArithmeticOperation op, const Window &window);

template <typename ScalarType, typename OutputScalarType = uint8_t>
void elementwise_comparison_op(const ITensor *in1, const ITensor *in2, ITensor *out, ComparisonOperation op, const Window &window);
} // namespace cpu
} // namespace arm_compute
#endif /* SRC_CORE_SVE_KERNELS_ELEMENTWISE_LIST_H */
