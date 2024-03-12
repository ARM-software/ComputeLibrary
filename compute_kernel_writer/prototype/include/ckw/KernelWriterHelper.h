/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef CKW_INCLUDE_CKW_KERNELWRITERHELPER_H
#define CKW_INCLUDE_CKW_KERNELWRITERHELPER_H

#include "ckw/KernelWriter.h"
#include "ckw/TensorOperand.h"
#include "ckw/TileOperand.h"

#include <iostream>
#include <type_traits>

/*
 * By including this header file you will be able to supplement the default
 * Compute Kernel Writer API with additional syntax to help ease the use of CKW.
 *
 * To use the KernelWriterHelper you need to wrap your instance of KernelWriter
 * (or any class deriving from KernelWriter):
 *      KernelWriterHelper<KernelWriter> writer;
 * The resulting writer object comprises the original KernelWriter
 * functionality (drop-in replacement), but extends the syntax as follows.
 *
 * Common functions/operators have natural syntax:
 *  1. Unary expressions:
 *          writer.op_assign(dst, !src);        // Logical NOT
 *          writer.op_assign(dst, ~src);        // Bitwise NOT
 *
 *  2. Binary expressions:
 *          writer.op_assign(dst, lhs + rhs);   // Addition
 *          writer.op_assign(dst, lhs - rhs);   // Subtraction
 *          writer.op_assign(dst, lhs * rhs);   // Multiplication
 *          writer.op_assign(dst, lhs / rhs);   // Division
 *          writer.op_assign(dst, lhs % rhs);   // Modulo
 *          writer.op_assign(dst, lhs == rhs);  // Equality
 *          writer.op_assign(dst, lhs < rhs);   // Less-than
 *          writer.op_assign(dst, lhs <= rhs);  // Less-than-or-equal
 *          writer.op_assign(dst, lhs > rhs);   // Greater-than
 *          writer.op_assign(dst, lhs >= rhs);  // Greater-than-or-equal
 *          writer.op_assign(dst, lhs ^ rhs);   // Bitwise XOR
 *          writer.op_assign(dst, logical_and(lhs, rhs));  // Logical AND
 *          writer.op_assign(dst, logical_or(lhs, rhs));   // Logical OR
 *
 *  3. Unary elementwise functions:
 *          writer.op_assign(dst, exp(src));    // Exponent
 *          writer.op_assign(dst, tanh(src));   // Hyperbolic tangent
 *          writer.op_assign(dst, sqrt(src));   // Square root
 *          writer.op_assign(dst, erf(src));    // Error function
 *          writer.op_assign(dst, fabs(src));   // Absolute of floating-point number
 *          writer.op_assign(dst, log(src));    // Natural logarithm
 *          writer.op_assign(dst, round(src));  // Round
 *          writer.op_assign(dst, sizeOf(src)); // sizeof
 *
 *  4. Binary elementwise functions:
 *          writer.op_assign(dst, max(first, second));      // Max
 *          writer.op_assign(dst, min(first, second));      // Min
 *
 *  5. Ternary elementwise functions:
 *          writer.op_assign(dst, select(first, second, third));    // Select
 *
 * NOTE: All the above examples support nesting, so you could write
 * something like: writer.op_assign(dst, src * (log(arg) + sqrt(abs(arg)));
 *
 *
 *  6. If-statements. The preceding syntax also allows easier writing of if-statements:
 *          writer.op_if(<cond>, <body>);
 *
 *     For example:
 *          writer.op_if(exp(first_arg) == dst, [&]{
 *              //...
 *          }).op_else_if(exp(first_arg) > dst, [&]{
 *              //...
 *          }).op_else([&] {
 *              //...
 *          });
 *
 *  7. For-loops. A similar syntax exists for for-loops:
 *          writer.op_for_loop(<cond>, <updater>, <body>);
 *
 *     For example:
 *          writer.op_for_loop(index < limit, index += step, [&]{
 *              //...
 *          });
 *
 * NOTE: There are limitations on the for-loop <cond> and <updater> parameters.
 * In neither the <cond> (Binary expression) or <updater> (Increment/Decrement)
 * is it allowed to use nesting. For example, `(index + other) < limit` and
 * `index < round(limit)` are invalid <cond> parameters. This is because the
 * semantics of for-loops rely on the condition being evaluated at every iteration,
 * but as temporary variables might be defined for nested expressions the semantics
 * cannot be guaranteed.
 */

namespace ckw
{

// ==================================================
// Type traits
// ==================================================

/** Specifies if the type can be used as an operand for functions (e.g. max), operations (e.g. *), or assignments. */
template <typename T>
struct can_be_operand : ::std::false_type
{
};

/** Specifies if the type can be assigned/written to. */
template <typename T>
struct can_be_assigned : ::std::false_type
{
};

template <>
struct can_be_operand<TileOperand &> : ::std::true_type
{
};

template <>
struct can_be_assigned<TileOperand &> : ::std::true_type
{
};

// ==================================================
// Assignment
// ==================================================

/** AST node for assignments.
 *
 * Note that \p TRight must be an operand, and \p TLeft must be assignable.
 *
 * @tparam TLeft The type of the destination of the assignment.
 * @tparam TRight The type of the source assigned to the destination.
 */
template <typename TLeft,
          typename TRight,
          typename = ::std::enable_if<can_be_operand<TRight>::value && can_be_assigned<TLeft>::value>>
struct Assignment
{
    TLeft        lhs;
    TRight       rhs;
    AssignmentOp opcode;
};

/** Represents the expression: `\p lhs += \p rhs`.
 *
 * @tparam      TLeft    The type of the LHS of the assignment.
 * @tparam      TRight   The type of the RHS of the assignment.
 * @param[in]   lhs      The LHS of the assignment.
 * @param[in]   rhs      The RHS of the assignment.
 * @return      The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline Assignment<TLeft, TRight> operator+=(TLeft &&lhs, TRight &&rhs)
{
    return Assignment<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), AssignmentOp::Increment};
}

/** Represents the expression: `\p lhs -= \p rhs`.
 *
 * @tparam      TLeft    The type of the LHS of the assignment.
 * @tparam      TRight   The type of the RHS of the assignment.
 * @param[in]   lhs    The LHS of the assignment.
 * @param[in]   rhs    The RHS of the assignment.
 * @return      The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline Assignment<TLeft, TRight> operator-=(TLeft &&lhs, TRight &&rhs)
{
    return Assignment<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), AssignmentOp::Decrement};
}

// ==================================================
// Unary expression
// ==================================================

/** AST node for unary expressions.
 *
 * Note that \p TSrc must be an operand.
 *
 * @tparam TSrc The type of the argument to the expression.
 */
template <typename TSrc, typename = ::std::enable_if<can_be_operand<TSrc>::value>>
struct UnaryExpression
{
    TSrc    src;
    UnaryOp opcode;
};

template <typename TLeft>
struct can_be_operand<UnaryExpression<TLeft>> : ::std::true_type
{
};

/** Represents the expression: `!\p src`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
inline UnaryExpression<TSrc> operator!(TSrc &&src)
{
    return UnaryExpression<TSrc>{std::forward<TSrc>(src), UnaryOp::LogicalNot};
}

/** Represents the expression: `~\p src`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
inline UnaryExpression<TSrc> operator~(TSrc &&src)
{
    return UnaryExpression<TSrc>{std::forward<TSrc>(src), UnaryOp::BitwiseNot};
}

// ==================================================
// Binary expressions
// ==================================================

/** AST node for binary expressions.
 *
 * Note that both \p TLeft and \p TRight must be operands.
 *
 * @tparam TLeft  The type of the left argument of the expression.
 * @tparam TRight The type of the right argument of the expression.
 */
template <typename TLeft,
          typename TRight,
          typename = ::std::enable_if_t<can_be_operand<TLeft>::value && can_be_operand<TRight>::value>>
struct BinaryExpression
{
    TLeft    lhs;
    TRight   rhs;
    BinaryOp opcode;
};

template <typename TLeft, typename TRight>
struct can_be_operand<BinaryExpression<TLeft, TRight>> : ::std::true_type
{
};

/** Represents the expression: `\p lhs + \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator+(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Add};
}

/** Represents the expression: `\p lhs - \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator-(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Sub};
}

/** Represents the expression: `\p lhs * \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator*(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Mul};
}

/** Represents the expression: `\p lhs / \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator/(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Div};
}

/** Represents the expression: `\p lhs % \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator%(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Mod};
}

/** Represents the expression: `\p lhs == \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator==(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Equal};
}

/** Represents the expression: `\p lhs < \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator<(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Less};
}

/** Represents the expression: `\p lhs <= \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator<=(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::LessEqual};
}

/** Represents the expression: `\p lhs > \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator>(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::Greater};
}

/** Represents the expression: `\p lhs >= \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator>=(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::GreaterEqual};
}

/** Represents the expression: `\p lhs ^ \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> operator^(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::BitwiseXOR};
}

/** Represents the expression: `\p lhs && \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> logical_and(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::LogicalAnd};
}

/** Represents the expression: `\p lhs && \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight, typename... TOps>
inline BinaryExpression<BinaryExpression<TLeft, TRight>, TOps...> logical_and(TLeft &&lhs, TRight &&rhs, TOps &&...ops)
{
    return logical_and(
        BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::LogicalAnd},
        std::forward<TOps>(ops)...);
}

/** Represents the expression: `\p lhs || \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight>
inline BinaryExpression<TLeft, TRight> logical_or(TLeft &&lhs, TRight &&rhs)
{
    return BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::LogicalOr};
}

/** Represents the expression: `\p lhs || \p rhs`.
 *
 * @tparam     TLeft  The type of the LHS of the expression.
 * @tparam     TRight The type of the RHS of the expression.
 * @param[in]  lhs    The LHS of the expression.
 * @param[in]  rhs    The RHS of the expression.
 * @return     The resulting AST node.
 */
template <typename TLeft, typename TRight, typename... TOps>
inline BinaryExpression<BinaryExpression<TLeft, TRight>, TOps...> logical_or(TLeft &&lhs, TRight &&rhs, TOps &&...ops)
{
    return logical_or(
        BinaryExpression<TLeft, TRight>{std::forward<TLeft>(lhs), std::forward<TRight>(rhs), BinaryOp::LogicalOr},
        std::forward<TOps>(ops)...);
}

// ==================================================
// Unary elementwise functions
// ==================================================

/** AST node for unary elementwise functions.
 *
 * Note that \p TSrc must be an operand.
 *
 * @tparam TSrc The type of the argument to the function.
 */
template <typename TSrc, typename = ::std::enable_if<can_be_operand<TSrc>::value>>
struct UnaryElementwiseFunction
{
    TSrc          src;
    UnaryFunction opcode;
};

template <typename TLeft>
struct can_be_operand<UnaryElementwiseFunction<TLeft>> : ::std::true_type
{
};

/** Represents the expression: `exp(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> exp(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::Exp};
}

/** Represents the expression: `tanh(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> tanh(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::Tanh};
}

/** Represents the expression: `sqrt(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> sqrt(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::Sqrt};
}

/** Represents the expression: `erf(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> erf(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::Erf};
}

/** Represents the expression: `fabs(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> fabs(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::Fabs};
}

/** Represents the expression: `log(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> log(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::Log};
}

/** Represents the expression: `round(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> round(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::Round};
}

/** Represents the expression: `sizeof(\p src)`.
 *
 * @tparam      TSrc The type of the argument.
 * @param[in]   src  The argument.
 * @return      The resulting AST node.
 */
template <typename TSrc>
UnaryElementwiseFunction<TSrc> sizeOf(TSrc &&src)
{
    return UnaryElementwiseFunction<TSrc>{std::forward<TSrc>(src), UnaryFunction::SizeOf};
}

// ==================================================
// Binary elementwise functions
// ==================================================

/** AST node for binary elementwise functions.
 *
 * Note that both \p TFirst and \p TSecond must be operands.
 *
 * @tparam TFirst  The type of the left argument of the function.
 * @tparam TSecond The type of the right argument of the function.
 */
template <typename TFirst,
          typename TSecond,
          typename = ::std::enable_if<can_be_operand<TFirst>::value && can_be_operand<TSecond>::value>>
struct BinaryElementwiseFunction
{
    TFirst         first;
    TSecond        second;
    BinaryFunction opcode;
};

template <typename TFirst, typename TSecond>
struct can_be_operand<BinaryElementwiseFunction<TFirst, TSecond>> : ::std::true_type
{
};

/** Represents the function call: `max(\p first, \p second)`.
 *
 * @tparam      TFirst  The type of the first argument.
 * @tparam      TSecond The type of the second argument.
 * @param[in]   first   The first argument.
 * @param[in]   second  The second argument.
 * @return      The resulting AST node.
 */
template <typename TFirst, typename TSecond>
BinaryElementwiseFunction<TFirst, TSecond> max(TFirst &&first, TSecond &&second)
{
    return BinaryElementwiseFunction<TFirst, TSecond>{std::forward<TFirst>(first), std::forward<TSecond>(second),
                                                      BinaryFunction::Max};
}

/** Represents the function call: `min(\p first, \p second)`.
 *
 * @tparam      TFirst  The type of the first argument.
 * @tparam      TSecond The type of the second argument.
 * @param[in]   first   The first argument.
 * @param[in]   second  The second argument.
 * @return      The resulting AST node.
 */
template <typename TFirst, typename TSecond>
BinaryElementwiseFunction<TFirst, TSecond> min(TFirst &&first, TSecond &&second)
{
    return BinaryElementwiseFunction<TFirst, TSecond>{std::forward<TFirst>(first), std::forward<TSecond>(second),
                                                      BinaryFunction::Min};
}

// ==================================================
// Ternary elementwise functions
// ==================================================

/** AST node for ternary elementwise functions.
 *
 * Note that \p TFirst, \p TSecond, and \p TThird all must be operands.
 *
 * @tparam TFirst The type of the first argument to the function.
 * @tparam TSecond The type of the second argument to the function.
 * @tparam TThird The type of the third argument to the function.
 */
template <typename TFirst,
          typename TSecond,
          typename TThird,
          typename = ::std::enable_if<can_be_operand<TFirst>::value && can_be_operand<TSecond>::value &&
                                      can_be_operand<TThird>::value>>
struct TernaryElementwiseFunction
{
    TFirst          first;
    TSecond         second;
    TThird          third;
    TernaryFunction opcode;
};

template <typename TFirst, typename TSecond, typename TThird>
struct can_be_operand<TernaryElementwiseFunction<TFirst, TSecond, TThird>> : ::std::true_type
{
};

/** Represents the function call: `select(\p first, \p second, \p third)`.
 *
 * @tparam      TFirst  The type of the first argument.
 * @tparam      TSecond The type of the second argument.
 * @tparam      TThird  The type of the third argument.
 * @param[in]   first   The first argument.
 * @param[in]   second  The second argument.
 * @param[in]   third   The third argument.
 * @return      The resulting AST node.
 */
template <typename TFirst, typename TSecond, typename TThird>
TernaryElementwiseFunction<TFirst, TSecond, TThird> select(TFirst &&first, TSecond &&second, TThird &&third)
{
    return TernaryElementwiseFunction<TFirst, TSecond, TThird>{std::forward<TFirst>(first),
                                                               std::forward<TSecond>(second),
                                                               std::forward<TThird>(third), TernaryFunction::Select};
}

/** Helper class used to extend a KernelWriter with additional functionality
 * in order to make writing easier.
 *
 * This extension automatically handles creation of temporary variables, and
 * allows nested function calls and operations.
 *
 * @tparam TWriter The type of KernelWriter to be overloaded. This must inherit from KernelWriter.
 */
template <class TWriter, typename = std::enable_if<std::is_base_of<KernelWriter, TWriter>::value>>
class KernelWriterHelper : public TWriter
{
public:
    using TWriter::TWriter;

    // ==================================================
    // If-statements
    // ==================================================

    // Un-hide original implementation, in case the original implementation is required.
    using TWriter::op_if;

    /** Represents the if-statement: `if(\p cond) { \p body }`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] cond The BinaryExpression representing the condition.
     * @param[in] body The body of the if-statement.
     */
    KernelWriterHelper<TWriter> &op_if(const BinaryExpression<TileOperand &, TileOperand &> &cond,
                                       const std::function<void()>                          &body)
    {
        TWriter::op_if(cond.lhs, cond.opcode, cond.rhs, body);
        return *this;
    }

    /** Represents the if-statement: `if(\p cond) { \p body }`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] cond The BinaryExpression representing the condition.
     * @param[in] body The body of the if-statement.
     */
    template <typename TRight>
    KernelWriterHelper<TWriter> &op_if(const BinaryExpression<TileOperand &, TRight> &cond,
                                       const std::function<void()>                   &body)
    {
        auto &tmp1 = declare_temp_tile(cond.lhs.tile_info());
        op_assign(tmp1, cond.rhs);
        TWriter::op_if(cond.lhs, cond.opcode, tmp1, body);
        return *this;
    }

    /** Represents the if-statement: `if(\p cond) { \p body }`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] cond The BinaryExpression representing the condition.
     * @param[in] body The body of the if-statement.
     */
    template <typename TLeft>
    KernelWriterHelper<TWriter> &op_if(const BinaryExpression<TLeft, TileOperand &> &cond,
                                       const std::function<void()>                  &body)
    {
        auto &tmp1 = declare_temp_tile(cond.rhs.tile_info());
        op_assign(tmp1, cond.lhs);
        TWriter::op_if(tmp1, cond.opcode, cond.rhs, body);
        return *this;
    }

    // Un-hide original implementation, in case the original implementation is required.
    using TWriter::op_else_if;

    /** Represents the else-if-statement: `else if(\p cond) { \p body }`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] cond The BinaryExpression representing the condition.
     * @param[in] body The body of the else-if-statement.
     */
    KernelWriterHelper<TWriter> &op_else_if(const BinaryExpression<TileOperand &, TileOperand &> &cond,
                                            const std::function<void()>                          &body)
    {
        TWriter::op_else_if(cond.lhs, cond.opcode, cond.rhs, body);
        return *this;
    }

    /** Represents the else-if-statement: `else if(\p cond) { \p body }`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] cond The BinaryExpression representing the condition.
     * @param[in] body The body of the else-if-statement.
     */
    template <typename TRight>
    KernelWriterHelper<TWriter> &op_else_if(const BinaryExpression<TileOperand &, TRight> &cond,
                                            const std::function<void()>                   &body)
    {
        auto &tmp1 = declare_temp_tile(cond.lhs.tile_info());
        op_assign(tmp1, cond.rhs);
        TWriter::op_else_if(cond.lhs, cond.opcode, tmp1, body);
        return *this;
    }

    /** Represents the else-if-statement: `else if(\p cond) { \p body }`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] cond The BinaryExpression representing the condition.
     * @param[in] body The body of the else-if-statement.
     */
    template <typename TLeft>
    KernelWriterHelper<TWriter> &op_else_if(const BinaryExpression<TLeft, TileOperand &> &cond,
                                            const std::function<void()>                  &body)
    {
        auto &tmp1 = declare_temp_tile(cond.rhs.tile_info());
        op_assign(tmp1, cond.lhs);
        TWriter::op_else_if(tmp1, cond.opcode, cond.rhs, body);
        return *this;
    }

    // ==================================================
    // For-loops
    // ==================================================

    // Un-hide original implementation, in case the original implementation is required.
    using TWriter::op_for_loop;

    /** Represents the for-loop: `for(;\p cond; \p updater) { \p body }`.
     *
     * The BinaryExpression for the condition and the Assignment
     * for the updater are unpacked and their components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] cond    The BinaryExpression representing the condition.
     * @param[in] updater The Assignment representing the updater.
     * @param[in] body    The body of the for-loop.
     */
    void op_for_loop(const BinaryExpression<TileOperand &, TileOperand &> &cond,
                     const Assignment<TileOperand &, TileOperand &>       &updater,
                     const std::function<void()>                          &body)
    {
        TWriter::op_for_loop(cond.lhs, cond.opcode, cond.rhs, updater.lhs, updater.opcode, updater.rhs, body);
    }

    // ==================================================
    // Unary expressions
    // ==================================================

    // Un-hide original implementation, in case the original implementation is required.
    using TWriter::op_assign;

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The UnaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The UnaryExpression representing the expression to be evaluated and assigned.
     */
    void op_assign(const TileOperand &dst, const UnaryExpression<TileOperand &> &exp)
    {
        TWriter::op_unary_expression(dst, exp.opcode, exp.src);
    }

    // ==================================================
    // Binary expressions
    // ==================================================

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryExpression representing the expression to be evaluated and assigned.
     */
    void op_assign(const TileOperand &dst, const BinaryExpression<TileOperand &, TileOperand &> &exp)
    {
        TWriter::op_binary_expression(dst, exp.lhs, exp.opcode, exp.rhs);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryExpression representing the expression to be evaluated and assigned.
     */
    template <typename TRight>
    void op_assign(const TileOperand &dst, const BinaryExpression<TileOperand &, TRight> &exp)
    {
        std::cout << "Beginning assignment!" << std::endl;
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.rhs);
        TWriter::op_binary_expression(dst, exp.lhs, exp.opcode, tmp1);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryExpression representing the expression to be evaluated and assigned.
     */
    template <typename TLeft>
    void op_assign(const TileOperand &dst, const BinaryExpression<TLeft, TileOperand &> &exp)
    {
        std::cout << "Beginning assignment!" << std::endl;
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.lhs);
        TWriter::op_binary_expression(dst, tmp1, exp.opcode, exp.rhs);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryExpression is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryExpression representing the expression to be evaluated and assigned.
     */
    template <typename TLeft, typename TRight>
    void op_assign(const TileOperand &dst, const BinaryExpression<TLeft, TRight> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        auto &tmp2 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.lhs);
        op_assign(tmp2, exp.rhs);
        TWriter::op_binary_expression(dst, tmp1, exp.opcode, tmp2);
    }

    // ==================================================
    // Unary elementwise functions
    // ==================================================

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The UnaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The UnaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    void op_assign(const TileOperand &dst, const UnaryElementwiseFunction<TileOperand &> &exp)
    {
        TWriter::op_unary_elementwise_function(dst, exp.opcode, exp.src);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The UnaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The UnaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TArg>
    void op_assign(const TileOperand &dst, const UnaryElementwiseFunction<TArg> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.lhs);
        TWriter::op_unary_elementwise_function(dst, exp.opcode, tmp1);
    }

    // ==================================================
    // Binary elementwise functions
    // ==================================================

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    void op_assign(const TileOperand &dst, const BinaryElementwiseFunction<TileOperand &, TileOperand &> &exp)
    {
        TWriter::op_binary_elementwise_function(dst, exp.opcode, exp.first, exp.second);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TRight>
    void op_assign(const TileOperand &dst, const BinaryElementwiseFunction<TileOperand &, TRight> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.second);
        TWriter::op_binary_elementwise_function(dst, exp.opcode, exp.first, tmp1);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TLeft>
    void op_assign(const TileOperand &dst, const BinaryElementwiseFunction<TLeft, TileOperand &> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.first);
        TWriter::op_binary_elementwise_function(dst, exp.opcode, tmp1, exp.second);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The BinaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The BinaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TLeft, typename TRight>
    void op_assign(const TileOperand &dst, const BinaryElementwiseFunction<TLeft, TRight> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        auto &tmp2 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.first);
        op_assign(tmp2, exp.second);
        TWriter::op_binary_elementwise_function(dst, exp.opcode, tmp1, tmp2);
    }

    // ==================================================
    // Ternary elementwise functions
    // ==================================================

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    void op_assign(const TileOperand                                                             &dst,
                   const TernaryElementwiseFunction<TileOperand &, TileOperand &, TileOperand &> &exp)
    {
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, exp.first, exp.second, exp.third);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TFirst>
    void op_assign(const TileOperand &dst, const TernaryElementwiseFunction<TFirst, TileOperand &, TileOperand &> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.first);
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, tmp1, exp.second, exp.third);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TSecond>
    void op_assign(const TileOperand &dst, const TernaryElementwiseFunction<TileOperand &, TSecond, TileOperand &> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.second);
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, exp.first, tmp1, exp.third);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TThird>
    void op_assign(const TileOperand &dst, const TernaryElementwiseFunction<TileOperand &, TileOperand &, TThird> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.third);
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, exp.first, exp.second, tmp1);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TFirst, typename TSecond>
    void op_assign(const TileOperand &dst, const TernaryElementwiseFunction<TFirst, TSecond, TileOperand &> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        auto &tmp2 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.first);
        op_assign(tmp2, exp.second);
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, tmp1, tmp2, exp.third);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TFirst, typename TThird>
    void op_assign(const TileOperand &dst, const TernaryElementwiseFunction<TFirst, TileOperand &, TThird> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        auto &tmp2 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.first);
        op_assign(tmp2, exp.third);
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, tmp1, exp.second, tmp2);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TSecond, typename TThird>
    void op_assign(const TileOperand &dst, const TernaryElementwiseFunction<TileOperand &, TSecond, TThird> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info());
        auto &tmp2 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.second);
        op_assign(tmp2, exp.third);
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, exp.first, tmp1, tmp2);
    }

    /** Represents the assignment: `\p dst = \p exp`.
     *
     * The TernaryElementwiseFunction is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] dst The tile which is assigned to.
     * @param[in] exp The TernaryElementwiseFunction representing the expression to be evaluated and assigned.
     */
    template <typename TFirst, typename TSecond, typename TThird>
    void op_assign(const TileOperand &dst, const TernaryElementwiseFunction<TFirst, TSecond, TThird> &exp)
    {
        auto &tmp1 = declare_temp_tile(dst.tile_info(), dst.tile_info(), dst.tile_info());
        auto &tmp2 = declare_temp_tile(dst.tile_info());
        auto &tmp3 = declare_temp_tile(dst.tile_info());
        op_assign(tmp1, exp.first);
        op_assign(tmp2, exp.second);
        op_assign(tmp3, exp.third);
        TWriter::op_ternary_elementwise_function(dst, exp.opcode, tmp1, tmp2, tmp3);
    }

    // ==================================================
    // Assignments
    // ==================================================

    /** Represents the assignment: `\p lhs += \p rhs` or `\p lhs -= \p rhs`.
     *
     * The Assignment is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @param[in] exp The Assignment representing the expression to be evaluated.
     */
    void op_assign(const Assignment<TileOperand &, TileOperand &> &exp)
    {
        if (exp.opcode == AssignmentOp::Increment)
        {
            TWriter::op_binary_expression(exp.lhs, exp.lhs, BinaryOp::Add, exp.rhs);
        }
        else if (exp.opcode == AssignmentOp::Decrement)
        {
            TWriter::op_binary_expression(exp.lhs, exp.lhs, BinaryOp::Sub, exp.rhs);
        }
    }

    /** Represents the assignment: `\p lhs += \p rhs` or `\p lhs -= \p rhs`.
     *
     * The Assignment is unpacked and its components are forwarded to
     * the underlying KernelWriter's implementation.
     *
     * @tparam    TRight The type of the RHS of the assignment.
     * @param[in] exp    The Assignment representing the expression to be evaluated.
     */
    template <typename TRight>
    void op_assign(const Assignment<TileOperand &, TRight> &exp)
    {
        auto &tmp1 = declare_temp_tile(exp.lhs.tile_info());
        op_assign(tmp1, exp.rhs);
        op_assign(Assignment<TileOperand &, TileOperand &>{exp.lhs, tmp1, exp.opcode});
    }

private:
    unsigned int temp_var_counter = 0;

    /** Return the current counter value, then increment it.
     *
     * @return The current counter value.
     */
    int next_ctr()
    {
        return temp_var_counter++;
    }

    /** Gets the next temporary variable counter value,
     * and returns a suitable temporary variable name.
     *
     * @return A temporary variable name.
     */
    std::string next_tmp_var_name()
    {
        return "tmp_" + std::to_string(next_ctr());
    }

    /** Returns the argument.
     *
     * Used for recursion with the variadic function version of this function.
     *
     * @param[in] arg The TileInfo to return.
     * @return    The \p arg.
     */
    TileInfo get_largest_size(const TileInfo &arg)
    {
        return arg;
    }

    /** Returns a TileInfo object where the size in each dimension (width, height) is the largest
     * of either TileInfo argument in the corresponding dimension.
     *
     * @tparam    TOps   Must be of TileInfo type.
     * @param[in] first  A TileInfo object.
     * @param[in] second A TileInfo object.
     * @param[in] ops    A number of TileInfo objects.
     * @return    A TileInfo object which represents the largest shape in each dimension across the arguments.
     */
    template <typename... TOps, typename = ::std::enable_if_t<std::is_same<TOps..., TileInfo>::value>>
    TileInfo get_largest_size(const TileInfo &first, const TileInfo &second, const TOps &...ops)
    {
        TileInfo largest = {first.data_type(), std::max(first.width(), second.width()),
                            std::max(first.height(), second.height())};
        return get_largest_size(largest, ops...);
    }

    /** Helper function to define a suitable TileOperand with appropriate TileInfo
     * such that broadcasting is taken into account, based on the arguments provided.
     *
     * @tparam     TArgs Must be of TileInfo type.
     * @param[in]  args  A number of TileInfo which determine the shape of the TileOperand to declare.
     * @return     A newly created TileOperand.
     */
    template <typename... TArgs, typename = ::std::enable_if_t<std::is_same<TArgs..., TileInfo>::value>>
    TileOperand &declare_temp_tile(const TArgs &...args)
    {
        return TWriter::declare_tile(next_tmp_var_name().c_str(), get_largest_size(args...));
    }
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_KERNELWRITERHELPER_H
