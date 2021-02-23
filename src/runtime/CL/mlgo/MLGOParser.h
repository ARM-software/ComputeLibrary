/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_RUNTIME_CL_MLGO_MLGO_PARSER_H
#define SRC_RUNTIME_CL_MLGO_MLGO_PARSER_H

#include "src/runtime/CL/mlgo/MLGOHeuristics.h"

#include <deque>
#include <istream>
#include <string>
#include <utility>

/** A DotMLGO file parser (LL(k) parser)
 *
 * The grammar of DotMLGO is defined as the following ENBF:
 *
 * delim = "," | "\n"; // Note that delimiters are omitted from the definition below
 *
 * mlgo = header, heuristics-table, {heuristic-tree};
 *
 * header = "<header>", gemm-version, ip-type, "</header>";
 * gemm-version = "gemm-version",  "[", int,  int,  int, "]";
 * ip-type = "ip-type",  ("gpu" | "cpu");
 *
 * heiristics-table = "<heuristics-table>", {heuristics-table-entry}, "</heuristics-table>";
 * heuristics-table-entry = entry-id,  ip-name,  num-cores, data-type,  gpu-priority,  gpu-behavior,  heuristic-type,  free-vars;
 * entry-id = int;
 * ip-name = char-sequence;
 * num-cores = int;
 * data-type = "f32" | "f16" | "qasymm8";
 * gpu-priority = "best-performance" | "best-memory-usage";
 * gpu-behavior = "static" | "dynamic";
 * heuristic-type = "gemm-type" | "gemm-config-native" | "gemm-config-reshaped-only-rhs" | "gemm-config-reshaped";
 * free-vars = "[", {char-sequence}, "]";
 *
 * heuristic-tree = "<heuristic",  entry-id, ">", {tree-node}, "</heuristic>";
 * tree-node = branch-node | leaf-node;
 * branch-node = "b",  entry-id,  lhs-type,  lhs-value,  conditional-op,  rhs-type,  rhs-value,  true-node,  false-node;
 * lhs-type = comparator-type;
 * lhs-value = comparator-value;
 * rhs-type = comparator-type;
 * rhs-value = comparator-value;
 * comparator-type = "var" | "num" | "enum";
 * comparator-value = char-sequence | float;
 * conditional-op = "<" | "<=" | "==" | ">=" | ">";
 * true-node = entry-id;
 * false-node = entry-id;
 * leaf-node = "l",  entry-id,  heuristic-type,  leaf-value;
 * leaf-value = gemm-type | gemm-config-native | gemm-config-reshaped-only-rhs | gemm-config-reshaped
 * gemm-type = "native" | "reshaped-only-rhs" | "reshaped";
 * gemm-config-native = "[", int, int, int, "]";
 * gemm-config-reshaped-only-rhs = "[", int, int, int, int, bool, bool, bool, "]";
 * gemm-config-reshaped = "[", int, int, int, int, int, bool, bool, bool, bool, "]";
 */

namespace arm_compute
{
namespace mlgo
{
namespace parser
{
/** Type of Token */
enum class TokenType
{
    L_List = '[', /**< List open */
    R_List = ']', /**< List close */
    Int,          /**< Integral */
    Float,        /**< Floating */
    Text,         /**< Text/String */
    End,          /**< End of stream */
};

struct CharPosition
{
    bool operator==(const CharPosition &other) const
    {
        return ln == other.ln && col == other.col;
    }

    size_t ln{ 0 };
    size_t col{ 0 };
};

/** Token */
struct Token
{
    Token(TokenType t, std::string v, CharPosition pos)
        : type{ t }, value{ v }, pos{ pos }
    {
    }

    bool operator==(const Token &other) const
    {
        return type == other.type && value == other.value && pos == other.pos;
    }

    TokenType    type;  /**< Token type */
    std::string  value; /**< Token value */
    CharPosition pos;
};

/** A stream of token */
class TokenStream
{
    // NOTE: _tokens is never empty. The end of token stream is signalled by the End Token
public:
    static constexpr size_t max_look_ahead = 10;

public:
    /** Constructor
     *
     * @param[in] s      Input stream
     * @param[in] delims Delimiter characters packed in a string. Each char from the string can be used as a delim on its own
     */
    TokenStream(std::istream &s, const std::string &delims = ",\n");

    /** Check if there're more (non-End) Tokens
     * @return true  If there are more tokens
     * @return false If reached end of stream (only End token)
     */
    explicit operator bool() const;

    /** Get and pop off the current token
     *
     * @return Token
     */
    Token take();

    /** Peek the next ith token
     *
     * @param[in] i The next ith token. i < @ref max_look_ahead.
     *
     * @return Token
     */
    Token peek(size_t i = 0);

    /** Get the position of the current token
     *
     * @return CharPosition
     */
    CharPosition current_pos() const
    {
        return _tokens.front().pos;
    }

private:
    void read();

    Token recognize_tok(char ch);

    Token num_st(std::string value = "");

    Token float_after_dp_st(std::string value = "");

    Token text_st(std::string value = "");

    bool reached_end() const;

    bool is_delim(char ch) const;

    std::string       _delims;
    std::istream     &_istream;
    std::deque<Token> _tokens;
    CharPosition      _lookahead_pos;
};

/** Parse and construct a @ref MLGOHeuristics from input stream
 *
 * @param[in] in Input stream
 *
 * @return MLGOHeuristics
 */
std::pair<bool, MLGOHeuristics> parse_mlgo(std::istream &in);

} // namespace parser
} // namespace mlgo
} // namespace arm_compute
#endif //SRC_RUNTIME_CL_MLGO_MLGO_PARSER_H