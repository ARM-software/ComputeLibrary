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
#include "src/runtime/CL/mlgo/MLGOParser.h"

#include "arm_compute/core/Log.h"

#include "src/runtime/CL/mlgo/Utils.h"

#include <sstream>

#define CHECK(parser_expr, valid_var) \
    (parser_expr);                    \
    if (!valid_var)                   \
        return;

#define CHECK_DEFAULT(parser_expr, valid_var, default_val) \
    (parser_expr);                                         \
    if (!valid_var)                                        \
        return default_val;

#ifdef ARM_COMPUTE_LOGGING_ENABLED

#define FAIL_WITH_MSG(valid_var, pos, msg)           \
    std::stringstream ss;                            \
    ss << "MLGOParser Error: " << pos << " " << msg; \
    ARM_COMPUTE_LOG_INFO_MSG_CORE(ss.str().c_str()); \
    valid_var = false;                               \
    return;

#define FAIL_WITH_MSG_DEFAULT(valid_var, default_val, pos, msg) \
    std::stringstream ss;                                       \
    ss << "MLGOParser Error: " << pos << " " << msg;            \
    ARM_COMPUTE_LOG_INFO_MSG_CORE(ss.str().c_str());            \
    valid_var = false;                                          \
    return default_val;

#define LOG_TOKEN_POS(tokens, pos_var) const auto pos_var = tokens.current_pos();

#else // ARM_COMPUTE_LOGGING_ENABLED

#define FAIL_WITH_MSG(valid_var, pos, msg) \
    valid_var = false;                     \
    return;

#define FAIL_WITH_MSG_DEFAULT(valid_var, default_val, pos, msg) \
    valid_var = false;                                          \
    return default_val;

#define LOG_TOKEN_POS(tokens, pos_var)

#endif // ARM_COMPUTE_LOGGING_ENABLED
namespace
{
void ltrim(std::string &str)
{
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace(ch); }));
}

void rtrim(std::string &str)
{
    str.erase(std::find_if(str.rbegin(), str.rend(), [](char ch) { return !std::isspace(ch); }).base(), str.end());
}

void trim(std::string &str)
{
    ltrim(str);
    rtrim(str);
}
} // namespace

namespace arm_compute
{
namespace mlgo
{
namespace parser
{
enum class ComparatorType
{
    Enum,
    Num,
    Var
};

TokenStream::TokenStream(std::istream &s, const std::string &delims)
    : _delims{delims}, _istream{s}, _tokens{}, _lookahead_pos{}
{
    read();
}

TokenStream::operator bool() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_tokens.empty(), "TokenStream can never be empty");
    return !reached_end();
}

Token TokenStream::take()
{
    ARM_COMPUTE_ERROR_ON_MSG(_tokens.empty(), "TokenStream can never be empty");
    Token t = _tokens.front();
    _tokens.pop_front();
    if (_tokens.empty())
    {
        read();
    }
    return t;
}
Token TokenStream::peek(size_t i)
{
    ARM_COMPUTE_ERROR_ON_MSG(_tokens.empty(), "TokenStream can never be empty");
    ARM_COMPUTE_ERROR_ON_MSG(i >= max_look_ahead, "TokenStream: Exceeding max look ahead");
    // NOTE: If i exceeds the stream (_istream.eof()), read() automatically appends a End token at the end
    while (_istream && _tokens.size() <= i)
    {
        read();
    }
    size_t ind = std::min(i, _tokens.size() - 1);
    return _tokens[ind];
}

void advance(CharPosition &pos, char ch)
{
    if (ch == '\n')
    {
        pos.ln += 1;
        pos.col = 0;
    }
    else
    {
        pos.col += 1;
    }
}
void rewind(CharPosition &pos)
{
    pos.col -= 1;
}
void TokenStream::read()
{
    char ch;
    // Skip any leading space and delim characters
    do
    {
        // Reached eof
        if (!_istream.get(ch))
        {
            if (!reached_end())
            {
                _tokens.emplace_back(TokenType::End, "", _lookahead_pos);
            }
            return;
        }
        advance(_lookahead_pos, ch);
    } while (std::isspace(ch) || is_delim(ch));
    // Read chars until we hit a delim or eof
    auto orig_pos = _lookahead_pos;
    auto tok      = recognize_tok(ch);
    rewind(orig_pos);
    tok.pos = orig_pos;
    // Trim leading and trailing white spaces
    trim(tok.value);
    _tokens.push_back(tok);
}

Token TokenStream::recognize_tok(char ch)
{
    if (ch == '[')
    {
        return Token{TokenType::L_List, "", _lookahead_pos};
    }
    else if (ch == ']')
    {
        return Token{TokenType::R_List, "", _lookahead_pos};
    }
    else if (ch == '.')
    {
        return float_after_dp_st(std::string{ch});
    }
    else if (std::isdigit(ch))
    {
        return num_st(std::string{ch});
    }
    else
    {
        return text_st(std::string{ch});
    }
}

Token TokenStream::num_st(std::string value)
{
    char ch{};
    while (_istream.get(ch))
    {
        advance(_lookahead_pos, ch);
        if (ch == '.')
        {
            return float_after_dp_st(value + ch);
        }
        else if (!std::isdigit(ch))
        {
            if (!is_delim(ch) && !std::isspace(ch))
            {
                rewind(_lookahead_pos);
                _istream.unget();
            }
            break;
        }
        value += ch;
    }
    return Token{TokenType::Int, value, _lookahead_pos};
}

Token TokenStream::float_after_dp_st(std::string value)
{
    char ch{};
    while (_istream.get(ch))
    {
        advance(_lookahead_pos, ch);
        if (!std::isdigit(ch))
        {
            if (!is_delim(ch) && !std::isspace(ch))
            {
                rewind(_lookahead_pos);
                _istream.unget();
            }
            break;
        }
        value += ch;
    }
    return Token{TokenType::Float, value, _lookahead_pos};
}

Token TokenStream::text_st(std::string value)
{
    char ch{};
    while (_istream.get(ch))
    {
        advance(_lookahead_pos, ch);
        if (is_delim(ch))
        {
            break;
        }
        if (ch == '[' || ch == ']')
        {
            rewind(_lookahead_pos);
            _istream.unget();
            break;
        }
        value += ch;
    }
    return Token{TokenType::Text, value, _lookahead_pos};
}

bool TokenStream::reached_end() const
{
    return _tokens.size() == 1 && _tokens.front().type == TokenType::End;
}

bool TokenStream::is_delim(char ch) const
{
    return _delims.find(ch) != std::string::npos;
}

void end(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    auto tok = in.take();
    if (tok.type != TokenType::End)
    {
        FAIL_WITH_MSG(valid, pos, "Unexpected token at the end of stream");
    }
}

bool bool_val(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    auto tok = in.take();
    if (tok.type != TokenType::Int)
    {
        FAIL_WITH_MSG_DEFAULT(valid, false, pos, "Expect bool or int token");
    }
    bool val{};
    std::stringstream(tok.value) >> val;
    return val;
}

int int_val(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    auto tok = in.take();
    if (tok.type != TokenType::Int)
    {
        FAIL_WITH_MSG_DEFAULT(valid, -1, pos, "Expect int token");
    }
    int val{};
    std::stringstream(tok.value) >> val;
    return val;
}

unsigned int uint_val(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    int val = CHECK_DEFAULT(int_val(in, valid), valid, 0);
    if (val < 0)
    {
        FAIL_WITH_MSG_DEFAULT(valid, 0, pos, "Expect unsigned int token");
    }
    return static_cast<unsigned int>(val);
}

float float_val(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    auto tok = in.take();
    if (tok.type != TokenType::Float)
    {
        FAIL_WITH_MSG_DEFAULT(valid, 0.f, pos, "Expect float token");
    }
    float val{};
    std::stringstream(tok.value) >> val;
    return val;
}

std::string text_val(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    auto tok = in.take();
    if (tok.type != TokenType::Text || tok.value.empty())
    {
        FAIL_WITH_MSG_DEFAULT(valid, "", pos, "Expect a non-empty text token");
    }
    return tok.value;
}

bool accept_text(TokenStream &in, const std::string &c_str, bool take = true)
{
    auto tok = in.peek();
    if (tok.type == TokenType::Text && tok.value == c_str)
    {
        if (take)
        {
            in.take();
        }
        return true;
    }
    return false;
}

void expect_text(TokenStream &in, const std::string &str, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (!accept_text(in, str))
    {
        FAIL_WITH_MSG(valid, pos, std::string("Expect text token: ") + str);
    }
}

bool accept_l_list(TokenStream &in)
{
    auto tok = in.peek();
    if (tok.type == TokenType::L_List)
    {
        in.take();
        return true;
    }
    return false;
}

void expect_l_list(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (!accept_l_list(in))
    {
        FAIL_WITH_MSG(valid, pos, "Expect '['");
    }
}

bool accept_r_list(TokenStream &in)
{
    auto tok = in.peek();
    if (tok.type == TokenType::R_List)
    {
        in.take();
        return true;
    }
    return false;
}

void expect_r_list(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (!accept_r_list(in))
    {
        FAIL_WITH_MSG(valid, pos, "Expect ']'");
    }
}

ConditionalOp conditional_op(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "<="))
    {
        return ConditionalOp::LE;
    }
    else if (accept_text(in, ">="))
    {
        return ConditionalOp::GE;
    }
    else if (accept_text(in, "=="))
    {
        return ConditionalOp::EQ;
    }
    else if (accept_text(in, "<"))
    {
        return ConditionalOp::LT;
    }
    else if (accept_text(in, ">"))
    {
        return ConditionalOp::GT;
    }
    else
    {
        FAIL_WITH_MSG_DEFAULT(valid, ConditionalOp::EQ, pos, "Expect conditional op");
    }
}

void gemm_version(TokenStream &in, bool &valid)
{
    CHECK(expect_text(in, "gemm-version", valid), valid);
    CHECK(expect_l_list(in, valid), valid);
    CHECK(uint_val(in, valid), valid);
    CHECK(uint_val(in, valid), valid);
    CHECK(uint_val(in, valid), valid);
    CHECK(expect_r_list(in, valid), valid);
}

void ip_type(TokenStream &in, bool &valid)
{
    CHECK(expect_text(in, "ip-type", valid), valid);
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "gpu"))
    {
        ;
    }
    else if (accept_text(in, "cpu"))
    {
        ;
    }
    else
    {
        FAIL_WITH_MSG(valid, pos, "Expect ip type");
    }
}

void header(TokenStream &in, bool &valid)
{
    CHECK(expect_text(in, "<header>", valid), valid);
    CHECK(gemm_version(in, valid), valid);
    CHECK(ip_type(in, valid), valid);
    CHECK(expect_text(in, "</header>", valid), valid);
}

DataType data_type(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "f16"))
    {
        return DataType::F16;
    }
    else if (accept_text(in, "f32"))
    {
        return DataType::F32;
    }
    else if (accept_text(in, "qasymm8"))
    {
        return DataType::QASYMM8;
    }
    else
    {
        FAIL_WITH_MSG_DEFAULT(valid, DataType::QASYMM8, pos, "Expect data type");
    }
}

ComparatorType comparator_type(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "var"))
    {
        return ComparatorType::Var;
    }
    else if (accept_text(in, "num"))
    {
        return ComparatorType::Num;
    }
    else if (accept_text(in, "enum"))
    {
        return ComparatorType::Enum;
    }
    else
    {
        FAIL_WITH_MSG_DEFAULT(valid, ComparatorType::Num, pos, "Expect comparator type");
    }
}

HeuristicType heuristic_type(TokenStream &in, bool &valid, bool take = true)
{
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "gemm-type", take))
    {
        return HeuristicType::GEMM_Type;
    }
    else if (accept_text(in, "gemm-config-native", take))
    {
        return HeuristicType::GEMM_Config_Native;
    }
    else if (accept_text(in, "gemm-config-reshaped-only-rhs", take))
    {
        return HeuristicType::GEMM_Config_Reshaped_Only_RHS;
    }
    else if (accept_text(in, "gemm-config-reshaped", take))
    {
        return HeuristicType::GEMM_Config_Reshaped;
    }
    else
    {
        FAIL_WITH_MSG_DEFAULT(valid, HeuristicType::GEMM_Config_Reshaped, pos, "Expect heuristic type");
    }
}

void expect_heuristic_type(TokenStream &in, HeuristicType expected_ht, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    auto ht = CHECK(heuristic_type(in, valid, false), valid);
    if (ht != expected_ht)
    {
        FAIL_WITH_MSG(valid, pos, "Unexpected heuristic type");
    }
    CHECK(heuristic_type(in, valid, true), valid);
}

GEMMType gemm_type(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "native"))
    {
        return GEMMType::NATIVE;
    }
    else if (accept_text(in, "reshaped-only-rhs"))
    {
        return GEMMType::RESHAPED_ONLY_RHS;
    }
    else if (accept_text(in, "reshaped"))
    {
        return GEMMType::RESHAPED;
    }
    else
    {
        FAIL_WITH_MSG_DEFAULT(valid, GEMMType::RESHAPED_ONLY_RHS, pos, "Expect gemm type");
    }
}

GEMMConfigNative gemm_config_native(TokenStream &in, bool &valid)
{
    const auto invalid_val = GEMMConfigNative{};
    CHECK_DEFAULT(expect_l_list(in, valid), valid, invalid_val);
    const auto m0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto n0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto k0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    CHECK_DEFAULT(expect_r_list(in, valid), valid, invalid_val);
    return GEMMConfigNative{m0, n0, k0};
}

GEMMConfigReshapedOnlyRHS gemm_config_reshaped_only_rhs(TokenStream &in, bool &valid)
{
    const auto invalid_val = GEMMConfigReshapedOnlyRHS{};
    CHECK_DEFAULT(expect_l_list(in, valid), valid, invalid_val);
    const auto m0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto n0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto k0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto h0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto ir = CHECK_DEFAULT(bool_val(in, valid), valid, invalid_val);
    const auto tr = CHECK_DEFAULT(bool_val(in, valid), valid, invalid_val);
    const auto ex = CHECK_DEFAULT(bool_val(in, valid), valid, invalid_val);
    CHECK_DEFAULT(expect_r_list(in, valid), valid, invalid_val);
    return GEMMConfigReshapedOnlyRHS{m0, n0, k0, h0, ir, tr, ex};
}

GEMMConfigReshaped gemm_config_reshaped(TokenStream &in, bool &valid)
{
    const auto invalid_val = GEMMConfigReshaped{};
    CHECK_DEFAULT(expect_l_list(in, valid), valid, invalid_val);
    const auto m0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto n0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto k0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto v0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto h0 = CHECK_DEFAULT(uint_val(in, valid), valid, invalid_val);
    const auto il = CHECK_DEFAULT(bool_val(in, valid), valid, invalid_val);
    const auto ir = CHECK_DEFAULT(bool_val(in, valid), valid, invalid_val);
    const auto tr = CHECK_DEFAULT(bool_val(in, valid), valid, invalid_val);
    const auto ex = CHECK_DEFAULT(bool_val(in, valid), valid, invalid_val);
    CHECK_DEFAULT(expect_r_list(in, valid), valid, invalid_val);
    return GEMMConfigReshaped{m0, n0, k0, v0, h0, il, ir, tr, ex};
}

void gpu_priority(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "best-performance"))
    {
        ;
    }
    else if (accept_text(in, "best-memory-usage"))
    {
        ;
    }
    else
    {
        FAIL_WITH_MSG(valid, pos, "Expect gpu priority");
    }
}

void gpu_behavior(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    if (accept_text(in, "static"))
    {
        ;
    }
    else if (accept_text(in, "dynamic"))
    {
        ;
    }
    else
    {
        FAIL_WITH_MSG(valid, pos, "Expect ip type");
    }
}

void free_vars(TokenStream &in, bool &valid)
{
    CHECK(expect_l_list(in, valid), valid);
    while (!accept_r_list(in))
    {
        CHECK(text_val(in, valid), valid);
    }
}

void heuristics_table_entry(TokenStream &in, MLGOHeuristics &h, bool &valid)
{
    const auto id = CHECK(uint_val(in, valid), valid);
    const auto ip = CHECK(text_val(in, valid), valid);
    CHECK(uint_val(in, valid), valid); // Num cores
    const auto dt = CHECK(data_type(in, valid), valid);
    CHECK(gpu_priority(in, valid), valid);
    CHECK(gpu_behavior(in, valid), valid);
    const auto ht = CHECK(heuristic_type(in, valid), valid);
    CHECK(free_vars(in, valid), valid);
    HeuristicTree t(id, ht, ip, dt);
    valid = CHECK(h.add_heuristic_tree(std::move(t)), valid);
}

void heuristics_table(TokenStream &in, MLGOHeuristics &h, bool &valid)
{
    CHECK(expect_text(in, "<heuristics-table>", valid), valid);
    while (!accept_text(in, "</heuristics-table>"))
    {
        CHECK(heuristics_table_entry(in, h, valid), valid);
    }
}

Condition condition(TokenStream &in, bool &valid)
{
    LOG_TOKEN_POS(in, pos);
    // NOTE: Only simplified Conditions are accepted, which means the lhs comparator type is fixed to Var and that of
    // the rhs is fixed to Num (float)
    const auto invalid_val = Condition{};
    const auto l_t         = CHECK_DEFAULT(comparator_type(in, valid), valid, invalid_val);
    const auto l_v         = CHECK_DEFAULT(text_val(in, valid), valid, invalid_val);
    const auto c_o         = CHECK_DEFAULT(conditional_op(in, valid), valid, invalid_val);
    const auto r_t         = CHECK_DEFAULT(comparator_type(in, valid), valid, invalid_val);
    const auto r_v         = CHECK_DEFAULT(float_val(in, valid), valid, invalid_val);
    if (l_t != ComparatorType::Var || r_t != ComparatorType::Num)
    {
        FAIL_WITH_MSG_DEFAULT(valid, invalid_val, pos,
                              "Only accept LHS type to be Var (string) and RHS type to be Num (float)");
    }
    return Condition{l_v, c_o, r_v};
}

void heuristic_tree(TokenStream &in, MLGOHeuristics &h, bool &valid)
{
    CHECK(expect_text(in, "<heuristic", valid), valid);
    const auto tree_id = CHECK(uint_val(in, valid), valid);
    CHECK(expect_text(in, ">", valid), valid);
    HeuristicTree *t                     = nullptr;
    std::tie(valid, t)                   = CHECK(h.get_heuristic_tree(tree_id), valid);
    const HeuristicType t_heuristic_type = std::get<0>(t->index());
    while (!accept_text(in, "</heuristic>"))
    {
        LOG_TOKEN_POS(in, pos);
        if (accept_text(in, "b"))
        {
            // Branch node
            const auto id   = CHECK(uint_val(in, valid), valid);
            const auto cond = CHECK(condition(in, valid), valid);
            const auto t_id = CHECK(uint_val(in, valid), valid);
            const auto f_id = CHECK(uint_val(in, valid), valid);
            valid           = CHECK(t->add_branch(id, cond, t_id, f_id), valid);
        }
        else if (accept_text(in, "l"))
        {
            // Leaf node
            const auto id = CHECK(uint_val(in, valid), valid);
            // NOTE: Heuristic type within each tree appears to be redundant (same information can be obtained from the
            // heuristic table). For now it remains as a step for validation.
            LOG_TOKEN_POS(in, pos);
            CHECK(expect_heuristic_type(in, t_heuristic_type, valid), valid);
            switch (t_heuristic_type)
            {
                case HeuristicType::GEMM_Type:
                {
                    const auto g_type = CHECK(gemm_type(in, valid), valid);
                    valid             = CHECK(t->add_leaf(id, g_type), valid);
                    break;
                }
                case HeuristicType::GEMM_Config_Native:
                {
                    const auto g_c = CHECK(gemm_config_native(in, valid), valid);
                    valid          = CHECK(t->add_leaf(id, g_c), valid);
                    break;
                }
                case HeuristicType::GEMM_Config_Reshaped_Only_RHS:
                {
                    const auto g_c = CHECK(gemm_config_reshaped_only_rhs(in, valid), valid);
                    valid          = CHECK(t->add_leaf(id, g_c), valid);
                    break;
                }
                case HeuristicType::GEMM_Config_Reshaped:
                {
                    const auto g_c = CHECK(gemm_config_reshaped(in, valid), valid);
                    valid          = CHECK(t->add_leaf(id, g_c), valid);
                    break;
                }
                default:
                {
                    FAIL_WITH_MSG(valid, pos, "Unexpected heuristic type");
                }
            }
        }
        else
        {
            FAIL_WITH_MSG(valid, pos, "Expect tree node type");
        }
    }
    // Perform semantic checks in the middle of parsing so that it can fail fast should there be any invalidities
    valid = CHECK(h.check_heuristic_tree(tree_id), valid);
}

MLGOHeuristics mlgo(TokenStream &in, bool &valid)
{
    MLGOHeuristics h;
    CHECK_DEFAULT(header(in, valid), valid, h);
    CHECK_DEFAULT(heuristics_table(in, h, valid), valid, h);
    while (accept_text(in, "<heuristic", false))
    {
        CHECK_DEFAULT(heuristic_tree(in, h, valid), valid, h);
    }
    CHECK_DEFAULT(end(in, valid), valid, h);
    valid = CHECK_DEFAULT(h.check_all(), valid, h);
    return h;
}

std::pair<bool, MLGOHeuristics> parse_mlgo(std::istream &in)
{
    auto tokens = TokenStream(in);
    bool valid  = true;
    auto h      = mlgo(tokens, valid);
    return std::make_pair(std::move(valid), std::move(h));
}
} // namespace parser
} // namespace mlgo
} // namespace arm_compute

#undef CHECK
#undef CHECK_DEFAULT
#undef FAIL_WITH_MSG
#undef FAIL_WITH_MSG_DEFAULT
