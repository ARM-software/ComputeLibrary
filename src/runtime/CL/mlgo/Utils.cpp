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
#include "src/runtime/CL/mlgo/Utils.h"

namespace arm_compute
{
namespace mlgo
{
std::ostream &operator<<(std::ostream &os, const GEMMConfigNative &config)
{
    return os << "Native:{"
           << "m0: " << config.m0 << ", "
           << "n0: " << config.n0 << ", "
           << "k0: " << config.k0 << ", "
           << "}";
}
std::ostream &operator<<(std::ostream &os, const GEMMConfigReshapedOnlyRHS &config)
{
    return os << "ReshapedOnlyRHS:{"
           << "m0: " << config.m0 << ", "
           << "n0: " << config.n0 << ", "
           << "k0: " << config.k0 << ", "
           << "h0: " << config.h0 << ", "
           << "interleave_rhs: " << config.interleave_rhs << ", "
           << "transpose_rhs: " << config.transpose_rhs << ", "
           << "export_cl_image: " << config.export_cl_image
           << "}";
}
std::ostream &operator<<(std::ostream &os, const GEMMConfigReshaped &config)
{
    return os << "Reshaped:{"
           << "m0: " << config.m0 << ", "
           << "n0: " << config.n0 << ", "
           << "k0: " << config.k0 << ", "
           << "v0: " << config.v0 << ", "
           << "h0: " << config.h0 << ", "
           << "interleave_lhs: " << config.interleave_lhs << ", "
           << "interleave_rhs: " << config.interleave_rhs << ", "
           << "transpose_rhs: " << config.transpose_rhs << ", "
           << "export_cl_image: " << config.export_cl_image
           << "}";
}
std::ostream &operator<<(std::ostream &os, const HeuristicType &ht)
{
    switch(ht)
    {
        case HeuristicType::GEMM_Type:
        {
            os << "GEMM_Type";
            break;
        }
        case HeuristicType::GEMM_Config_Reshaped_Only_RHS:
        {
            os << "GEMM_Config_Reshaped_Only_RHS";
            break;
        }
        case HeuristicType::GEMM_Config_Reshaped:
        {
            os << "GEMM_Config_Reshaped";
            break;
        }
        default:
        {
            os << "Unknown";
            break;
        }
    }
    return os;
}
std::ostream &operator<<(std::ostream &os, const DataType &dt)
{
    switch(dt)
    {
        case DataType::F32:
        {
            os << "F32";
            break;
        }
        case DataType::F16:
        {
            os << "F16";
            break;
        }
        case DataType::QASYMM8:
        {
            os << "QASYMM8";
            break;
        }
        default:
        {
            os << "Unknown";
            break;
        }
    }
    return os;
}
std::ostream &operator<<(std::ostream &os, const HeuristicTree::Index &index)
{
    HeuristicType ht;
    std::string   ip;
    DataType      dt;
    std::tie(ht, ip, dt) = index;
    os << "Index(";
    os << "HeuristicType=" << ht << ",";
    os << "IP=" << ip << ",";
    os << "DataType=" << dt;
    os << ")";
    return os;
}

namespace parser
{
std::ostream &operator<<(std::ostream &os, CharPosition pos)
{
    os << "(Ln: " << pos.ln << ", Col: " << pos.col << ")";
    return os;
}
} // namespace parser

} // namespace mlgo

} // namespace arm_compute