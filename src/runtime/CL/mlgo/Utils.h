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
#ifndef SRC_RUNTIME_CL_MLGO_UTILS_H
#define SRC_RUNTIME_CL_MLGO_UTILS_H

#include "src/runtime/CL/mlgo/Common.h"
#include "src/runtime/CL/mlgo/HeuristicTree.h"
#include "src/runtime/CL/mlgo/MLGOHeuristics.h"
#include "src/runtime/CL/mlgo/MLGOParser.h"

#include <ostream>
#include <string>

namespace arm_compute
{
namespace mlgo
{
std::ostream &operator<<(std::ostream &os, const GEMMConfigNative &config);
std::ostream &operator<<(std::ostream &os, const GEMMConfigReshapedOnlyRHS &config);
std::ostream &operator<<(std::ostream &os, const GEMMConfigReshaped &config);
std::ostream &operator<<(std::ostream &os, HeuristicType ht);
std::ostream &operator<<(std::ostream &os, DataType dt);
std::ostream &operator<<(std::ostream &os, const HeuristicTree::Index &index);
std::ostream &operator<<(std::ostream &os, const Query &query);
std::string to_string(const GEMMConfigNative &config);
std::string to_string(const GEMMConfigReshapedOnlyRHS &config);
std::string to_string(const GEMMConfigReshaped &config);
std::string to_string(const Query &query);
namespace parser
{
std::ostream &operator<<(std::ostream &os, const CharPosition &pos);
} // namespace parser
} // namespace mlgo
} // namespace arm_compute

#endif //SRC_RUNTIME_CL_MLGO_UTILS_H