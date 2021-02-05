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
#include "arm_compute/runtime/CL/CLGEMMHeuristicsHandle.h"

#include "src/runtime/CL/mlgo/MLGOHeuristics.h"

namespace arm_compute
{
CLGEMMHeuristicsHandle::CLGEMMHeuristicsHandle()
    : _heuristics(std::make_unique<mlgo::MLGOHeuristics>())
{
}
CLGEMMHeuristicsHandle::~CLGEMMHeuristicsHandle() = default;

bool CLGEMMHeuristicsHandle::reload_from_file(const std::string &filename)
{
    return _heuristics->reload_from_file(filename);
}
const mlgo::MLGOHeuristics *CLGEMMHeuristicsHandle::get() const
{
    return _heuristics.get();
}

} // namespace arm_compute
