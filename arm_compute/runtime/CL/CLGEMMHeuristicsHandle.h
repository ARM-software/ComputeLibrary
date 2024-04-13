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
#ifndef ARM_COMPUTE_RUNTIME_CL_CL_GEMM_HEURISTICS_HANDLE_H
#define ARM_COMPUTE_RUNTIME_CL_CL_GEMM_HEURISTICS_HANDLE_H

#include <memory>

namespace arm_compute
{
namespace mlgo
{
/** Forward declaration for the underlying heuristics: MLGOHeuristics */
class MLGOHeuristics;
} // namespace mlgo

/** Handle for loading and retrieving GEMM heuristics */
class CLGEMMHeuristicsHandle
{
public:
    /** Constructor */
    CLGEMMHeuristicsHandle();
    /** Destructor */
    ~CLGEMMHeuristicsHandle();
    /** Prevent Copy Construct */
    CLGEMMHeuristicsHandle(const CLGEMMHeuristicsHandle &) = delete;
    /** Prevent Copy Assignment */
    CLGEMMHeuristicsHandle &operator=(const CLGEMMHeuristicsHandle &) = delete;
    /** Default Move Constructor */
    CLGEMMHeuristicsHandle(CLGEMMHeuristicsHandle &&) = default;
    /** Default Move Assignment */
    CLGEMMHeuristicsHandle &operator=(CLGEMMHeuristicsHandle &&) = default;
    /** (Re)Load the heuristics from reading a dotmlgo file
     *
     * @param[in] filename Path to the dotmlgo file
     *
     * @return bool Signals if the reload succeeded or failed
     */
    bool reload_from_file(const std::string &filename);
    /** Return a pointer to underlying heuristics for querying purposes
     *
     * @return MLGOHeuristics*
     */
    const mlgo::MLGOHeuristics *get() const;

private:
    std::unique_ptr<mlgo::MLGOHeuristics> _heuristics; /**< Pointer to underlying heuristics */
};

} // namespace arm_compute

#endif // ARM_COMPUTE_RUNTIME_CL_CL_GEMM_HEURISTICS_HANDLE_H