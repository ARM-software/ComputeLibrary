/*
 * Copyright (c) 2019, 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_IRUNTIME_CONTEXT_H
#define ARM_COMPUTE_IRUNTIME_CONTEXT_H

namespace arm_compute
{
// Forward declarations
class IScheduler;
class IAssetManager;

/** Context interface */
class IRuntimeContext
{
public:
    /** Destructor */
    virtual ~IRuntimeContext() = default;
    /** Scheduler accessor
      *
      * @note Scheduler is used to schedule workloads
      *
      * @return The scheduler registered to the context
      */
    virtual IScheduler *scheduler() = 0;
    /** Asset manager accessor
     *
     * @note Asset manager is used to manage objects/tensors within functions
     *
     * @return The asset manager registered to the context
     */
    virtual IAssetManager *asset_manager() = 0;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_IRUNTIME_CONTEXT_H */
