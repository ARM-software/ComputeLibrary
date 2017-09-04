/*
 * Copyright (c) 2017 ARM Limited.
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
#include "UserConfiguration.h"

#include "ProgramOptions.h"

#include <string>

namespace arm_compute
{
namespace test
{
UserConfiguration::UserConfiguration(const ProgramOptions &options)
{
    std::random_device::result_type tmp_seed = 0;
    if(options.get("seed", tmp_seed))
    {
        seed = tmp_seed;
    }

    std::string tmp_path;
    if(options.get("path", tmp_path))
    {
        path = tmp_path;
    }

    unsigned int tmp_threads = 0;
    if(options.get("threads", tmp_threads))
    {
        threads = tmp_threads;
    }
}
} // namespace test
} // namespace arm_compute
