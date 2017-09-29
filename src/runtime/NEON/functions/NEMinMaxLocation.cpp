/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEMinMaxLocation.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute;

NEMinMaxLocation::NEMinMaxLocation()
    : _min_max(), _min_max_loc()
{
}

void NEMinMaxLocation::configure(const IImage *input, void *min, void *max, ICoordinates2DArray *min_loc, ICoordinates2DArray *max_loc, uint32_t *min_count, uint32_t *max_count)
{
    _min_max.configure(input, min, max);
    _min_max_loc.configure(input, min, max, min_loc, max_loc, min_count, max_count);
}

void NEMinMaxLocation::run()
{
    _min_max.reset();

    /* Run min max kernel */
    NEScheduler::get().schedule(&_min_max, Window::DimY);

    /* Run min max location */
    NEScheduler::get().schedule(&_min_max_loc, Window::DimY);
}
