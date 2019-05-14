/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/CL/tuners/CLLWSList.h"

namespace arm_compute
{
namespace cl_tuner
{
size_t CLLWSList::size()
{
    return search_space_shape.total_size();
}

cl::NDRange CLLWSListExhaustive::operator[](size_t index)
{
    ARM_COMPUTE_ERROR_ON(index >= size());
    auto coords = index2coords(search_space_shape, index);
    return cl::NDRange{ coords[0] + 1U, coords[1] + 1U, coords[2] + 1U };
}

CLLWSListExhaustive::CLLWSListExhaustive(const cl::NDRange &gws)
{
    ARM_COMPUTE_UNUSED(gws);
    search_space_shape = TensorShape(max_lws_supported_x,
                                     max_lws_supported_y,
                                     max_lws_supported_z);
}

cl::NDRange CLLWSListNormal::operator[](size_t index)
{
    ARM_COMPUTE_ERROR_ON(index >= size());
    auto coords = index2coords(search_space_shape, index);
    return cl::NDRange{ _lws_x[coords[0]], _lws_y[coords[1]], _lws_z[coords[2]] };
}

CLLWSListNormal::CLLWSListNormal(const cl::NDRange &gws)
{
    auto lws_x_max = std::min(static_cast<unsigned int>(gws[0]), max_lws_supported_x);
    auto lws_y_max = std::min(static_cast<unsigned int>(gws[1]), max_lws_supported_y);
    auto lws_z_max = std::min(static_cast<unsigned int>(gws[2]), max_lws_supported_z);

    // Initialize the LWS values to test
    initialize_lws_values(_lws_x, gws[0], lws_x_max, gws[2] > 16); // Explore lws that are not factors of gws only when gws[2] > 16
    initialize_lws_values(_lws_y, gws[1], lws_y_max, gws[2] > 16); // Explore lws that are not factors of gws only when gws[2] > 16
    initialize_lws_values(_lws_z, gws[2], lws_z_max, false);

    search_space_shape = TensorShape(_lws_x.size(), _lws_y.size(), _lws_z.size());
}

void CLLWSListNormal::initialize_lws_values(std::vector<unsigned int> &lws, unsigned int gws, unsigned int lws_max, bool mod_let_one)
{
    lws.push_back(1);

    for(unsigned int i = 2; i <= lws_max; ++i)
    {
        // Power of two condition
        const bool is_power_of_two = (i & (i - 1)) == 0;

        // Condition for the module accordingly with the mod_let_one flag
        const bool mod_cond = mod_let_one ? (gws % i) <= 1 : (gws % i) == 0;

        if(mod_cond || is_power_of_two)
        {
            lws.push_back(i);
        }
    }
}

CLLWSListRapid::CLLWSListRapid(const cl::NDRange &gws)
{
    auto lws_x_max = std::min(static_cast<unsigned int>(gws[0]), 8u); // Limit exploration to 1 - 8
    auto lws_y_max = std::min(static_cast<unsigned int>(gws[1]), 4u); // Limit exploration to 1 - 4
    auto lws_z_max = std::min(static_cast<unsigned int>(gws[2]), 4u); // Limit exploration to 1 - 4

    // Initialize the LWS values to test
    initialize_lws_values(_lws_x, lws_x_max);
    initialize_lws_values(_lws_y, lws_y_max);
    initialize_lws_values(_lws_z, lws_z_max);

    search_space_shape = TensorShape(_lws_x.size(), _lws_y.size(), _lws_z.size());
}

void CLLWSListRapid::initialize_lws_values(std::vector<unsigned int> &lws, unsigned int lws_max)
{
    lws.push_back(1);

    for(unsigned int i = 2; i <= lws_max; i *= 4)
    {
        lws.push_back(i);
    }
}
} // namespace cl_tuner
} // namespace arm_compute
