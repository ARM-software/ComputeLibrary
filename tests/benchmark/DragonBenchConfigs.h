/*
 * Copyright (c) 2018 ARM Limited.
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
//FIXME / INTERNAL_ONLY: This file should not be released!
#ifndef __ARM_COMPUTE_DRAGONBENCH_CONFIGS_H__
#define __ARM_COMPUTE_DRAGONBENCH_CONFIGS_H__

#include <string>
#include <vector>

#include "dragonbench/conv2d/conv2d.hpp"
#include "dragonbench/fully_connected/fully_connected.hpp"

namespace arm_compute
{
// Stream operators
inline ::std::ostream &operator<<(::std::ostream &os, const Conv2D &conv2d_config)
{
    os << "network_name=" << conv2d_config.network_name << ":";
    os << "layer_name=" << conv2d_config.layer_name << ":";
    os << "id=" << conv2d_config.id << ":";
    os << "Input_NCHW="
       << conv2d_config.ibatch << ','
       << conv2d_config.ch_in << ','
       << conv2d_config.dim_in_h << ','
       << conv2d_config.dim_in_w << ":";
    os << "Output_NCHW="
       << conv2d_config.ibatch << ','
       << conv2d_config.ch_out << ','
       << conv2d_config.dim_out_h << ','
       << conv2d_config.dim_out_w << ":";
    os << "Weights_HW="
       << conv2d_config.kern_h << ','
       << conv2d_config.kern_w << ":";
    os << "Stride_HW="
       << conv2d_config.stride_h << ','
       << conv2d_config.stride_w << ":";
    os << "Padding=" << conv2d_config.padding << ":";
    return os;
}
inline std::string to_string(const Conv2D &conv2d_config)
{
    std::stringstream str;
    str << conv2d_config;
    return str.str();
}
inline ::std::ostream &operator<<(::std::ostream &os, const Fully_Connected &fc_config)
{
    os << "network_name=" << fc_config.network_name << ":";
    os << "layer_name=" << fc_config.layer_name << ":";
    os << "id=" << fc_config.id << ":";
    os << "M=" << fc_config.m << ":";
    os << "N=" << fc_config.n << ":";
    os << "K=" << fc_config.k << ":";
    return os;
}
inline std::string to_string(const Fully_Connected &fc_config)
{
    std::stringstream str;
    str << fc_config;
    return str.str();
}

namespace test
{
namespace benchmark
{
// Conv 2D benchmarks
extern conv2d_configs silverwing_cfgs;
extern conv2d_configs sunfyre_cfgs;
extern conv2d_configs syrax_cfgs;

// Fully Connected benchmarks
extern fully_connected_configs dreamfyre_cfgs;
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_DRAGONBENCH_CONFIGS_H__ */
