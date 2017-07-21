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
#ifndef __ARM_COMPUTE_TEST_TENSOR_VISITORS_H__
#define __ARM_COMPUTE_TEST_TENSOR_VISITORS_H__

#include "Tensor.h"
#include "TensorOperations.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/Lut.h"

#include "boost_wrapper.h"

#include <algorithm>
#include <map>
#include <memory>
#include <ostream>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace tensor_visitors
{
// Min max location visitor
struct min_max_location_visitor : public boost::static_visitor<>
{
public:
    explicit min_max_location_visitor(void *min, void *max, IArray<Coordinates2D> &min_loc, IArray<Coordinates2D> &max_loc, uint32_t &min_count, uint32_t &max_count)
        : _min(min), _max(max), _min_loc(min_loc), _max_loc(max_loc), _min_count(min_count), _max_count(max_count)
    {
    }
    template <typename T1>
    void operator()(const Tensor<T1> &in) const
    {
        tensor_operations::min_max_location(in, _min, _max, _min_loc, _max_loc, _min_count, _max_count);
    }

private:
    void                  *_min;
    void                  *_max;
    IArray<Coordinates2D> &_min_loc;
    IArray<Coordinates2D> &_max_loc;
    uint32_t              &_min_count;
    uint32_t              &_max_count;
};
// Absolute Difference visitor
struct absolute_difference_visitor : public boost::static_visitor<>
{
public:
    template <typename T1, typename T2, typename T3>
    void operator()(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out) const
    {
        tensor_operations::absolute_difference(in1, in2, out);
    }
};
// Arithmetic Addition visitor
struct arithmetic_addition_visitor : public boost::static_visitor<>
{
public:
    explicit arithmetic_addition_visitor(ConvertPolicy convert_policy)
        : _policy(convert_policy)
    {
    }

    template <typename T1, typename T2, typename T3>
    void operator()(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out) const
    {
        tensor_operations::arithmetic_addition(in1, in2, out, _policy);
    }

private:
    ConvertPolicy _policy;
};
// Arithmetic Subtraction visitor
struct arithmetic_subtraction_visitor : public boost::static_visitor<>
{
public:
    explicit arithmetic_subtraction_visitor(ConvertPolicy convert_policy)
        : _policy(convert_policy)
    {
    }

    template <typename T1, typename T2, typename T3>
    void operator()(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out) const
    {
        tensor_operations::arithmetic_subtraction(in1, in2, out, _policy);
    }

private:
    ConvertPolicy _policy;
};
// Depth Convert visitor
struct depth_convert_visitor : public boost::static_visitor<>
{
public:
    explicit depth_convert_visitor(ConvertPolicy policy, uint32_t shift)
        : _policy(policy), _shift(shift)
    {
    }

    template <typename T1, typename T2>
    void operator()(const Tensor<T1> &in, Tensor<T2> &out) const
    {
        tensor_operations::depth_convert(in, out, _policy, _shift);
    }

private:
    ConvertPolicy _policy;
    uint32_t      _shift;
};
// GEMM visitor
struct gemm_visitor : public boost::static_visitor<>
{
public:
    explicit gemm_visitor(const TensorVariant &in1, const TensorVariant &in2, const TensorVariant &in3, float alpha, float beta)
        : _in1(in1), _in2(in2), _in3(in3), _alpha(alpha), _beta(beta)
    {
    }

    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in1 = boost::get<Tensor<T>>(_in1);
        const Tensor<T> &in2 = boost::get<Tensor<T>>(_in2);
        const Tensor<T> &in3 = boost::get<Tensor<T>>(_in3);
        tensor_operations::gemm(in1, in2, in3, out, _alpha, _beta);
    }

private:
    const TensorVariant &_in1, &_in2, &_in3;
    float                _alpha;
    float                _beta;
};
// Pixel-wise Multiplication visitor
struct pixel_wise_multiplication_visitor : public boost::static_visitor<>
{
public:
    explicit pixel_wise_multiplication_visitor(float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
        : _scale(scale), _convert_policy(convert_policy), _rounding_policy(rounding_policy)
    {
    }

    template <typename T1, typename T2, typename T3>
    void operator()(const Tensor<T1> &in1, const Tensor<T2> &in2, Tensor<T3> &out) const
    {
        tensor_operations::pixel_wise_multiplication(in1, in2, out, _scale, _convert_policy, _rounding_policy);
    }

private:
    float          _scale;
    ConvertPolicy  _convert_policy;
    RoundingPolicy _rounding_policy;
};
// Fixed Point Pixel-wise Multiplication visitor
struct fixed_point_pixel_wise_multiplication_visitor : public boost::static_visitor<>
{
public:
    explicit fixed_point_pixel_wise_multiplication_visitor(const TensorVariant &in1, const TensorVariant &in2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
        : _in1(in1), _in2(in2), _scale(scale), _convert_policy(convert_policy), _rounding_policy(rounding_policy)
    {
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in1 = boost::get<Tensor<T>>(_in1);
        const Tensor<T> &in2 = boost::get<Tensor<T>>(_in2);
        tensor_operations::fixed_point_pixel_wise_multiplication(in1, in2, out, _scale, _convert_policy, _rounding_policy);
    }
    template < typename T, typename std::enable_if < !std::is_integral<T>::value, int >::type = 0 >
    void operator()(Tensor<T> &out) const
    {
        ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

private:
    const TensorVariant &_in1;
    const TensorVariant &_in2;
    float                _scale;
    ConvertPolicy        _convert_policy;
    RoundingPolicy       _rounding_policy;
};
// Table lookup operation
template <typename T1>
struct table_lookup : public boost::static_visitor<>
{
public:
    explicit table_lookup(const TensorVariant &in, std::map<T1, T1> &lut)
        : _in(in), _lut(lut)
    {
    }

    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        const auto &in = boost::get<Tensor<T>>(_in);
        tensor_operations::table_lookup(in, out, _lut);
    }

private:
    const TensorVariant &_in;
    std::map<T1, T1> &_lut;
};
template struct arm_compute::test::validation::tensor_visitors::table_lookup<uint8_t>;
template struct arm_compute::test::validation::tensor_visitors::table_lookup<int16_t>;

// Batch Normalization Layer visitor
struct batch_normalization_layer_visitor : public boost::static_visitor<>
{
public:
    explicit batch_normalization_layer_visitor(const TensorVariant &in, const TensorVariant &mean, const TensorVariant &var, const TensorVariant &beta, const TensorVariant &gamma, float epsilon,
                                               int fixed_point_position = 0)
        : _in(in), _mean(mean), _var(var), _beta(beta), _gamma(gamma), _epsilon(epsilon), _fixed_point_position(fixed_point_position)
    {
    }

    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in    = boost::get<Tensor<T>>(_in);
        const Tensor<T> &mean  = boost::get<Tensor<T>>(_mean);
        const Tensor<T> &var   = boost::get<Tensor<T>>(_var);
        const Tensor<T> &beta  = boost::get<Tensor<T>>(_beta);
        const Tensor<T> &gamma = boost::get<Tensor<T>>(_gamma);
        tensor_operations::batch_normalization_layer(in, out, mean, var, beta, gamma, _epsilon, _fixed_point_position);
    }

private:
    const TensorVariant &_in, &_mean, &_var, &_beta, &_gamma;
    float                _epsilon;
    int                  _fixed_point_position;
};
// Convolution Layer visitor
struct convolution_layer_visitor : public boost::static_visitor<>
{
public:
    explicit convolution_layer_visitor(const TensorVariant &in, const TensorVariant &weights, const TensorVariant &bias, PadStrideInfo conv_info)
        : _in(in), _weights(weights), _bias(bias), _conv_info(conv_info)
    {
    }

    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in      = boost::get<Tensor<T>>(_in);
        const Tensor<T> &weights = boost::get<Tensor<T>>(_weights);
        const Tensor<T> &bias    = boost::get<Tensor<T>>(_bias);
        tensor_operations::convolution_layer(in, weights, bias, out, _conv_info);
    }

private:
    const TensorVariant &_in;
    const TensorVariant &_weights;
    const TensorVariant &_bias;
    PadStrideInfo        _conv_info;
};
// Depth Concatenate Layer visitor
struct depth_concatenate_layer_visitor : public boost::static_visitor<>
{
public:
    explicit depth_concatenate_layer_visitor(const std::vector<TensorVariant> &srcs)
        : _srcs(srcs)
    {
    }

    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        std::vector<const Tensor<T> *> srcs;
        srcs.resize(_srcs.size());
        std::transform(_srcs.begin(), _srcs.end(), srcs.begin(), [](const TensorVariant & t)
        {
            return &(boost::get<Tensor<T>>(t));
        });
        tensor_operations::depth_concatenate_layer(srcs, out);
    }

private:
    const std::vector<TensorVariant> &_srcs;
};
// Fully Connected Layer visitor
struct fully_connected_layer_visitor : public boost::static_visitor<>
{
public:
    explicit fully_connected_layer_visitor(const TensorVariant &in, const TensorVariant &weights, const TensorVariant &bias)
        : _in(in), _weights(weights), _bias(bias)
    {
    }
    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in      = boost::get<Tensor<T>>(_in);
        const Tensor<T> &weights = boost::get<Tensor<T>>(_weights);
        const Tensor<T> &bias    = boost::get<Tensor<T>>(_bias);
        tensor_operations::fully_connected_layer(in, weights, bias, out);
    }

private:
    const TensorVariant &_in;
    const TensorVariant &_weights;
    const TensorVariant &_bias;
};

// Pooling layer
struct pooling_layer_visitor : public boost::static_visitor<>
{
public:
    explicit pooling_layer_visitor(const TensorVariant &in, PoolingLayerInfo pool_info, int fixed_point_position = 0)
        : _in(in), _pool_info(pool_info), _fixed_point_position(fixed_point_position)
    {
    }

    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in = boost::get<Tensor<T>>(_in);
        tensor_operations::pooling_layer(in, out, _pool_info, _fixed_point_position);
    }

private:
    const TensorVariant &_in;
    PoolingLayerInfo     _pool_info;
    int                  _fixed_point_position;
};

// ROI Pooling layer
struct roi_pooling_layer_visitor : public boost::static_visitor<>
{
public:
    explicit roi_pooling_layer_visitor(const TensorVariant &in, const std::vector<ROI> &rois, ROIPoolingLayerInfo pool_info)
        : _in(in), _rois(rois), _pool_info(pool_info)
    {
    }

    template <typename T>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in = boost::get<Tensor<T>>(_in);
        tensor_operations::roi_pooling_layer(in, out, _rois, _pool_info);
    }

private:
    const TensorVariant    &_in;
    const std::vector<ROI> &_rois;
    ROIPoolingLayerInfo     _pool_info;
};

// Fixed Point operations visitor
struct fixed_point_operation_visitor : public boost::static_visitor<>
{
public:
    explicit fixed_point_operation_visitor(const TensorVariant &in, FixedPointOp op)
        : _in(in), _op(op)
    {
    }

    template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
    void operator()(Tensor<T> &out) const
    {
        const Tensor<T> &in = boost::get<Tensor<T>>(_in);
        tensor_operations::fixed_point_operation(in, out, _op);
    }
    template < typename T, typename std::enable_if < !std::is_integral<T>::value, int >::type = 0 >
    void operator()(Tensor<T> &out) const
    {
        ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

private:
    const TensorVariant &_in;
    FixedPointOp         _op;
};
// Print Tensor visitor
struct print_visitor : public boost::static_visitor<>
{
public:
    explicit print_visitor(std::ostream &out)
        : _out(out)
    {
    }

    template <typename T>
    void operator()(const Tensor<T> &in) const
    {
        tensor_operations::print(in, _out);
    }

private:
    std::ostream &_out;
};
} // namespace tensor_visitors
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* __ARM_COMPUTE_TEST_TENSOR_VISITORS_H__ */
