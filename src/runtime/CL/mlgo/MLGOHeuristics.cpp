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
#include "src/runtime/CL/mlgo/MLGOHeuristics.h"

#include "arm_compute/core/Log.h"

#include "src/runtime/CL/mlgo/MLGOParser.h"
#include "src/runtime/CL/mlgo/Utils.h"

#include <fstream>

namespace arm_compute
{
namespace mlgo
{
bool operator==(const GEMMConfigNative &lhs, const GEMMConfigNative &rhs)
{
    return std::tie(lhs.m0, lhs.n0, lhs.k0) == std::tie(rhs.m0, rhs.n0, rhs.k0);
}
bool operator==(const GEMMConfigReshapedOnlyRHS &lhs, const GEMMConfigReshapedOnlyRHS &rhs)
{
    return std::tie(lhs.m0, lhs.n0, lhs.k0, lhs.h0, lhs.interleave_rhs, lhs.transpose_rhs, lhs.export_cl_image) ==
           std::tie(rhs.m0, rhs.n0, rhs.k0, rhs.h0, rhs.interleave_rhs, rhs.transpose_rhs, rhs.export_cl_image);
}
bool operator==(const GEMMConfigReshaped &lhs, const GEMMConfigReshaped &rhs)
{
    return std::tie(lhs.m0, lhs.n0, lhs.k0, lhs.v0, lhs.h0, lhs.interleave_lhs, lhs.interleave_rhs, lhs.transpose_rhs,
                    lhs.export_cl_image) == std::tie(rhs.m0, rhs.n0, rhs.k0, rhs.v0, rhs.h0, rhs.interleave_lhs,
                                                     rhs.interleave_rhs, rhs.transpose_rhs, rhs.export_cl_image);
}

constexpr size_t MLGOHeuristics::_max_num_trees;

MLGOHeuristics::MLGOHeuristics() : _indices{}, _trees{}, _tree_valid{}, _valid{false}
{
}

std::pair<bool, GEMMType> MLGOHeuristics::query_gemm_type(const Query &query) const
{
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics querying gemm type. %s.", to_string(query).c_str());
    const auto invalid = GEMMType::RESHAPED;
    if (!_valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Invalid DotMLGO. Use default heuristics instead");
        return {false, invalid};
    }
    auto      index = std::make_tuple(HeuristicType::GEMM_Type, query.ip_target, query.data_type);
    GEMMShape shape_query{query.m, query.n, query.k, query.b};
    if (_trees.find(index) == _trees.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Cannot find tree index");
        return {false, invalid};
    }
    return _trees.at(index).query<GEMMType>(shape_query);
}
std::pair<bool, GEMMConfigNative> MLGOHeuristics::query_gemm_config_native(const Query &query) const
{
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics querying gemm config native. %s.",
                                              to_string(query).c_str());
    const auto invalid = GEMMConfigNative{};
    if (!_valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Invalid DotMLGO. Use default heuristics instead");
        return {false, invalid};
    }
    auto      index = std::make_tuple(HeuristicType::GEMM_Config_Native, query.ip_target, query.data_type);
    GEMMShape shape_query{query.m, query.n, query.k, query.b};
    if (_trees.find(index) == _trees.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Cannot find tree index");
        return {false, invalid};
    }
    return _trees.at(index).query<GEMMConfigNative>(shape_query);
}
std::pair<bool, GEMMConfigReshapedOnlyRHS> MLGOHeuristics::query_gemm_config_reshaped_only_rhs(const Query &query) const
{
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics querying gemm config reshaped only rhs. %s.",
                                              to_string(query).c_str());
    const auto invalid = GEMMConfigReshapedOnlyRHS{};
    if (!_valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Invalid DotMLGO. Use default heuristics instead");
        return {false, invalid};
    }
    auto      index = std::make_tuple(HeuristicType::GEMM_Config_Reshaped_Only_RHS, query.ip_target, query.data_type);
    GEMMShape shape_query{query.m, query.n, query.k, query.b};
    if (_trees.find(index) == _trees.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Cannot find tree index");
        return {false, invalid};
    }
    return _trees.at(index).query<GEMMConfigReshapedOnlyRHS>(shape_query);
}
std::pair<bool, GEMMConfigReshaped> MLGOHeuristics::query_gemm_config_reshaped(const Query &query) const
{
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics querying gemm config reshaped. %s.",
                                              to_string(query).c_str());
    const auto invalid = GEMMConfigReshaped{};
    if (!_valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Invalid DotMLGO. Use default heuristics instead");
        return {false, invalid};
    }
    auto      index = std::make_tuple(HeuristicType::GEMM_Config_Reshaped, query.ip_target, query.data_type);
    GEMMShape shape_query{query.m, query.n, query.k, query.b};
    if (_trees.find(index) == _trees.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Cannot find tree index");
        return {false, invalid};
    }
    return _trees.at(index).query<GEMMConfigReshaped>(shape_query);
}

bool MLGOHeuristics::check_heuristic_tree(HeuristicTree::TreeID id)
{
    bool           status;
    HeuristicTree *tree{nullptr};
    std::tie(status, tree) = get_heuristic_tree(id);
    if (!status)
    {
        return status;
    }
    status = tree->check();
    if (!status)
    {
        return status;
    }
    _tree_valid[id] = true;
    return true;
}

bool MLGOHeuristics::check_all() const
{
    // Tree validities are already checked and cached.
    bool all_trees_are_checked =
        std::find_if(_tree_valid.begin(), _tree_valid.end(), [](auto v) { return !v.second; }) == _tree_valid.end();
    if (!all_trees_are_checked)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Missing checks on some trees. Make sure to call check_heuristic_tree after each "
                                      "tree is completed. This could also indicate there are no trees in the dotmlgo");
        return false;
    }

    // Other top level checks...

    return true;
}

std::pair<bool, HeuristicTree *> MLGOHeuristics::get_heuristic_tree(HeuristicTree::TreeID id)
{
    if (_indices.find(id) == _indices.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Cannot find tree with id %zu", id);
        return std::make_pair(false, nullptr);
    }
    const auto index = _indices[id];

    if (_trees.find(index) == _trees.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Cannot find tree index");
        return std::make_pair(false, nullptr);
    }
    auto &t = _trees[index];

    return std::make_pair(true, &t);
}

bool MLGOHeuristics::add_heuristic_tree(HeuristicTree &&t)
{
    if (_indices.size() >= _max_num_trees)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Exceeding the max number of trees allowed: %zu", _max_num_trees);
        return false;
    }
    // PRE: correctness of t is guaranteed by the tree construction process
    // Ensure unique id
    const auto id = t.id();
    if (_indices.find(id) != _indices.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Cannot add redundant trees; tree id %zu already exists", id);
        return false;
    }

    // Ensure unique index
    const auto index = t.index();
    if (_trees.find(index) != _trees.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Cannot add redundant trees; tree index already exists");
        return false;
    }

    _indices[id]    = index;
    _trees[index]   = std::move(t);
    _tree_valid[id] = false;
    return true;
}

bool MLGOHeuristics::reload_from_file(const std::string &filename)
{
    std::ifstream fs;
    fs.exceptions(std::ifstream::badbit);
    fs.open(filename, std::ios::in);
    if (!fs.is_open())
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Cannot open DotMLGO file %s. Use default heuristics instead",
                                                  filename.c_str());
        return _valid = false;
    }
    return reload_from_stream(fs);
}

bool MLGOHeuristics::reload_from_stream(std::istream &in)
{
    auto parsed = parser::parse_mlgo(in);
    if (!parsed.first)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("DotMLGO parsing failed. Use default heuristics instead");
        return _valid = false;
    }
    *this = std::move(parsed.second);
    ARM_COMPUTE_LOG_INFO_MSG_CORE("DotMLGO loaded successfully");
    return _valid = true;
}

} // namespace mlgo
} // namespace arm_compute
