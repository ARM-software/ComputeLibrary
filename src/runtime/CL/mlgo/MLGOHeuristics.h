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
#ifndef SRC_RUNTIME_CL_MLGO_MLGO_HEURISTICS_H
#define SRC_RUNTIME_CL_MLGO_MLGO_HEURISTICS_H

#include "src/runtime/CL/mlgo/Common.h"
#include "src/runtime/CL/mlgo/HeuristicTree.h"

#include <iostream>
#include <map>
#include <string>
#include <utility>
namespace arm_compute
{
namespace mlgo
{
/** Query interface */
struct Query
{
    std::string  ip_target; /**< The name of the IP target */
    DataType     data_type; /**< Data type */
    unsigned int m;         /**< Number of rows for the lhs matrix. Lhs matrix NOT transposed */
    unsigned int n;         /**< Number of columns for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int k;         /**< Number of rows for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int b;         /**< Batch size */
};

bool operator==(const GEMMConfigNative &lhs, const GEMMConfigNative &rhs);
bool operator==(const GEMMConfigReshapedOnlyRHS &lhs, const GEMMConfigReshapedOnlyRHS &rhs);
bool operator==(const GEMMConfigReshaped &lhs, const GEMMConfigReshaped &rhs);

/** MLGOHeuristics for configuring GEMM kernels */
class MLGOHeuristics
{
public:
    /** Constructor */
    MLGOHeuristics();
    /** Default Destructor */
    ~MLGOHeuristics() = default;
    /** Prevent Copy Construct */
    MLGOHeuristics(const MLGOHeuristics &) = delete;
    /** Prevent Copy Assignment */
    MLGOHeuristics &operator=(const MLGOHeuristics &) = delete;
    /** Default Move Constructor */
    MLGOHeuristics(MLGOHeuristics &&) = default;
    /** Default Move Assignment */
    MLGOHeuristics &operator=(MLGOHeuristics &&) = default;
    /** Query the gemm type
     *
     * @param[in] query Query
     *
     * @return std::pair<bool, GEMMType>  signals if the query succeeded or failed
     */
    std::pair<bool, GEMMType> query_gemm_type(const Query &query) const;
    /** Query the gemm configuration for native kernel
     *
     * @param[in] query Query
     *
     * @return std::pair<bool, GEMMConfigNative>   bool signals if the query succeeded or failed
     */
    std::pair<bool, GEMMConfigNative> query_gemm_config_native(const Query &query) const;
    /** Query the gemm configuration for reshaped only rhs kernel
     *
     * @param[in] query Query
     *
     * @return std::pair<bool, GEMMConfigReshapedOnlyRHS>   bool signals if the query succeeded or failed
     */
    std::pair<bool, GEMMConfigReshapedOnlyRHS> query_gemm_config_reshaped_only_rhs(const Query &query) const;
    /** Query the gemm configuration for reshaped kernel
     *
     * @param[in] query Query
     *
     * @return std::pair<bool, GEMMConfigReshaped>   bool signals if the query succeeded or failed
     */
    std::pair<bool, GEMMConfigReshaped> query_gemm_config_reshaped(const Query &query) const;
    /** (Re)Load the heuristics from reading a dotmlgo file
     *
     * @param[in] filename Path to the dotmlgo file
     *
     * @return bool Signals if the reload succeeded or failed
     */
    bool reload_from_file(const std::string &filename);
    /** (Re)Load the heuristics from reading an input stream
     *
     * @param[in] istream Istream containing mlgo heuristics
     *
     * @return bool Signals if the reload succeeded or failed
     */
    bool reload_from_stream(std::istream &istream);

    /** Get the heuristic tree from tree id
     *
     * @param[in] id Tree id.
     *
     * @return HeuristicTree&
     */
    std::pair<bool, HeuristicTree *> get_heuristic_tree(HeuristicTree::TreeID id);
    /** Add a heuristic tree
     * @param t Heuristic tree to be added
     */
    bool add_heuristic_tree(HeuristicTree &&t);

    /** Check the validity of the heuristic tree.
     *
     * @param id ID of the tree to be checked
     *
     * @return bool
     */
    bool check_heuristic_tree(HeuristicTree::TreeID id);

    /** Check the overall validity of the heuristics.
     * @return bool
     */
    bool check_all() const;

private:
    static constexpr size_t _max_num_trees{ 100 }; /**< Max number of trees that can be added*/

private:
    // There exists a one-to-one mappipng between TreeID and Index, either can be used to identify a @ref HeuristicTree
    std::map<HeuristicTree::TreeID, HeuristicTree::Index> _indices;    /**< A mapping from TreeID to Index */
    std::map<HeuristicTree::Index, HeuristicTree>         _trees;      /**< A mapping from Index to HeuristicTree */
    std::map<HeuristicTree::TreeID, bool>                 _tree_valid; /**< Result cache of the tree validity checks */
    bool _valid;                                                       /**< Overall validity */
};

} // namespace mlgo
} // namespace arm_compute
#endif //SRC_RUNTIME_CL_MLGO_MLGO_HEURISTICS_H