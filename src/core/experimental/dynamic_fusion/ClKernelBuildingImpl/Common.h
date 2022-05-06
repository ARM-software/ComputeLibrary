/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#error "This experimental feature must be enabled with -DENABLE_EXPERIMENTAL_DYNAMIC_FUSION"
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */

#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMMON_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMMON_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "src/core/common/Macros.h"
#include "support/Requires.h"
#include "support/StringSupport.h"

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"

#include <iostream>
#include <queue>
#include <stack>
#include <string>
#include <unordered_set>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** We introduce the concept of *Shared Variables* in the context of kernel building.
 *  They are variables that can be accessed / shared among all the kernel components within a single kernel.
 *  For now we consider 2 groups of shared variables:
 *      Argument: The argument variables (parameters) of a kernel
 *      Automatic: The automatic variables declared inside a kernel
 *  All Shared Variables have the same kernel scope, and are thus visible to all kernel components
*/

enum class SharedVarIO
{
    Input,
    Output
};

enum class SharedVarGroup
{
    Argument, // Parameters to a kernel function  == dst or src tensors of the whole blueprint graph
    Automatic // Automatic variables declared within the kernel body == intermediate tensors of the whole blueprint graph
};

/** Specifies a shared variable link for a component.
 * It describes all the information that's available when a component is constructed / added:
 *  e.g. its linkage (via ArgumentID and io) and its group
 * This is not shared variable on its own, but is used for instantiating a SharedVar when building the code
 */
struct SharedVarLink
{
    ArgumentID  arg_id{ g_arg_placeholder };
    SharedVarIO io{ SharedVarIO::Input };
    bool        is_empty() const
    {
        return arg_id == g_arg_placeholder;
    }
};

/** A table of all the variables used in the kernel / blueprint
 * Because we limit the DependencyGraph in the blueprint to a Linear Sequence for now, we only allow ** a single global variable (the accumulator) **
 *
 * NOTE: the order they appear in the table is the order of their "declaration" in the component code, and is also their ID
 * NOTE: the variables all have the scope of the full kernel function
 */
class SharedVarTable
{
public:
    /** A fully realized SharedVarLink
     */
    struct SharedVar
    {
        ArgumentID            arg_id{ g_arg_placeholder };
        SharedVarIO           io{ SharedVarIO::Input };
        SharedVarGroup        group{ SharedVarGroup::Argument };
        std::string           uniq_name{}; // Unique name, also the final variable name used in the built code
        ClKernelArgDescriptor desc{};      // Automatic variables can and should still be described using this struct
        bool                  is_empty() const
        {
            return arg_id == g_arg_placeholder;
        }
    };

    class Arguments
    {
    public:
        Arguments() = default;
        void add_var(const SharedVar &var)
        {
            ARM_COMPUTE_ERROR_ON(var.group != SharedVarGroup::Argument);
            _vars.push_back(var);
        }
        std::vector<SharedVar> get_all_vars() const
        {
            return _vars;
        }
        std::vector<SharedVar> get_src_vars() const
        {
            std::vector<SharedVar> src_vars;
            std::copy_if(_vars.begin(), _vars.end(), std::back_inserter(src_vars), [](const SharedVar & var)
            {
                return var.io == SharedVarIO::Input;
            });
            return src_vars;
        }
        SharedVar get_dst_var() const
        {
            std::vector<SharedVar> dst_vars;
            std::copy_if(_vars.begin(), _vars.end(), std::back_inserter(dst_vars), [](const SharedVar & var)
            {
                return var.io == SharedVarIO::Output;
            });
            ARM_COMPUTE_ERROR_ON(dst_vars.size() != 1);
            return dst_vars.at(0);
        }

    private:
        std::vector<SharedVar> _vars{};
    };

    /** Create a SharedVar for a corresponding SharedVarLink (contains ArgumentID). If one has already been created for the SharedVarLink, simply return it instead of creating a new one
     *
     * @note: The order of insertion is important. There is one precondition:
     *        PRECOND: The components have been sorted topologically / is being traversed in topological order
     *                 This ensures that all the consumer var links (Output, Automatic Links) can consume (return) the producer var links when they're referred
     */
    void add(SharedVarLink var_link, SharedVarGroup group, ClKernelArgDescriptor runtime_desc, const std::string &name = "unnamed")
    {
        ARM_COMPUTE_ERROR_ON_MSG(var_link.is_empty(), "Non-empty SharedVarLink expected");
        if(!get(var_link).is_empty())
        {
            return;
        }

        auto              var_id = _num_var;
        std::stringstream ss;
        ss << name << "_" << var_id;
        const auto uniq_name = ss.str();
        SharedVar  var{ var_link.arg_id, var_link.io, group, uniq_name, runtime_desc };

        if(group == SharedVarGroup::Argument)
        {
            _arguments.emplace(var_id, var);
            _arg_id_map.emplace(var_link.arg_id, var_id);
            _num_var++;
        }
        else if(group == SharedVarGroup::Automatic)
        {
            if(_global_vars.empty())
            {
                if(var_link.io == SharedVarIO::Output)
                {
                    _global_vars.emplace(var_id, var);
                    _arg_id_map.emplace(var_link.arg_id, var_id);
                    _num_var++;
                }
                else
                {
                    ARM_COMPUTE_ERROR("Component likely not traversed in topological order");
                }
            }
            else
            {
                // Associate additional SharedVarLinks with the single global shared variable
                const auto global_var_id     = _global_vars.begin()->first;
                _arg_id_map[var_link.arg_id] = global_var_id;
            }
        }
        else
        {
            ARM_COMPUTE_ERROR("Unrecognised SharedVarGroup");
        }
    }

    /** Get the SharedVar associated with @p var_link
     *
     * @param var_link
     * @return SharedVar
     */
    SharedVar get(const SharedVarLink &var_link) const
    {
        const SharedVar empty_var{};
        if(_arg_id_map.find(var_link.arg_id) != _arg_id_map.end())
        {
            const auto var_id  = _arg_id_map.at(var_link.arg_id);
            const auto arg_var = _arguments.find(var_id);
            if(arg_var != _arguments.end())
            {
                return arg_var->second;
            }
            else
            {
                return _global_vars.at(var_id);
            }
        }
        return empty_var;
    }

    /** @note The arguments are returned in the order they are added
     */
    Arguments get_kernel_arguments() const
    {
        Arguments args{};
        for(const auto &a : _arguments)
        {
            args.add_var(a.second);
        }
        return args;
    }

private:
    using VarID = int32_t;

private:
    std::map<VarID, SharedVar>  _global_vars{}; // Shared, global variable
    std::map<VarID, SharedVar>  _arguments{};
    std::map<ArgumentID, VarID> _arg_id_map{}; // Track ArgumentIDs that have already been added
    VarID _num_var{ 0 };
};

enum class ComponentType
{
    Simple,
    Complex,
    Store
};

using ComponentID   = DependencyGraph::Id;
using ComponentList = std::vector<ComponentID>;
class IClKernelComponent
{
public:
    using Link = SharedVarLink;
    using Tag  = std::string;
    struct TagVal
    {
        TagVal() = default;
        TagVal(const SharedVarTable::SharedVar &var)
            : value{ var.uniq_name }
        {
        }

        template <typename T, ARM_COMPUTE_REQUIRES_TA(std::is_integral<T>::value)>
        TagVal(T val)
            : value{ support::cpp11::to_string(val) }
        {
        }

        TagVal(const std::string &val)
            : value{ val }
        {
        }

        TagVal(const char *val)
            : value{ std::string(val) }
        {
        }

        TagVal(const DataType &data_type)
            : value{ get_cl_type_from_data_type(data_type) }
        {
        }

        std::string value{};
    };
    using TagLUT = std::unordered_map<Tag, TagVal>; // Used to instantiating a code template / replacing tags
public:
    IClKernelComponent(ClKernelBlueprint *blueprint)
        : _blueprint(blueprint)
    {
    }

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(IClKernelComponent);

    virtual ~IClKernelComponent()                        = default;
    virtual ComponentType     get_component_type() const = 0;
    virtual std::vector<Link> get_links() const          = 0;
    virtual std::string       name() const               = 0;

    // @note: some tags can be unused since they could be used only for the macros, or only for the component code
    static std::string replace_tags(const std::string &code_template, const TagLUT &tags)
    {
        std::string replaced_code    = "";
        bool        scanning_pattern = false;
        std::string pattern_found    = "";
        for(size_t i = 0; i < code_template.size() - 1; ++i)
        {
            if(!scanning_pattern)
            {
                if(code_template[i] == '{' && code_template[i + 1] == '{')
                {
                    i += 1;
                    scanning_pattern = true;
                    pattern_found    = "";
                }
                else
                {
                    replaced_code += code_template[i];
                }
            }
            else
            {
                if(code_template[i] == '}' && code_template[i + 1] == '}')
                {
                    i += 1;
                    scanning_pattern = false;
                    std::string err  = "Pattern " + pattern_found + " not found in tags";
                    ARM_COMPUTE_ERROR_ON_MSG(tags.find(pattern_found) == tags.end(), err.c_str());
                    replaced_code += tags.find(pattern_found)->second.value;
                }
                else
                {
                    pattern_found += code_template[i];
                }
            }
        }

        return replaced_code;
    }
    ComponentID id() const
    {
        return _id;
    }
    void set_id(ComponentID id)
    {
        _id = id;
    }

    virtual std::set<std::string> get_headers_list() const
    {
        return std::set<std::string> {};
    }

    virtual std::string get_additional_macros() const
    {
        return "";
    }

    virtual std::string get_component_code() const
    {
        return "";
    }

    virtual Window get_window() const
    {
        return Window{};
    }
    /** Get the tag look-up table used to instantiate the component code.
     *
     * @param vtable
     * @return TagLUT
     */
    virtual TagLUT get_tag_lut(const SharedVarTable &vtable) const = 0;

    /** Allocate all shared variables used by the component in the @p vtable
     *
     * @param vtable
     */
    virtual void allocate_shared_vars(SharedVarTable &vtable) const = 0;

    virtual std::string get_dst_addr_calculation() const
    {
        return "";
    }

    /** Generate config id of the component
     *
     * @return std::string
     */
    virtual std::string generate_config_id() const
    {
        return "";
    }

    virtual CLBuildOptions generate_build_options() const
    {
        return CLBuildOptions{};
    }

protected:
    ClKernelBlueprint *_blueprint;

private:
    ComponentID _id{};
};

using ComponentUniquePtr = std::unique_ptr<IClKernelComponent>;

/** Intermediate representation of the final, complete kernel source.
 */
struct ClKernelBlueprint::Implementation
{
public:
    Implementation()  = default;
    ~Implementation() = default;

public:
    Status update_merge_point(ArgumentID t_id, ArgumentID merge_point)
    {
        return _graph.update_merge_point(t_id, merge_point);
    }

    ArgumentID add_kernel_tensor(ITensorInfo *tensor_info, ArgumentID merge_point = DependencyGraph::empty_id())
    {
        const auto id = _graph.add_tensor(merge_point);
        if(_kernel_tensors.find(id) == _kernel_tensors.end())
        {
            _kernel_tensors.insert(std::make_pair(id, tensor_info));
        }
        return id;
    }

    void set_tile_info(const TileDescriptor &tile_info)
    {
        _tile_info = tile_info;
    }

    SharedVarGroup group(ArgumentID arg_id) const
    {
        if(arg_id == g_arg_placeholder)
        {
            // In case of placeholder, don't care what we return;
            return SharedVarGroup::Argument;
        }
        return _shared_var_group_lut.at(arg_id);
    }

    void validate_arg_ids(std::initializer_list<ArgumentID> args) const
    {
        for(const auto arg_id : args)
        {
            ARM_COMPUTE_UNUSED(arg_id);
            ARM_COMPUTE_ERROR_ON_MSG(_kernel_tensors.find(arg_id) == _kernel_tensors.end() && arg_id != g_arg_placeholder,
                                     "Trying to use an argument that hasn't been added to the blueprint");
        }
    }

    void add_component(ComponentUniquePtr component)
    {
        if(component->get_component_type() == ComponentType::Complex)
        {
            ++_num_complex_components;
            ARM_COMPUTE_ERROR_ON_MSG(_num_complex_components > 1, "Only one complex component per blueprint is supported.");
        }

        // Get an unique ID for the component that's being added
        std::vector<ArgumentID> src_tensors;
        std::vector<ArgumentID> dst_tensors;
        for(const auto &link : component->get_links())
        {
            if(link.is_empty())
            {
                continue;
            }
            if(link.io == SharedVarIO::Input)
            {
                src_tensors.push_back(link.arg_id);
            }
            else
            {
                dst_tensors.push_back(link.arg_id);
            }
        }
        const ComponentID component_id = _graph.add_operator(src_tensors, dst_tensors).second;
        component->set_id(component_id);

        // Add this component to the component graph. Don't connect it to anything yet
        _component_graph.emplace(component_id, ComponentList{});

        // For every { arg_id, arg_io } passed along with this component...
        for(const auto &link : component->get_links())
        {
            const ArgumentID &arg_id = link.arg_id;
            const SharedVarIO &arg_io = link.io;

            // Add the arg_id to the map describing the input/output relationship between an argument and the components that use it, if it doesn't yet exist there
            if(_outgoing_components.find(arg_id) == _outgoing_components.end())
            {
                _outgoing_components.emplace(arg_id, ComponentList{});
                _incoming_components.emplace(arg_id, ComponentList{});
            }

            // If it's an input argument, connect any other component that has it as output with this component
            // Additionally, set this component as one that treats this argument as "Input" (append to index 0)
            // This is used so that we keep track of whether two components use the same argument, one as input and one as output
            if(arg_io == SharedVarIO::Input)
            {
                for(const auto &prev_component : _incoming_components[arg_id])
                {
                    _component_graph[prev_component].push_back(component_id);
                }

                _outgoing_components[arg_id].push_back(component_id);
            }
            // If it's an output argument, connect this component with any other component that has it as input
            // Additionally, set this component as one that treats this argument as "Output" (append to index 1)
            else
            {
                if(component->get_component_type() == ComponentType::Store)
                {
                    ARM_COMPUTE_ERROR_ON_MSG(_dst_id >= 0, "Trying to add more than one dst argument to the graph");
                    _dst_id = arg_id;
                }

                for(const auto &subseq_component : _outgoing_components[arg_id])
                {
                    _component_graph[component_id].push_back(subseq_component);
                }

                _incoming_components[arg_id].push_back(component_id);
            }
        }

        ARM_COMPUTE_ERROR_ON_MSG(_graph.get_root_ops().size() != 1, "Trying to add more than one root to the graph");

        // Finally, add this component to the dictionary of components
        _components.insert(std::make_pair(component_id, std::move(component)));
    }

    std::string build_kernel_name() const
    {
        std::string name = "";

        traverse([&](std::stack<ComponentID> stack)
        {
            name += _components.find(stack.top())->second->name() + (stack.size() > 2 ? "___" : "");
        });

        return name;
    }

    std::string build_code()
    {
        ARM_COMPUTE_ERROR_ON_MSG(_graph_root < 0, "No root found in the component graph");

        // These data structures will hold the data from all the components in the blueprint
        std::set<std::string>    headers_list{};
        std::set<std::string>    additional_macros{};
        std::vector<std::string> component_codes{}; // vector because order matters

        // Step 1: Allocate all kernel argument shared variables before generating the component code
        auto stack = topological_sort();
        while(!stack.empty())
        {
            auto  curr_component_id = stack.top();
            auto &curr_component    = _components.find(curr_component_id)->second;

            curr_component->allocate_shared_vars(_vtable);

            stack.pop();
        }
        // Step 2: Generate component codes
        stack = topological_sort();
        while(!stack.empty())
        {
            auto  curr_component_id = stack.top();
            auto &curr_component    = _components.find(curr_component_id)->second;

            auto       curr_headers_list      = curr_component->get_headers_list();
            auto       curr_additional_macros = curr_component->get_additional_macros();
            auto       curr_component_code    = curr_component->get_component_code();
            const auto var_lut                = curr_component->get_tag_lut(_vtable); // Ideally can be merged with get_component_code once we have finer-grained code generation technique
            component_codes.push_back(IClKernelComponent::replace_tags(curr_component_code, var_lut));

            headers_list.insert(curr_headers_list.begin(), curr_headers_list.end());
            if(!curr_additional_macros.empty()) // Some components might not have any
            {
                additional_macros.insert(IClKernelComponent::replace_tags(curr_additional_macros, var_lut));
            }

            stack.pop();
        }

        // Step 3: Assemble the data gathered by traversing the graph into the string "code"
        std::string code = "";

        for(auto &header : headers_list)
        {
#if defined(EMBEDDED_KERNELS)
            code += CLKernelLibrary::get().get_program(header).first;
#else  // defined(EMBEDDED_KERNELS)
            code += "#include \"" + header + "\"\n";
#endif // defined(EMBEDDED_KERNELS)
        }

        for(auto &macros : additional_macros)
        {
            code += macros;
        }

        code += generate_kernel_signature(_vtable.get_kernel_arguments());

        code += "\n{\n\n";

        code += "    //------------------ START KERNEL_BUILDER_COORDINATE ---------------------\n\n";
        code += generate_global_section();
        code += "    //------------------ END KERNEL_BUILDER_COORDINATE ---------------------\n";

        for(auto &component_code : component_codes)
        {
            code += component_code;
        }

        code += "}\n";

        return code;
    }

    /** Generate config id of the entire kernel
     *
     * Format: kernel_name--comp0_config_id--comp1_config_id--...
     *
     * @return std::string
     */
    std::string build_config_id() const
    {
        std::string config_id = build_kernel_name();
        traverse([&](std::stack<ComponentID> stack)
        {
            config_id += "--" + _components.find(stack.top())->second->generate_config_id() + "--";
        });

        return config_id;
    }

    CLBuildOptions build_options() const
    {
        CLBuildOptions build_opts{};

        traverse([&](std::stack<ComponentID> stack)
        {
            build_opts.add_options(_components.find(stack.top())->second->generate_build_options().options());
        });

        return build_opts;
    }

    TileDescriptor get_tile_info() const
    {
        return _tile_info;
    }

    Window get_execution_window() const
    {
        ARM_COMPUTE_ERROR_ON_MSG(_graph_root < 0, "No root found in the component graph");
        ARM_COMPUTE_ERROR_ON_MSG(_dst_id == -1, "Destination Tensor Id should be ready before calling get_execution_window()");

        return _components.find(_graph_root)->second->get_window();
    }

    ArgumentID get_dst_id() const
    {
        return _dst_id;
    }

    ClKernelArgList get_arguments() const
    {
        ClKernelArgList arg_list{};
        for(const auto &arg_var : _vtable.get_kernel_arguments().get_all_vars())
        {
            arg_list[arg_var.desc.arg_id] = arg_var.desc;
        }
        return arg_list;
    }

    /** Get the arguments as shared vars from the vtable
     *
     * @return SharedVarTable::Arguments
     */
    SharedVarTable::Arguments get_argument_shared_vars() const
    {
        return _vtable.get_kernel_arguments();
    }

    const ITensorInfo *get_kernel_argument_info(const ArgumentID id) const
    {
        auto it = _kernel_tensors.find(id);
        if(it != _kernel_tensors.end())
        {
            return it->second;
        }
        return nullptr;
    }

    ITensorInfo *get_kernel_argument_info(const ArgumentID id)
    {
        auto it = _kernel_tensors.find(id);
        if(it != _kernel_tensors.end())
        {
            return it->second;
        }
        return nullptr;
    }
    /** Finalize graph construction. Graph is expected to not mutate after being finalized
     */
    void finalize()
    {
        cache_root_component();
        assign_shared_var_group();
    }

    DependencyGraph get_graph() const
    {
        return _graph;
    }

private:
    void cache_root_component()
    {
        const auto roots = _graph.get_root_ops();
        ARM_COMPUTE_ERROR_ON_MSG(roots.size() != 1, "Trying to add more than one root to the graph");
        _graph_root = roots.at(0);
    }
    /** Assign the group for each shared var. Can only be performed at the end of the graph construction, before building
     */
    void assign_shared_var_group()
    {
        for(const auto &tensor : _kernel_tensors)
        {
            const auto tensor_id = tensor.first;
            if(_graph.is_src_tensor(tensor_id) || _graph.is_dst_tensor(tensor_id))
            {
                _shared_var_group_lut[tensor_id] = SharedVarGroup::Argument;
            }
            else
            {
                _shared_var_group_lut[tensor_id] = SharedVarGroup::Automatic;
            }
        }
    }

    void topological_sort_utility(ComponentID component_id, std::unordered_set<ComponentID> &visited, std::stack<ComponentID> &stack) const
    {
        visited.insert(component_id);

        for(auto connected_component : _component_graph.find(component_id)->second)
        {
            if(visited.find(connected_component) == visited.end())
            {
                topological_sort_utility(connected_component, visited, stack);
            }
        }

        stack.push(component_id);
    }

    std::stack<ComponentID> topological_sort() const
    {
        std::stack<ComponentID>         stack{};
        std::unordered_set<ComponentID> visited{};

        topological_sort_utility(_graph_root, visited, stack);

        return stack;
    }

    void traverse(const std::function<void(std::stack<ComponentID>)> &func) const
    {
        std::stack<ComponentID> stack = topological_sort();

        while(!stack.empty())
        {
            func(stack);
            stack.pop();
        }
    }

    std::string generate_argument_declaration(const SharedVarTable::SharedVar &var) const
    {
        ARM_COMPUTE_ERROR_ON_MSG(var.group != SharedVarGroup::Argument, "An argument declaration can only be generated from a kernel argument");
        std::string code;
        switch(var.desc.tensor_arg_type)
        {
            case ClKernelTensorArgType::Vector:
            {
                code += "\n    VECTOR_DECLARATION(" + var.uniq_name + ")";
                break;
            }
            case ClKernelTensorArgType::Image:
            {
                code += "\n    IMAGE_DECLARATION(" + var.uniq_name + ")";
                break;
            }
            case ClKernelTensorArgType::Image_3D:
            {
                code += "\n    IMAGE_DECLARATION(" + var.uniq_name + "),";
                code += "\n    uint " + var.uniq_name + "_stride_z";
                break;
            }
            case ClKernelTensorArgType::Image_3D_Export_To_ClImage2D:
            {
                code += "\n    __read_only image2d_t " + var.uniq_name + "_img,";
                code += "\n    uint " + var.uniq_name + "_stride_z";
                break;
            }
            case ClKernelTensorArgType::Tensor_4D_t_Buffer:
            {
                code += "\n    TENSOR4D_T(" + var.uniq_name + ", BUFFER)";
                break;
            }
            case ClKernelTensorArgType::Tensor_4D_t_Image:
            {
                code += "\n    TENSOR4D_T(" + var.uniq_name + ", IMAGE)";
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported declaration generation for ClKernelTensorArgType");
            }
        }
        return code;
    }

    std::string generate_kernel_signature(const SharedVarTable::Arguments &argument_list) const
    {
        std::string code = "\n__kernel void " + build_kernel_name() + "(";

        for(const auto &arg : argument_list.get_all_vars())
        {
            code += generate_argument_declaration(arg) + ",";
        }

        code[code.length() - 1] = ')';

        return code;
    }

    std::string generate_global_section() const
    {
        auto       dst_info   = get_kernel_argument_info(_dst_id);
        auto       dst_w      = dst_info->dimension(0);
        const auto tile_w     = std::max(1, get_execution_window().x().step());
        const auto tile_h     = std::max(1, get_execution_window().y().step());
        auto       leftover_w = dst_w % tile_w;

        std::string code = "";
        code += std::string("    int cout = GET_SPATIAL_IDX(0, ") + std::to_string(tile_w) + ", " + std::to_string(leftover_w) + ");\n";
        code += std::string("    int mout = GET_SPATIAL_IDX(1, ") + std::to_string(tile_h) + ", " + "0);\n";
        code += std::string("    int bout = GET_SPATIAL_IDX(2, 1, 0);\n\n");

        switch(_tile_info.clipping)
        {
            case ClippingStrategy::TOP_LEFT:
                code += "    const bool g_cond_x = (cout == 0);\n";
                code += "    const bool g_cond_y = (mout == 0);\n";
                break;
            case ClippingStrategy::TOP_RIGHT:
                code += "    const bool g_cond_x = ((cout + 1) * " + std::to_string(tile_w) + " >= " + std::to_string(_tile_info.boundaries.x()) + ");\n";
                code += "    const bool g_cond_y = (mout == 0);\n";
                break;
            case ClippingStrategy::BOTTOM_LEFT:
                code += "    const bool g_cond_x = (cout == 0);\n";
                code += "    const bool g_cond_y = ((mout + 1) * " + std::to_string(tile_h) + " >= " + std::to_string(_tile_info.boundaries.y()) + ");\n";
                break;
            case ClippingStrategy::BOTTOM_RIGHT:
                code += "    const bool g_cond_x = ((cout + 1) * " + std::to_string(tile_w) + " >= " + std::to_string(_tile_info.boundaries.x()) + ");\n";
                code += "    const bool g_cond_y = ((mout + 1) * " + std::to_string(tile_h) + " >= " + std::to_string(_tile_info.boundaries.y()) + ");\n";
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported clipping strategy");
        }

        return code;
    }

    TileDescriptor _tile_info{};

    int32_t _num_complex_components{};

    ArgumentID _dst_id{ -1 }; // Initially set to -1, which means the graph has no dst yet, since node IDs are positive numbers

    DependencyGraph _graph{};

    // Tensors, components and IDs with corresponding ptrs (except intermediate)
    std::unordered_map<ComponentID, ComponentUniquePtr> _components{};
    std::unordered_map<ArgumentID, ITensorInfo *>       _kernel_tensors{};
    // Argument group lookup. Can be replaced by extending the ArgumentID type to include group info
    std::unordered_map<ArgumentID, SharedVarGroup> _shared_var_group_lut{};

    // Tracks all variables (e.g.: kernel arguments, kernel "global variables")
    SharedVarTable _vtable{};

    // Component directed graph (represented by an adjecency list of Component IDs)
    // This is used to understand the ordering and bindings between components when generating the kernel
    // It's initially set to -1 which means the graph has no root yet, since node IDs are positive numbers
    ComponentID _graph_root{ -1 };
    std::unordered_map<ComponentID, ComponentList> _component_graph{};

    // Additional data structures used to define the relationships between components and arguments
    // For each argument, it contains the list of components that consider it as an incoming or an outgoing argument
    // E.g. tensor0  -> component0 -> tensor1
    // _outgoing_components[tensor0] == {component0} (component0 is the outgoing component of tensor0. Component0 treats tensor0 as an input tensor)
    // _incoming_components[tensor1] == {component0} (component0 is the incoming component of tensor1. Component1 treats tensor1 as an output tensor)
    std::unordered_map<ArgumentID, ComponentList> _outgoing_components{};
    std::unordered_map<ArgumentID, ComponentList> _incoming_components{};
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMMON_H