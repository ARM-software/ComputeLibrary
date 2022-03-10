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
#if defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)

#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMMON_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMMON_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "src/core/common/Macros.h"

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"

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
    Argument, // Parameters to a kernel function
    Automatic // Automatic variables declared within the kernel body
};

/** Specifies a shared variable link for a component.
 * It describes all the information that's available when a component is constructed / added:
 *  e.g. its linkage (via ArgumentID and io) and its group
 * This is not shared variable on its own, but is used for instantiating a SharedVar when building the code
 */
struct SharedVarLink
{
    ArgumentID     arg_id{ g_arg_placeholder };
    SharedVarIO    io{ SharedVarIO::Input };
    SharedVarGroup group{ SharedVarGroup::Argument };
    bool           is_empty() const
    {
        return arg_id == g_arg_placeholder;
    }
};

/** A table of all the variables used in the kernel / blueprint
 * NOTE: the order they appear in the table is the order of their "declaration" in the component code, and is also their ID
 * NOTE: the variables all have the scope of the full kernel function
 */
class SharedVarTable
{
public:
    struct SharedVar
    {
        SharedVarGroup               group;
        std::string                  uniq_name; // Unique name, also the final variable name used in the built code
        ClKernelArgRuntimeDescriptor desc;      // Automatic variables can and should still be described using this struct
    };

    using Arguments = std::vector<SharedVar>;

    /** @note: The order of insertion is important. There is one precondition:
     *        PRECOND: The components have been sorted topologically / is being traversed in topological order
     *                 This ensures that all the consumer var links (Output, Automatic Links) can consume (return) the producer var links when they're referred
     */
    SharedVar add(SharedVarLink var_link, ClKernelArgRuntimeDescriptor runtime_desc, const std::string &name = "unnamed")
    {
        ARM_COMPUTE_ERROR_ON_MSG(var_link.is_empty(), "Non-empty SharedVarLink expected");
        auto              var_id = _num_var;
        std::stringstream ss;
        ss << name << "_" << var_id;
        const auto uniq_name = ss.str();
        SharedVar  var{ var_link.group, uniq_name, runtime_desc };

        if(var_link.group == SharedVarGroup::Argument)
        {
            _arguments.emplace(var_id, var);
            _num_var++;
            _var_id_lut[var_link.arg_id] = var_id;
        }
        else if(var_link.group == SharedVarGroup::Automatic)
        {
            if(var_link.io == SharedVarIO::Output)
            {
                _global_vars.emplace(var_id, var);
                _num_var++;
                _var_id_lut[var_link.arg_id] = var_id;
            }
            else
            {
                // For the input link, the var (and thus its arg_id) will always have been added by the time we get here if we traverse components in topological order
                var = get_var(var_link.arg_id);
            }
        }
        else
        {
            ARM_COMPUTE_ERROR("Unrecognised SharedVarGroup");
        }
        return var;
    }

    SharedVar get_var(ArgumentID arg_id) const
    {
        const auto var_id = _var_id_lut.at(arg_id); // arg_id has to exist in lut to begin with
        auto       it     = _global_vars.find(var_id);
        if(it != _global_vars.end())
        {
            return it->second;
        }
        it = _arguments.find(var_id);
        if(it != _arguments.end())
        {
            return it->second;
        }
        ARM_COMPUTE_ERROR("Cannot find component variable");
    }

    /** @note The arguments are returned in the order they are added
     */
    Arguments get_kernel_arguments() const
    {
        Arguments args{};
        for(const auto &a : _arguments)
        {
            args.push_back(a.second);
        }
        return args;
    }

private:
    using VarID = int32_t;

private:
    std::map<VarID, SharedVar>            _global_vars{};
    std::map<VarID, SharedVar>            _arguments{};
    std::unordered_map<ArgumentID, VarID> _var_id_lut{};
    VarID _num_var{ 0 };
};

enum class ComponentType
{
    Simple,
    Complex,
    Store
};

using ComponentID   = int32_t;
using ComponentList = std::vector<ComponentID>;
class IClKernelComponent
{
public:
    using Link = SharedVarLink;
    using Tag  = std::string;
    struct TagVal
    {
        TagVal() = default;
        TagVal(SharedVarTable::SharedVar var)
            : value{ var.uniq_name }
        {
        }

        TagVal(ComponentID id)
            : value{ std::to_string(id) }
        {
        }

        std::string value{};
    };
    using TagLUT = std::unordered_map<Tag, TagVal>; // Used to instantiating a code template / replacing tags
public:
    IClKernelComponent(const ClKernelBlueprint *blueprint)
        : _blueprint(blueprint)
    {
    }

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(IClKernelComponent);

    virtual ~IClKernelComponent()                        = default;
    virtual ComponentType     get_component_type() const = 0;
    virtual std::vector<Link> get_links() const          = 0;
    virtual std::string       name() const               = 0;

    static std::string replace_tags(const std::string &code_template, const TagLUT &tags)
    {
        std::string                     replaced_code = "";
        std::unordered_set<std::string> used_tags{};
        bool                            scanning_pattern = false;
        std::string                     pattern_found    = "";
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
                    used_tags.insert(pattern_found);
                }
                else
                {
                    pattern_found += code_template[i];
                }
            }
        }
        // Check for unused tags
        for(const auto &tag : tags)
        {
            ARM_COMPUTE_UNUSED(tag);
            ARM_COMPUTE_ERROR_ON_MSG(used_tags.find(tag.first) == used_tags.end(), "Warning: unused tags");
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
    /** "Allocate" all shared variables used in a component to the @p vtable, and generate a TagLUT used to instantiate the component code
     *
     * @param vtable
     * @return TagLUT
     */
    virtual TagLUT allocate_vars(SharedVarTable &vtable) const = 0;

    virtual std::string get_dst_addr_calculation() const
    {
        return "";
    }

protected:
    const ClKernelBlueprint *_blueprint;

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
    ArgumentID add_kernel_argument(const ClTensorDescriptor &tensor_desc)
    {
        _kernel_arguments.insert(std::make_pair(_num_args, tensor_desc));
        _shared_var_group_lut[_num_args] = SharedVarGroup::Argument;
        return _num_args++;
    }

    ArgumentID add_intermediate_tensor()
    {
        _intermediate_tensors.insert(_num_args);
        _shared_var_group_lut[_num_args] = SharedVarGroup::Automatic;
        return _num_args++;
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
            ARM_COMPUTE_ERROR_ON_MSG(_kernel_arguments.find(arg_id) == _kernel_arguments.end() && _intermediate_tensors.find(arg_id) == _intermediate_tensors.end() && arg_id != g_arg_placeholder,
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

        // This flag specifies if the current component is the root of the component graph
        // If the root is set to -1, it means that a root hasn't been added yet
        bool is_graph_root = true;

        // Get an unique ID for the component that's being added
        const ComponentID component_id = _num_components++;
        component->set_id(component_id);

        // Add this component to the component graph. Don't connect it to anything yet
        _component_graph.emplace(component_id, ComponentList{});

        int32_t positional_arg = 0;

        // For every { arg_id, arg_io } passed along with this component...
        for(const auto &link : component->get_links())
        {
            const ArgumentID &arg_id = link.arg_id;
            const SharedVarIO &arg_io = link.io;

            // A component is considered root only if all its input arguments are kernel arguments (or placeholders, which means nullptr)
            // This performs a check on every argument, and if one of them doesn't respect the condition, the component is not considered root
            is_graph_root &= (_kernel_arguments.find(arg_id) != _kernel_arguments.end()) || (arg_io == SharedVarIO::Output) || (arg_id == g_arg_placeholder);

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

            ++positional_arg;
        }

        if(is_graph_root)
        {
            ARM_COMPUTE_ERROR_ON_MSG(_graph_root >= 0, "Trying to add more than one root to the graph");
            _graph_root = component_id;
        }

        // Finally, add this component to the dictionary of components
        _components.insert(std::make_pair(component_id, std::move(component)));
    }

    std::string build_kernel_name() const
    {
        std::string name = "";

        auto stack = topological_sort();
        while(!stack.empty())
        {
            name += _components.find(stack.top())->second->name() + (stack.size() > 2 ? "___" : "");
            stack.pop();
        }

        return name;
    }

    std::string build_code()
    {
        ARM_COMPUTE_ERROR_ON_MSG(_graph_root < 0, "No root found in the component graph");

        // These data structures will hold the data from all the components in the blueprint
        std::set<std::string>    headers_list{};
        std::set<std::string>    additional_macros{};
        std::vector<std::string> component_codes{}; // vector because order matters

        // Go through the components graph (topological sort) and fill the data structures above
        auto stack = topological_sort();
        while(!stack.empty())
        {
            auto  curr_component_id = stack.top();
            auto &curr_component    = _components.find(curr_component_id)->second;

            auto       curr_headers_list      = curr_component->get_headers_list();
            auto       curr_additional_macros = curr_component->get_additional_macros();
            auto       curr_component_code    = curr_component->get_component_code();
            const auto var_lut                = curr_component->allocate_vars(_vtable); // Ideally can be merged with get_component_code once we have finer-grained code generation technique
            component_codes.push_back(IClKernelComponent::replace_tags(curr_component_code, var_lut));

            headers_list.insert(curr_headers_list.begin(), curr_headers_list.end());
            if(!curr_additional_macros.empty()) // Some components might not have any
            {
                additional_macros.insert(curr_additional_macros);
            }

            stack.pop();
        }

        // This section assembles the data gathered by traversing the graph into the string "code"
        std::string code = "";

        for(auto &header : headers_list)
        {
            code += "#include \"" + header + "\"\n";
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

    std::string build_config_id() const
    {
        return "";
    }

    CLBuildOptions build_options() const
    {
        return CLBuildOptions{};
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
        for(const auto &arg_var : _vtable.get_kernel_arguments())
        {
            arg_list.push_back(arg_var.desc);
        }
        return arg_list;
    }

    const ClTensorDescriptor *get_kernel_argument(const ArgumentID id) const
    {
        auto it = _kernel_arguments.find(id);
        if(it != _kernel_arguments.end())
        {
            return &_kernel_arguments.find(id)->second;
        }
        return nullptr;
    }

    ITensorInfo *get_kernel_argument_info(const ArgumentID id) const
    {
        const ClTensorDescriptor *arg_desc = get_kernel_argument(id);
        if(arg_desc != nullptr)
        {
            return arg_desc->tensor_info;
        }
        return nullptr;
    }

private:
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

    std::string generate_argument_declaration(const SharedVarTable::SharedVar &var) const
    {
        ARM_COMPUTE_ERROR_ON_MSG(var.group != SharedVarGroup::Argument, "An argument declaration can only be generated from a kernel argument");
        std::string code;
        switch(var.desc.tensor_arg_type)
        {
            case TensorArgType::Image:
            {
                code += "IMAGE_DECLARATION(" + var.uniq_name + ")";
                break;
            }
            case TensorArgType::Image_3D:
            {
                code += "IMAGE_DECLARATION(" + var.uniq_name + "),\n";
                code += "uint " + var.uniq_name + "_stride_z";
                break;
            }
            case TensorArgType::Image_3D_Export_To_ClImage2D:
            {
                code += "__read_only image2d_t " + var.uniq_name + "_img,\n";
                code += "uint " + var.uniq_name + "_stride_z,\n";
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported declaration generation for TensorArgType");
            }
        }
        return code;
    }

    std::string generate_kernel_signature(const SharedVarTable::Arguments &argument_list) const
    {
        std::string code = "\n__kernel void " + build_kernel_name() + "(";

        for(const auto &arg : argument_list)
        {
            code += "\n    " + generate_argument_declaration(arg) + ",";
        }

        code[code.length() - 1] = ')';

        return code;
    }

    std::string generate_global_section() const
    {
        std::string code = "    uint g_x = get_global_id(0);\n";
        code += "    uint g_y = get_global_id(1);\n";
        code += "    uint g_z = get_global_id(2);\n\n";

        size_t tile_dim_x = _tile_info.empty() ? 1 : _tile_info.tile_dims.x();
        size_t tile_dim_y = _tile_info.empty() ? 1 : _tile_info.tile_dims.y();

        switch(_tile_info.clipping)
        {
            case ClippingStrategy::TOP_LEFT:
                code += "    const bool g_cond_x = (g_x == 0);\n";
                code += "    const bool g_cond_y = (g_y == 0);\n";
                break;
            case ClippingStrategy::TOP_RIGHT:
                code += "    const bool g_cond_x = ((g_x + 1) * " + std::to_string(tile_dim_x) + " >= " + std::to_string(_tile_info.boundaries.x()) + ");\n";
                code += "    const bool g_cond_y = (g_y == 0);\n";
                break;
            case ClippingStrategy::BOTTOM_LEFT:
                code += "    const bool g_cond_x = (g_x == 0);\n";
                code += "    const bool g_cond_y = ((g_y + 1) * " + std::to_string(tile_dim_y) + " >= " + std::to_string(_tile_info.boundaries.y()) + ");\n";
                break;
            case ClippingStrategy::BOTTOM_RIGHT:
                code += "    const bool g_cond_x = ((g_x + 1) * " + std::to_string(tile_dim_x) + " >= " + std::to_string(_tile_info.boundaries.x()) + ");\n";
                code += "    const bool g_cond_y = ((g_y + 1) * " + std::to_string(tile_dim_y) + " >= " + std::to_string(_tile_info.boundaries.y()) + ");\n";
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported clipping strategy");
        }

        code += "\n    REPEAT_VAR_INIT_TO_CONST(M0, uint, g_zout, 0);\n";
        code += "    REPEAT_VAR_INIT_TO_CONST(16, uint, g_zero, 0);\n\n";

        return code;
    }

    TileDescriptor _tile_info{};

    int32_t _num_args{};
    int32_t _num_components{};
    int32_t _num_complex_components{};

    ArgumentID _dst_id{ -1 };

    // Argument, components and intermediate tensors IDs with corresponding ptrs (except intermediate)
    std::unordered_map<ComponentID, ComponentUniquePtr> _components{};
    std::unordered_map<ArgumentID, ClTensorDescriptor>  _kernel_arguments{};
    std::unordered_set<ArgumentID> _intermediate_tensors{};
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

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)