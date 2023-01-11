/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#include "ClTemplateWriter.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/IGpuTemplateComponentWriter.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/// @note: some tags can be unused since they could be used only for the macros, or only for the component code
std::string ClTemplateWriter::replace_tags(const std::string &code_template, const TagLUT &tags)
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
ClTemplateWriter::~ClTemplateWriter()
{
}
ClTemplateWriter::ClTemplateWriter(const GpuKernelComponentGroup &components)
    : _components{ components }
{
}
std::string ClTemplateWriter::get_name()
{
    return write_kernel_name();
}
std::string ClTemplateWriter::get_code()
{
    return write_code();
}
std::string ClTemplateWriter::get_config_id()
{
    std::string config_id = get_name();
    for(const auto &comp : _components)
    {
        config_id += "--" + comp->template_writer()->get_config_id() + "--";
    }

    return config_id;
}

CLBuildOptions ClTemplateWriter::get_build_options()
{
    CLBuildOptions build_opts{};

    for(const auto &comp : _components)
    {
        build_opts.add_options(comp->template_writer()->get_build_options(_components).options());
    }

    return build_opts;
}

Window ClTemplateWriter::get_window() const
{
    const auto root_comp = _components.get_root_component();
    ARM_COMPUTE_ERROR_ON_MSG(root_comp == nullptr, "No root component found");
    return root_comp->template_writer()->get_window();
}

std::map<ITensorInfo::Id, GpuKernelArgument> ClTemplateWriter::get_tensors()
{
    // Assemble GpuKernelArguments
    std::map<ITensorInfo::Id, GpuKernelArgument> tensors;
    for(const auto t : _components.get_argument_tensors())
    {
        tensors.emplace(
            t->id(),
            GpuKernelArgument{ *t, _vtable.get_variable(t).kernel_argument_info });
    }
    return tensors;
}

std::string ClTemplateWriter::write_code()
{
    ARM_COMPUTE_ERROR_ON_MSG(_components.empty(), "No components found");

    // These data structures will hold the data from all the components in the blueprint
    std::set<std::string>    headers_list{};
    std::set<std::string>    additional_macros{};
    std::vector<std::string> component_codes{}; // vector because order matters

    // Pass 1: Declare all kernel variables
    for(auto &component : _components)
    {
        component->template_writer()->declare_variables(_vtable, _components);
    }
    // Pass 2: Generate component codes
    for(auto &component : _components)
    {
        const auto component_writer       = component->template_writer();
        auto       curr_headers_list      = component_writer->get_headers_list();
        auto       curr_additional_macros = component_writer->get_additional_macros();
        auto       curr_component_code    = component_writer->get_component_code(_components);
        const auto var_lut                = component_writer->get_tag_lut(_vtable, _components); // Ideally can be merged with get_component_code once we have finer-grained code generation technique
        component_codes.push_back(replace_tags(curr_component_code, var_lut));

        headers_list.insert(curr_headers_list.begin(), curr_headers_list.end());
        if(!additional_macros.empty()) // Some components might not have any
        {
            additional_macros.insert(replace_tags(curr_additional_macros, var_lut));
        }
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

    auto arguments = _components.get_argument_tensors();
    std::sort(arguments.begin(), arguments.end(), [](const ITensorInfo * l, const ITensorInfo * r)
    {
        return l->id() < r->id();
    });
    code += write_kernel_signature(_vtable.get_variable_list(arguments));

    code += "\n{\n\n";

    code += "    //------------------ START KERNEL_BUILDER_COORDINATE ---------------------\n\n";
    code += write_global_section();
    code += "    //------------------ END KERNEL_BUILDER_COORDINATE ---------------------\n";

    {
        const auto        tiles = _components.get_tiles();
        std::stringstream tiles_ss;

        tiles_ss << "    //------------------ START TILE DECLARATION ---------------------\n";

        for(auto tile : tiles)
        {
            const auto var       = _vtable.get_variable(tile);
            const auto data_type = get_cl_type_from_data_type(tile->data_type());
            const auto var_name  = var.uniq_name;

            tiles_ss << "    TILE(" << data_type << ", M0, N0, " << var_name << ");\n";
        }

        tiles_ss << "    //------------------ END TILE DECLARATION ---------------------\n";

        code += tiles_ss.str();
    }

    for(const auto &component_code : component_codes)
    {
        code += component_code;
        code += "\n";
    }

    code += "}\n";

    return code;
}
std::string ClTemplateWriter::write_global_section() const
{
    const auto dst_info   = _components.get_any_dst_tensor();
    const auto dst_w      = dst_info->dimension(0);
    const auto tile_w     = std::max(1, get_window().x().step());
    const auto tile_h     = std::max(1, get_window().y().step());
    auto       leftover_w = dst_w % tile_w;

    std::string code = "";
    code += std::string("    int g_ind_0 = GET_SPATIAL_IDX(0, ") + std::to_string(tile_w) + ", " + std::to_string(leftover_w) + ");\n";
    code += std::string("    int g_ind_1 = GET_SPATIAL_IDX(1, ") + std::to_string(tile_h) + ", " + "0);\n";
    code += std::string("    int g_ind_2 = GET_SPATIAL_IDX(2, 1, 0);\n\n");

    code += "    const bool g_cond_x = (g_ind_0 == 0);\n";
    code += "    const bool g_cond_y = (g_ind_1 == 0);\n";

    return code;
}
std::string ClTemplateWriter::write_argument_declaration(const GpuKernelVariableTable::TensorVariable &var) const
{
    std::string code;
    switch(var.kernel_argument_info.type)
    {
        case GpuKernelArgumentInfo::Type::Vector:
        {
            code += "\n    VECTOR_DECLARATION(" + var.uniq_name + ")";
            break;
        }
        case GpuKernelArgumentInfo::Type::Image:
        {
            code += "\n    IMAGE_DECLARATION(" + var.uniq_name + ")";
            break;
        }
        case GpuKernelArgumentInfo::Type::Image_3D:
        {
            code += "\n    IMAGE_DECLARATION(" + var.uniq_name + "),";
            code += "\n    unsigned int " + var.uniq_name + "_stride_z";
            break;
        }
        case GpuKernelArgumentInfo::Type::Image_3D_Export_To_ClImage2D:
        {
            code += "\n    __read_only image2d_t " + var.uniq_name + "_img,";
            code += "\n    unsigned int " + var.uniq_name + "_stride_z";
            break;
        }
        case GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer:
        {
            code += "\n    TENSOR4D_T(" + var.uniq_name + ", BUFFER)";
            break;
        }
        case GpuKernelArgumentInfo::Type::Tensor_4D_t_Image:
        {
            code += "\n    TENSOR4D_T(" + var.uniq_name + ", IMAGE)";
            break;
        }
        case GpuKernelArgumentInfo::Type::Tensor_3D:
        {
            code += "\n    TENSOR3D_DECLARATION(" + var.uniq_name + ")";
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported declaration generation for GpuKernelArgumentInfo::Type");
        }
    }
    return code;
}
std::string ClTemplateWriter::write_kernel_signature(const GpuKernelVariableTable::VariableList &argument_list) const
{
    std::string code = "\n__kernel void " + write_kernel_name() + "(";

    for(int i = 0; i < static_cast<int>(argument_list.size()) - 1; ++i)
    {
        code += write_argument_declaration(argument_list[i]) + ",";
    }
    if(static_cast<int>(argument_list.size()) - 1 >= 0)
    {
        code += write_argument_declaration(argument_list[argument_list.size() - 1]);
    }

    code += ')';

    return code;
}
std::string ClTemplateWriter::write_kernel_name() const
{
    if(_components.empty())
    {
        return "empty_kernel";
    }
    std::string name = _components.empty() ? "" : _components[0]->template_writer()->get_name();
    for(size_t i = 1; i < _components.size(); ++i)
    {
        name += "___";
        name += _components[i]->template_writer()->get_name();
    }

    return name;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
