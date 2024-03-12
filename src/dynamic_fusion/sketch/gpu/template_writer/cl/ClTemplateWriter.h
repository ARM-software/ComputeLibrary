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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATEWRITER
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATEWRITER

#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/IGpuKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/GpuKernelVariableTable.h"

#include <map>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Use a templated-string-based method to write kernel code
 *  It stitches the component code templates together based on the valid fusion configuration.
 *  It then instantiates the actual kernel code from the template and the generated tag lookup table.
 */
class ClTemplateWriter : public IGpuKernelWriter
{
public:
    /** Instantiates a kernel code string from the kernel code template
     * @note: some tags can be unused since they could be used only for the macros, or only for the component code
     *
     * @param[in] code_template Kernel code template
     * @param[in] tags          Tag lookup table
     *
     * @return std::string  Instantiated kernel string
     */
    static std::string replace_tags(const std::string &code_template, const TagLUT &tags);
    /** Default constructor */
    ClTemplateWriter() = default;
    /** Constructor
     *
     * @param[in] components Kernel component group from which the kernel will be generated
     */
    ClTemplateWriter(const GpuKernelComponentGroup &components);
    /** Destructor */
    ~ClTemplateWriter() override;
    /** Generate kernel name */
    std::string get_name() override;
    /** Generate kernel code */
    std::string get_code() override;
    /** Generate build options */
    CLBuildOptions get_build_options() override;
    /** Generate config id string of the entire kernel. This is used for tuning */
    std::string get_config_id() override;
    /** Generate execution window */
    Window get_window() const override;
    /** Get the kernel argument lists of the kernel*/
    std::map<ITensorInfo::Id, GpuKernelArgument> get_tensors() override;

private:
    std::string write_kernel_name() const;
    std::string write_code();
    std::string write_global_section() const;
    std::string write_argument_declaration(const GpuKernelVariableTable::TensorVariable &var) const;
    std::string write_kernel_signature(const GpuKernelVariableTable::VariableList &argument_list) const;

private:
    GpuKernelComponentGroup _components{};
    GpuKernelVariableTable  _vtable{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATEWRITER */
