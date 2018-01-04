/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <utility>
#include <vector>

using namespace arm_compute;

GCProgram::GCProgram()
    : _name(), _source()
{
}

GCProgram::GCProgram(std::string name, std::string source)
    : _name(std::move(name)), _source(std::move(source))
{
}

GLuint GCProgram::link_program(GLuint shader)
{
    GLuint program = ARM_COMPUTE_GL_CHECK(glCreateProgram());

    GLint   rvalue;
    GLsizei length;

    ARM_COMPUTE_GL_CHECK(glAttachShader(program, shader));
    ARM_COMPUTE_GL_CHECK(glLinkProgram(program));
    ARM_COMPUTE_GL_CHECK(glDetachShader(program, shader));
    ARM_COMPUTE_GL_CHECK(glDeleteShader(shader));

    // Check if there were some issues when linking the shader.
    ARM_COMPUTE_GL_CHECK(glGetProgramiv(program, GL_LINK_STATUS, &rvalue));

    if(rvalue == 0)
    {
        ARM_COMPUTE_GL_CHECK(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length));

        std::vector<GLchar> log(length);
        ARM_COMPUTE_GL_CHECK(glGetProgramInfoLog(program, length, nullptr, log.data()));
        ARM_COMPUTE_ERROR("Error: Linker log:\n%s\n", log.data());

        return 0;
    }

    ARM_COMPUTE_GL_CHECK(glUseProgram(program));

    return program;
}

GLuint GCProgram::compile_shader(const std::string &build_options)
{
    GLuint shader = ARM_COMPUTE_GL_CHECK(glCreateShader(GL_COMPUTE_SHADER));

    const char *src[]
    {
        "#version 310 es\n",
        build_options.c_str(),
        _source.c_str()
    };

    ARM_COMPUTE_GL_CHECK(glShaderSource(shader, sizeof(src) / sizeof(src[0]), src, nullptr));

    ARM_COMPUTE_GL_CHECK(glCompileShader(shader));

    // Check if there were any issues when compiling the shader
    GLint   rvalue;
    GLsizei length;

    ARM_COMPUTE_GL_CHECK(glGetShaderiv(shader, GL_COMPILE_STATUS, &rvalue));

    if(rvalue == 0)
    {
        ARM_COMPUTE_GL_CHECK(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length));

        std::vector<GLchar> log(length);
        ARM_COMPUTE_GL_CHECK(glGetShaderInfoLog(shader, length, nullptr, log.data()));

#ifdef ARM_COMPUTE_DEBUG_ENABLED
        std::istringstream ss(_source);
        std::stringstream  output_stream;
        std::string        line;
        size_t             line_num = 1;

        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("GLES Shader build options:\n%s\n", build_options.c_str());
        while(std::getline(ss, line, '\n'))
        {
            output_stream << std::setw(6) << line_num << ": " << line << std::endl;
            line_num++;
        }
        ARM_COMPUTE_LOG_INFO_STREAM_CORE("GLES Shader source code:\n"
                                         << output_stream.rdbuf());
#endif /* ARM_COMPUTE_DEBUG_ENABLED */

        ARM_COMPUTE_ERROR("Error: Compiler log:\n%s\n", log.data());

        return 0;
    }

    return shader;
}

GCKernel::GCKernel()
    : _name(), _program(), _shader_arguments(), _shader_params_ubo_name(), _shader_params_binding_point(), _shader_params_index(), _shader_params_size()
{
}

// Add a default destructor in cpp file to workaround the free unallocated value issue on Android
GCKernel::~GCKernel() // NOLINT
{
}

GCKernel::GCKernel(std::string name, GLuint program)
    : _name(std::move(name)),
      _program(program),
      _shader_arguments(),
      _shader_params_ubo_name(0),
      _shader_params_binding_point(0),
      _shader_params_index(0),
      _shader_params_size(0)
{
    _shader_arguments.clear();

    ARM_COMPUTE_GL_CHECK(glGenBuffers(1, &_shader_params_ubo_name));

    _shader_params_index = ARM_COMPUTE_GL_CHECK(glGetUniformBlockIndex(_program, _shader_params_name));
    ARM_COMPUTE_ERROR_ON_MSG((_shader_params_index == GL_INVALID_INDEX), "Failed to get index of %s", _shader_params_name);
    ARM_COMPUTE_GL_CHECK(glGetActiveUniformBlockiv(_program, _shader_params_index, GL_UNIFORM_BLOCK_DATA_SIZE, &_shader_params_size));
    ARM_COMPUTE_ERROR_ON_MSG((_shader_params_size == 0), "Failed to get size of %s", _shader_params_name);
}

void GCKernel::cleanup()
{
    ARM_COMPUTE_GL_CHECK(glDeleteBuffers(1, &_shader_params_ubo_name));
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_UNIFORM_BUFFER, 0));
    ARM_COMPUTE_GL_CHECK(glDeleteProgram(_program));
    ARM_COMPUTE_GL_CHECK(glUseProgram(0));
}

void GCKernel::use()
{
    ARM_COMPUTE_GL_CHECK(glUseProgram(_program));
}

void GCKernel::unuse()
{
    ARM_COMPUTE_GL_CHECK(glUseProgram(0));
}

void GCKernel::update_shader_params()
{
    ARM_COMPUTE_ERROR_ON_MSG((_shader_params_size != (int)(_shader_arguments.size() * sizeof(_shader_arguments[0]))), "Arguments size (%d) is not equal to shader params block size (%d)",
                             _shader_arguments.size() * sizeof(_shader_arguments[0]), _shader_params_size);

    ARM_COMPUTE_GL_CHECK(glUniformBlockBinding(_program, _shader_params_index, _shader_params_binding_point));
    ARM_COMPUTE_GL_CHECK(glBindBufferBase(GL_UNIFORM_BUFFER, _shader_params_binding_point, _shader_params_ubo_name));
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_UNIFORM_BUFFER, _shader_params_ubo_name));
    ARM_COMPUTE_GL_CHECK(glBufferData(GL_UNIFORM_BUFFER, _shader_params_size, _shader_arguments.data(), GL_DYNAMIC_DRAW));
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_UNIFORM_BUFFER, 0));
}

const std::map<std::string, std::string> GCKernelLibrary::_shader_program_map =
{
    { "absdiff", "absdiff.cs" },
    { "direct_convolution1x1", "direct_convolution1x1.cs" },
    { "direct_convolution3x3", "direct_convolution3x3.cs" },
    { "direct_convolution5x5", "direct_convolution5x5.cs" },
    { "pooling_layer_2", "pooling_layer.cs" },
    { "pooling_layer_3", "pooling_layer.cs" },
    { "pooling_layer_7", "pooling_layer.cs" },
    { "pooling_layer_3_optimized", "pooling_layer.cs" },
    { "pooling_layer_n", "pooling_layer.cs" },
    { "fill_image_borders_replicate", "fill_border.cs" },
    { "fill_image_borders_constant", "fill_border.cs" },
    { "gemm_accumulate_biases", "gemm.cs" },
    { "gemm_interleave4x4", "gemm.cs" },
    { "gemm_ma", "gemm.cs" },
    { "gemm_mm_interleaved_transposed", "gemm.cs" },
    { "gemm_mm_floating_point", "gemm.cs" },
    { "gemm_transpose1x4", "gemm.cs" },
    { "reshape_to_columns", "convolution_layer.cs" },
    { "im2col_kernel3x3_padx0_pady0", "convolution_layer.cs" },
    { "im2col_generic", "convolution_layer.cs" },
    { "im2col_reduced", "convolution_layer.cs" },
    { "col2im", "convolution_layer.cs" },
    { "transpose", "transpose.cs" },
    { "activation_layer", "activation_layer.cs" },
    { "softmax_layer_max", "softmax_layer.cs" },
    { "softmax_layer_shift_exp_sum", "softmax_layer.cs" },
    { "softmax_layer_norm", "softmax_layer.cs" },
    { "pixelwise_mul_float", "pixelwise_mul_float.cs" },
    { "normalization_layer", "normalization_layer.cs" },
    { "batchnormalization_layer", "batchnormalization_layer.cs" },
    { "concatenate_depth", "concatenate.cs" },
    { "dropout", "dropout.cs" },
    { "normalize_planar_yuv_layer", "normalize_planar_yuv_layer.cs" },
    { "scale_nearest_neighbour", "scale.cs" },
    { "arithmetic_add", "arithmetic_add.cs" },
    { "depthwise_convolution_3x3", "depthwise_convolution3x3.cs" },
};

const std::map<std::string, std::string> GCKernelLibrary::_program_source_map =
{
#ifdef EMBEDDED_KERNELS
    {
        "absdiff.cs",
#include "./cs_shaders/absdiff.csembed"
    },
    {
        "convolution_layer.cs",
#include "./cs_shaders/convolution_layer.csembed"
    },
    {
        "direct_convolution1x1.cs",
#include "./cs_shaders/direct_convolution1x1.csembed"
    },
    {
        "direct_convolution3x3.cs",
#include "./cs_shaders/direct_convolution3x3.csembed"
    },
    {
        "direct_convolution5x5.cs",
#include "./cs_shaders/direct_convolution5x5.csembed"
    },
    {
        "pooling_layer.cs",
#include "./cs_shaders/pooling_layer.csembed"
    },
    {
        "fill_border.cs",
#include "./cs_shaders/fill_border.csembed"
    },
    {
        "gemm.cs",
#include "./cs_shaders/gemm.csembed"
    },
    {
        "transpose.cs",
#include "./cs_shaders/transpose.csembed"
    },
    {
        "activation_layer.cs",
#include "./cs_shaders/activation_layer.csembed"
    },
    {
        "softmax_layer.cs",
#include "./cs_shaders/softmax_layer.csembed"
    },
    {
        "pixelwise_mul_float.cs",
#include "./cs_shaders/pixelwise_mul_float.csembed"
    },
    {
        "normalization_layer.cs",
#include "./cs_shaders/normalization_layer.csembed"
    },
    {
        "batchnormalization_layer.cs",
#include "./cs_shaders/batchnormalization_layer.csembed"
    },
    {
        "concatenate.cs",
#include "./cs_shaders/concatenate.csembed"
    },
    {
        "dropout.cs",
#include "./cs_shaders/dropout.csembed"
    },
    {
        "normalize_planar_yuv_layer.cs",
#include "./cs_shaders/normalize_planar_yuv_layer.csembed"
    },
    {
        "scale.cs",
#include "./cs_shaders/scale.csembed"
    },
    {
        "arithmetic_add.cs",
#include "./cs_shaders/arithmetic_add.csembed"
    },
    {
        "depthwise_convolution3x3.cs",
#include "./cs_shaders/depthwise_convolution3x3.csembed"
    },
#endif /* EMBEDDED_KERNELS */
};

GCKernelLibrary::GCKernelLibrary()
    : _display(EGL_NO_DISPLAY), _context(EGL_NO_CONTEXT), _frame_buffer(0), _tex_rt(0), _shader_path("./"), _programs_map(), _built_programs_map()
{
}

GCKernelLibrary &GCKernelLibrary::get()
{
    static GCKernelLibrary _kernel_library;
    return _kernel_library;
}

GCKernel GCKernelLibrary::create_kernel(const std::string &shader_name, const StringSet &build_options_set) const
{
    // Find which program contains the kernel
    auto shader_program_it = _shader_program_map.find(shader_name);

    if(_shader_program_map.end() == shader_program_it)
    {
        ARM_COMPUTE_ERROR("Shader %s not found in the GCKernelLibrary", shader_name.c_str());
    }

    // Check if the program has been built before with same build options.
    const std::string program_name       = shader_program_it->second;
    const std::string build_options      = stringify_set(build_options_set);
    const std::string built_program_name = program_name + "_" + build_options;
    auto              built_program_it   = _built_programs_map.find(built_program_name);

    GCKernel kernel;

    if(_built_programs_map.end() != built_program_it)
    {
        // If program has been built, retrieve to create kernel from it
        kernel = built_program_it->second;
    }
    else
    {
        GCProgram program = load_program(program_name);

        std::string source_name = _shader_path + shader_program_it->second;

        // load shader
        GLuint shader = program.compile_shader(build_options);

        // Build program
        GLuint gles_program = program.link_program(shader);

        // Create GCKernel
        kernel = GCKernel(shader_name, gles_program);

        // Add built program to internal map
        _built_programs_map.emplace(built_program_name, kernel);
    }

    kernel.use();
    kernel.clear_arguments();
    // set shader params binding point
    kernel.set_shader_params_binding_point(0);

    return kernel;
}

const std::string GCKernelLibrary::preprocess_shader(const std::string &shader_source) const
{
    enum class ParserStage
    {
        FIRST,
        SKIP_COMMENTS = FIRST,
        RESOLVE_INCLUDES,
        LAST
    };

    // Define a GLES compute shader parser function
    std::function<std::string(const std::string &, ParserStage, int)> cs_parser;
    cs_parser = [&](const std::string & src, ParserStage stage, int nested_level) -> std::string
    {
        std::string dst;

        if(stage == ParserStage::LAST || std::regex_match(src, std::regex(R"(\s*)")))
        {
            return src;
        }
        auto next_stage = static_cast<ParserStage>(static_cast<int>(stage) + 1);

        std::string search_pattern;
        switch(stage)
        {
            case ParserStage::SKIP_COMMENTS:
                search_pattern = R"((/\*([^*]|\n|(\*+([^*/]|\n)))*\*+/)|(//.*))";
                break;
            case ParserStage::RESOLVE_INCLUDES:
                search_pattern = R"rgx((?:^|\n)[ \t]*#include "(.*)")rgx";
                break;
            default:
                break;
        }

        std::regex  search_regex(search_pattern);
        std::smatch match;
        ptrdiff_t   parsed_pos = 0;
        if(std::regex_search(src, match, search_regex))
        {
            // Pass the content before the match to the next stage
            dst.append(cs_parser(src.substr(0, match.position()), next_stage, 0));
            parsed_pos = match.position() + match.length();

            // Deal with the matched content
            switch(stage)
            {
                case ParserStage::RESOLVE_INCLUDES:
                {
                    // Replace with the included file contents
                    // And parse the content from the first stage
                    const std::string source_name = _shader_path + match.str(1);
                    dst.append(cs_parser(read_file(source_name, false), ParserStage::FIRST, 0));
                    break;
                }
                case ParserStage::SKIP_COMMENTS:
                default:
                    dst.append(match.str());
                    break;
            }
            next_stage = stage;
        }
        dst.append(cs_parser(src.substr(parsed_pos, src.length() - parsed_pos), next_stage, 0));

        return dst;
    };

    return cs_parser(shader_source, ParserStage::FIRST, 0);
}

const GCProgram &GCKernelLibrary::load_program(const std::string &program_name) const
{
    const auto program_it = _programs_map.find(program_name);

    if(program_it != _programs_map.end())
    {
        return program_it->second;
    }

    GCProgram program;

#ifdef EMBEDDED_KERNELS
    const auto program_source_it = _program_source_map.find(program_name);

    if(_program_source_map.end() == program_source_it)
    {
        ARM_COMPUTE_ERROR("Embedded program for %s does not exist.", program_name.c_str());
    }

    program = GCProgram(program_name, program_source_it->second);
#else  /* EMBEDDED_KERNELS */
    // Check for binary
    std::string source_name = _shader_path + program_name;
    if(std::ifstream(source_name).is_open())
    {
        program = GCProgram(program_name, preprocess_shader(read_file(source_name, false)));
    }
    else
    {
        ARM_COMPUTE_ERROR("Shader file %s does not exist.", source_name.c_str());
    }
#endif /* EMBEDDED_KERNELS */

    // Insert program to program map
    const auto new_program = _programs_map.emplace(program_name, std::move(program));

    return new_program.first->second;
}

void GCKernelLibrary::setup_dummy_fbo()
{
    ARM_COMPUTE_GL_CHECK(glGenFramebuffers(1, &_frame_buffer));
    ARM_COMPUTE_GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, _frame_buffer));
    ARM_COMPUTE_GL_CHECK(glGenTextures(1, &_tex_rt));
    ARM_COMPUTE_GL_CHECK(glBindTexture(GL_TEXTURE_2D, _tex_rt));
    ARM_COMPUTE_GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr));
    ARM_COMPUTE_GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _tex_rt, 0));
}

GCKernelLibrary::~GCKernelLibrary()
{
    for(auto &program : _built_programs_map)
    {
        static_cast<GCKernel>(program.second).cleanup();
    }

    ARM_COMPUTE_GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
    ARM_COMPUTE_GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    ARM_COMPUTE_GL_CHECK(glDeleteTextures(1, &_tex_rt));
    ARM_COMPUTE_GL_CHECK(glDeleteFramebuffers(1, &_frame_buffer));
}

std::string GCKernelLibrary::stringify_set(const StringSet &s) const
{
    std::string concat_set;

    // Concatenate set
    for(const auto &el : s)
    {
        concat_set += el + "\n";
    }

    return concat_set;
}
