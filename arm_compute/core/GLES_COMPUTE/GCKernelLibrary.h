/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GCKERNELLIBRARY_H
#define ARM_COMPUTE_GCKERNELLIBRARY_H

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Utils.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace arm_compute
{
/** GCProgram class */
class GCProgram final
{
public:
    /** Default constructor. */
    GCProgram();
    /** Construct program from source file.
     *
     * @param[in] name   Program name.
     * @param[in] source Program source.
     */
    GCProgram(std::string name, std::string source);
    /** Default Copy Constructor. */
    GCProgram(const GCProgram &) = default;
    /** Default Move Constructor. */
    GCProgram(GCProgram &&) = default;
    /** Default copy assignment operator */
    GCProgram &operator=(const GCProgram &) = default;
    /** Default move assignment operator */
    GCProgram &operator=(GCProgram &&) = default;
    /** Returns program name.
     *
     * @return Program's name.
     */
    std::string name() const
    {
        return _name;
    }
    /** Link program.
     *
     * @param[in] shader Shader used to link program.
     *
     * @return linked program id .
     */
    GLuint link_program(GLuint shader);
    /** Compile shader.
     *
     * @param[in] build_options Shader build options.
     *
     * @return GLES shader object.
     */
    GLuint compile_shader(const std::string &build_options);

private:
    std::string _name;   /**< Program name. */
    std::string _source; /**< Source code for the program. */
};

/** GCKernel class */
class GCKernel final
{
public:
    /** Default Constructor. */
    GCKernel();
    /** Default Copy Constructor. */
    GCKernel(const GCKernel &) = default;
    /** Default Move Constructor. */
    GCKernel(GCKernel &&) = default;
    /** Default copy assignment operator */
    GCKernel &operator=(const GCKernel &) = default;
    /** Default move assignment operator */
    GCKernel &operator=(GCKernel &&) = default;
    /** Constructor.
     *
     * @param[in] name    Kernel name.
     * @param[in] program Built program.
     */
    GCKernel(std::string name, GLuint program);
    /** Destructor.
     */
    ~GCKernel();
    /** Returns kernel name.
     *
     * @return Kernel's name.
     */
    std::string name() const
    {
        return _name;
    }
    /** Get program id.
     *
     * @return program id.
     */
    GLuint get_program() const
    {
        return _program;
    }
    /** Use current program.
     *
     * @return program id.
     */
    void use();
    /** Unuse current program.
     *
     * @return program id.
     */
    void unuse();
    /** Set argument value at index of shader params.
     *
     * @param[in] idx   Index in shader params.
     * @param[in] value Argument value to be set.
     */
    template <class T>
    void set_argument(unsigned int idx, T value)
    {
        if(idx >= _shader_arguments.size())
        {
            _shader_arguments.resize(idx + 1, 0);
        }

        unsigned int *p        = reinterpret_cast<unsigned int *>(&value);
        _shader_arguments[idx] = *p;
    }
    /** Clear shader arguments.
     *
     */
    void clear_arguments()
    {
        _shader_arguments.clear();
    }
    /** Set shader params binding point.
     *
     * @param[in] binding Shader params binding point.
     */
    void set_shader_params_binding_point(unsigned int binding)
    {
        _shader_params_binding_point = binding;
    }
    /** Update shader params.
     *
     */
    void update_shader_params();
    /** Clean up program and ubo.
     *
     */
    void cleanup();

private:
    std::string                  _name;                                 /**< Kernel name */
    GLuint                       _program;                              /**< Linked program id */
    std::vector<unsigned int>    _shader_arguments;                     /**< Store all the values of the shader arguments */
    GLuint                       _shader_params_ubo_name;               /**< Uniform buffer object name for shader parameters */
    GLuint                       _shader_params_binding_point;          /**< The binding point of the uniform block for shader parameters */
    GLuint                       _shader_params_index;                  /**< The index of the uniform block */
    GLint                        _shader_params_size;                   /**< The uniform block data size in the shader */
    static constexpr const char *_shader_params_name = "shader_params"; /**< The uniform block name in the shader */
};

/** GCKernelLibrary class */
class GCKernelLibrary final
{
    using StringSet = std::set<std::string>;

public:
    /** Default Constructor. */
    GCKernelLibrary();
    /** Default Destructor */
    ~GCKernelLibrary();
    /** Prevent instances of this class from being copied */
    GCKernelLibrary(const GCKernelLibrary &) = delete;
    /** Prevent instances of this class from being copied */
    const GCKernelLibrary &operator=(const GCKernelLibrary &) = delete;
    /** Get the static instance of @ref GCKernelLibrary.
     * This method has been deprecated and will be removed in future releases.
     * @return The static instance of GCKernelLibrary.
     */
    static GCKernelLibrary &get();
    /** Initialises the kernel library.
     *
     * @param[in] shader_path (Optional) Path of the directory from which shader sources are loaded.
     * @param[in] dpy         (Optional) EGLdisplay set by external application.
     * @param[in] ctx         (Optional) EGLContext set by external application.
     */
    void init(std::string shader_path = "./", EGLDisplay dpy = EGL_NO_DISPLAY, EGLContext ctx = EGL_NO_CONTEXT);
    /** Sets the path that the shaders reside in.
     *
     * @param[in] shader_path Path of the shader.
     */
    void set_shader_path(const std::string &shader_path);
    /** Sets display and context to create kernel.
     *
     * @param[in] dpy EGLdisplay set by external application.
     * @param[in] ctx EGLContext set by external application.
     */
    void set_context(EGLDisplay dpy, EGLContext ctx);
    /** Creates a kernel from the kernel library.
     *
     * @param[in] shader_name       Shader name.
     * @param[in] build_options_set Shader build options as a set.
     *
     * @return The created kernel.
     */
    GCKernel create_kernel(const std::string &shader_name, const StringSet &build_options_set = {}) const;
    /** Serializes and saves programs to a binary. */
    void save_binary();
    /** Load serialized binary with all the programs. */
    void load_binary();
    /** Setup a dummy fbo to workaround an issue on Galaxy S8. */
    void setup_dummy_fbo();

private:
    /** Preprocess GLES shader
     *
     * @param[in] shader_source Source code of the shader to preprocess.
     *
     * @return Preprocessed GLES shader object.
     */
    std::string preprocess_shader(const std::string &shader_source) const;
    /** Load program and its dependencies.
     *
     * @param[in] program_name Name of the program to load.
     */
    const GCProgram &load_program(const std::string &program_name) const;
    /** Concatenates contents of a set into a single string.
     *
     * @param[in] s Input set to concatenate.
     *
     * @return Concatenated string.
     */
    std::string stringify_set(const StringSet &s) const;

    EGLDisplay  _display;                                                /**< Underlying EGL Display. */
    EGLContext  _context;                                                /**< Underlying EGL Context. */
    GLuint      _frame_buffer;                                           /**< Dummy fbo */
    GLuint      _tex_rt;                                                 /**< Dummy texture for render target */
    std::string _shader_path;                                            /**< Path to the shaders folder. */
    mutable std::map<std::string, const GCProgram>  _programs_map;       /**< Map with all already loaded program data. */
    mutable std::map<std::string, const GCKernel>   _built_programs_map; /**< Map with all already built program data. */
    static const std::map<std::string, std::string> _shader_program_map; /**< Map that associates kernel names with programs. */
    static const std::map<std::string, std::string> _program_source_map; /**< Contains sources for all programs.
                                                                              Used for compile-time shader inclusion. */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_GCKERNELLIBRARY_H */
