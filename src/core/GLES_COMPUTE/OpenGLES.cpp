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

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"

#include <dlfcn.h>
#include <iostream>
#include <vector>

using eglGetProcAddress_func         = __eglMustCastToProperFunctionPointerType EGLAPIENTRY (*)(const char *procname);
using eglBindAPI_func                = EGLBoolean EGLAPIENTRY (*)(EGLenum api);
using eglChooseConfig_func           = EGLBoolean EGLAPIENTRY (*)(EGLDisplay dpy, const EGLint *attrib_list, EGLConfig *configs, EGLint config_size, EGLint *num_config);
using eglCreateContext_func          = EGLContext EGLAPIENTRY (*)(EGLDisplay dpy, EGLConfig config, EGLContext share_context, const EGLint *attrib_list);
using eglDestroyContext_func         = EGLBoolean EGLAPIENTRY (*)(EGLDisplay dpy, EGLContext ctx);
using eglGetDisplay_func             = EGLDisplay EGLAPIENTRY (*)(EGLNativeDisplayType display_id);
using eglInitialize_func             = EGLBoolean EGLAPIENTRY (*)(EGLDisplay dpy, EGLint *major, EGLint *minor);
using eglMakeCurrent_func            = EGLBoolean EGLAPIENTRY (*)(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx);
using eglTerminate_func              = EGLBoolean EGLAPIENTRY (*)(EGLDisplay dpy);
using eglGetError_func               = EGLint         EGLAPIENTRY (*)();
using eglQueryString_func            = char const * EGLAPIENTRY (*)(EGLDisplay dpy, EGLint name);
using glAttachShader_func            = void GL_APIENTRY (*)(GLuint program, GLuint shader);
using glCompileShader_func           = void GL_APIENTRY (*)(GLuint shader);
using glCreateProgram_func           = GLuint GL_APIENTRY (*)();
using glCreateShader_func            = GLuint GL_APIENTRY (*)(GLenum type);
using glDeleteProgram_func           = void GL_APIENTRY (*)(GLuint program);
using glDeleteShader_func            = void GL_APIENTRY (*)(GLuint shader);
using glDetachShader_func            = void GL_APIENTRY (*)(GLuint program, GLuint shader);
using glGetProgramInfoLog_func       = void GL_APIENTRY (*)(GLuint program, GLsizei bufsize, GLsizei *length, GLchar *infolog);
using glGetProgramiv_func            = void GL_APIENTRY (*)(GLuint program, GLenum pname, GLint *params);
using glGetShaderInfoLog_func        = void GL_APIENTRY (*)(GLuint shader, GLsizei bufsize, GLsizei *length, GLchar *infolog);
using glGetShaderiv_func             = void GL_APIENTRY (*)(GLuint shader, GLenum pname, GLint *params);
using glLinkProgram_func             = void GL_APIENTRY (*)(GLuint program);
using glShaderSource_func            = void GL_APIENTRY (*)(GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length);
using glUseProgram_func              = void GL_APIENTRY (*)(GLuint program);
using glBindBuffer_func              = void GL_APIENTRY (*)(GLenum target, GLuint buffer);
using glBindBufferBase_func          = void GL_APIENTRY (*)(GLenum target, GLuint index, GLuint buffer);
using glBufferData_func              = void GL_APIENTRY (*)(GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage);
using glDeleteBuffers_func           = void GL_APIENTRY (*)(GLsizei n, const GLuint *buffers);
using glDispatchCompute_func         = void GL_APIENTRY (*)(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
using glFlush_func                   = void      GL_APIENTRY (*)();
using glGenBuffers_func              = void GL_APIENTRY (*)(GLsizei n, GLuint *buffers);
using glGetProgramResourceIndex_func = GLuint GL_APIENTRY (*)(GLuint program, GLenum programInterface, const GLchar *name);
using glGetUniformLocation_func      = GLint GL_APIENTRY (*)(GLuint program, const GLchar *name);
using glMapBufferRange_func          = void *GL_APIENTRY (*)(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
using glMemoryBarrier_func           = void GL_APIENTRY (*)(GLbitfield barriers);
using glUniform1ui_func              = void GL_APIENTRY (*)(GLint location, GLuint v0);
using glUnmapBuffer_func             = GLboolean GL_APIENTRY (*)(GLenum target);
using glGetError_func                = GLenum              GL_APIENTRY (*)();
using glGetActiveUniformBlockiv_func = void GL_APIENTRY (*)(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params);
using glUniformBlockBinding_func     = void GL_APIENTRY (*)(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
using glGetUniformBlockIndex_func    = GLuint GL_APIENTRY (*)(GLuint program, const GLchar *uniformBlockName);
using glGenTextures_func             = void GL_APIENTRY (*)(GLsizei n, GLuint *textures);
using glDeleteTextures_func          = void GL_APIENTRY (*)(GLsizei n, const GLuint *textures);
using glBindTexture_func             = void GL_APIENTRY (*)(GLenum target, GLuint texture);
using glTexImage2D_func              = void GL_APIENTRY (*)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type,
                                                            const GLvoid *pixels);
using glGenFramebuffers_func      = void GL_APIENTRY (*)(GLsizei n, GLuint *framebuffers);
using glDeleteFramebuffers_func   = void GL_APIENTRY (*)(GLsizei n, const GLuint *framebuffers);
using glBindFramebuffer_func      = void GL_APIENTRY (*)(GLenum target, GLuint framebuffer);
using glFramebufferTexture2D_func = void GL_APIENTRY (*)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);

class GLESSymbols
{
private:
    void init()
    {
        void *egl_handle    = dlopen("libEGL.so", RTLD_LAZY | RTLD_LOCAL);
        void *glesv2_handle = dlopen("libGLESv2.so", RTLD_LAZY | RTLD_LOCAL);
        void *glesv3_handle = dlopen("libGLESv3.so", RTLD_LAZY | RTLD_LOCAL);
        if(egl_handle == nullptr)
        {
            std::cerr << "Can't load libEGL.so: " << dlerror() << std::endl;
        }
        else
        {
#undef EGL_ENTRY
#define EGL_ENTRY(_api) _api = reinterpret_cast<_api##_func>(dlsym(egl_handle, #_api));
#include "./egl_entries.in"
#undef EGL_ENTRY

            if(eglGetProcAddress != nullptr)
            {
#undef EGL_ENTRY
#define EGL_ENTRY(_api)   \
    if((_api) == nullptr) \
        (_api) = reinterpret_cast<_api##_func>(eglGetProcAddress(#_api));
#include "./egl_entries.in"
#undef EGL_ENTRY

#undef GL_ENTRY
#define GL_ENTRY(_api) _api = reinterpret_cast<_api##_func>(eglGetProcAddress(#_api));
#include "./gl_entries.in"
#undef GL_ENTRY
            }

            std::vector<void *> handles = { glesv3_handle, glesv2_handle };
            for(auto &handle : handles)
            {
                if(handle != nullptr)
                {
#undef GL_ENTRY
#define GL_ENTRY(_api)    \
    if((_api) == nullptr) \
        (_api) = reinterpret_cast<_api##_func>(dlsym(handle, #_api));
#include "./gl_entries.in"
#undef GL_ENTRY
                }
            }

            //Don't call dlclose(handle) or all the symbols will be unloaded !
        }
    }
    bool _initialized = false;

public:
    static GLESSymbols &get()
    {
        static GLESSymbols symbols = GLESSymbols();
        if(!symbols._initialized)
        {
            symbols._initialized = true;
            symbols.init();
        }

        return symbols;
    }

#undef EGL_ENTRY
#undef GL_ENTRY
#define EGL_ENTRY(_api) _api##_func _api = nullptr;
#define GL_ENTRY(_api) EGL_ENTRY(_api)
#include "./egl_entries.in"
#include "./gl_entries.in"
#undef EGL_ENTRY
#undef GL_ENTRY
};

bool arm_compute::opengles31_is_available()
{
    return GLESSymbols::get().glDispatchCompute != nullptr;
}

__eglMustCastToProperFunctionPointerType EGLAPIENTRY eglGetProcAddress(const char *procname)
{
    auto func = GLESSymbols::get().eglGetProcAddress;
    if(func != nullptr)
    {
        return func(procname);
    }
    else
    {
        return nullptr;
    }
}

EGLBoolean EGLAPIENTRY eglBindAPI(EGLenum api)
{
    auto func = GLESSymbols::get().eglBindAPI;
    if(func != nullptr)
    {
        return func(api);
    }
    else
    {
        return EGL_FALSE;
    }
}

EGLBoolean EGLAPIENTRY eglChooseConfig(EGLDisplay dpy, const EGLint *attrib_list, EGLConfig *configs, EGLint config_size, EGLint *num_config)
{
    auto func = GLESSymbols::get().eglChooseConfig;
    if(func != nullptr)
    {
        return func(dpy, attrib_list, configs, config_size, num_config);
    }
    else
    {
        return EGL_FALSE;
    }
}

EGLContext EGLAPIENTRY eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share_context, const EGLint *attrib_list)
{
    auto func = GLESSymbols::get().eglCreateContext;
    if(func != nullptr)
    {
        return func(dpy, config, share_context, attrib_list);
    }
    else
    {
        return nullptr;
    }
}

EGLBoolean EGLAPIENTRY eglDestroyContext(EGLDisplay dpy, EGLContext ctx)
{
    auto func = GLESSymbols::get().eglDestroyContext;
    if(func != nullptr)
    {
        return func(dpy, ctx);
    }
    else
    {
        return EGL_FALSE;
    }
}

EGLDisplay EGLAPIENTRY eglGetDisplay(EGLNativeDisplayType display_id)
{
    auto func = GLESSymbols::get().eglGetDisplay;
    if(func != nullptr)
    {
        return func(display_id);
    }
    else
    {
        return nullptr;
    }
}

EGLBoolean EGLAPIENTRY eglInitialize(EGLDisplay dpy, EGLint *major, EGLint *minor)
{
    auto func = GLESSymbols::get().eglInitialize;
    if(func != nullptr)
    {
        return func(dpy, major, minor);
    }
    else
    {
        return EGL_FALSE;
    }
}

EGLBoolean EGLAPIENTRY eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx)
{
    auto func = GLESSymbols::get().eglMakeCurrent;
    if(func != nullptr)
    {
        return func(dpy, draw, read, ctx);
    }
    else
    {
        return EGL_FALSE;
    }
}

EGLBoolean EGLAPIENTRY eglTerminate(EGLDisplay dpy)
{
    auto func = GLESSymbols::get().eglTerminate;
    if(func != nullptr)
    {
        return func(dpy);
    }
    else
    {
        return EGL_FALSE;
    }
}

EGLint EGLAPIENTRY eglGetError()
{
    auto func = GLESSymbols::get().eglGetError;
    if(func != nullptr)
    {
        return func();
    }
    else
    {
        return GL_NO_ERROR;
    }
}

char const *EGLAPIENTRY eglQueryString(EGLDisplay dpy, EGLint name)
{
    auto func = GLESSymbols::get().eglQueryString;
    if(func != nullptr)
    {
        return func(dpy, name);
    }
    else
    {
        return nullptr;
    }
}

void GL_APIENTRY glAttachShader(GLuint program, GLuint shader)
{
    auto func = GLESSymbols::get().glAttachShader;
    if(func != nullptr)
    {
        return func(program, shader);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glCompileShader(GLuint shader)
{
    auto func = GLESSymbols::get().glCompileShader;
    if(func != nullptr)
    {
        return func(shader);
    }
    else
    {
        return;
    }
}

GLuint GL_APIENTRY glCreateProgram()
{
    auto func = GLESSymbols::get().glCreateProgram;
    if(func != nullptr)
    {
        return func();
    }
    else
    {
        return 0;
    }
}

GLuint GL_APIENTRY glCreateShader(GLenum type)
{
    auto func = GLESSymbols::get().glCreateShader;
    if(func != nullptr)
    {
        return func(type);
    }
    else
    {
        return 0;
    }
}

void GL_APIENTRY glDeleteProgram(GLuint program)
{
    auto func = GLESSymbols::get().glDeleteProgram;
    if(func != nullptr)
    {
        return func(program);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glDeleteShader(GLuint shader)
{
    auto func = GLESSymbols::get().glDeleteShader;
    if(func != nullptr)
    {
        return func(shader);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glDetachShader(GLuint program, GLuint shader)
{
    auto func = GLESSymbols::get().glDetachShader;
    if(func != nullptr)
    {
        return func(program, shader);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    auto func = GLESSymbols::get().glGetProgramInfoLog;
    if(func != nullptr)
    {
        return func(program, bufSize, length, infoLog);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glGetProgramiv(GLuint program, GLenum pname, GLint *params)
{
    auto func = GLESSymbols::get().glGetProgramiv;
    if(func != nullptr)
    {
        return func(program, pname, params);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    auto func = GLESSymbols::get().glGetShaderInfoLog;
    if(func != nullptr)
    {
        return func(shader, bufSize, length, infoLog);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glGetShaderiv(GLuint shader, GLenum pname, GLint *params)
{
    auto func = GLESSymbols::get().glGetShaderiv;
    if(func != nullptr)
    {
        return func(shader, pname, params);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glLinkProgram(GLuint program)
{
    auto func = GLESSymbols::get().glLinkProgram;
    if(func != nullptr)
    {
        return func(program);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glShaderSource(GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length)
{
    auto func = GLESSymbols::get().glShaderSource;
    if(func != nullptr)
    {
        return func(shader, count, string, length);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glUseProgram(GLuint program)
{
    auto func = GLESSymbols::get().glUseProgram;
    if(func != nullptr)
    {
        return func(program);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glBindBuffer(GLenum target, GLuint buffer)
{
    auto func = GLESSymbols::get().glBindBuffer;
    if(func != nullptr)
    {
        return func(target, buffer);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glBindBufferBase(GLenum target, GLuint index, GLuint buffer)
{
    auto func = GLESSymbols::get().glBindBufferBase;
    if(func != nullptr)
    {
        return func(target, index, buffer);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glBufferData(GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage)
{
    auto func = GLESSymbols::get().glBufferData;
    if(func != nullptr)
    {
        return func(target, size, data, usage);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glDeleteBuffers(GLsizei n, const GLuint *buffers)
{
    auto func = GLESSymbols::get().glDeleteBuffers;
    if(func != nullptr)
    {
        return func(n, buffers);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
    auto func = GLESSymbols::get().glDispatchCompute;
    if(func != nullptr)
    {
        return func(num_groups_x, num_groups_y, num_groups_z);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glFlush(void)
{
    auto func = GLESSymbols::get().glFlush;
    if(func != nullptr)
    {
        return func();
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glGenBuffers(GLsizei n, GLuint *buffers)
{
    auto func = GLESSymbols::get().glGenBuffers;
    if(func != nullptr)
    {
        return func(n, buffers);
    }
    else
    {
        return;
    }
}

GLuint GL_APIENTRY glGetProgramResourceIndex(GLuint program, GLenum programInterface, const GLchar *name)
{
    auto func = GLESSymbols::get().glGetProgramResourceIndex;
    if(func != nullptr)
    {
        return func(program, programInterface, name);
    }
    else
    {
        return GL_INVALID_INDEX;
    }
}

GLint GL_APIENTRY glGetUniformLocation(GLuint program, const GLchar *name)
{
    auto func = GLESSymbols::get().glGetUniformLocation;
    if(func != nullptr)
    {
        return func(program, name);
    }
    else
    {
        return -1;
    }
}

void *GL_APIENTRY glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    auto func = GLESSymbols::get().glMapBufferRange;
    if(func != nullptr)
    {
        return func(target, offset, length, access);
    }
    else
    {
        return nullptr;
    }
}

void GL_APIENTRY glMemoryBarrier(GLbitfield barriers)
{
    auto func = GLESSymbols::get().glMemoryBarrier;
    if(func != nullptr)
    {
        return func(barriers);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glUniform1ui(GLint location, GLuint v0)
{
    auto func = GLESSymbols::get().glUniform1ui;
    if(func != nullptr)
    {
        return func(location, v0);
    }
    else
    {
        return;
    }
}

GLboolean GL_APIENTRY glUnmapBuffer(GLenum target)
{
    auto func = GLESSymbols::get().glUnmapBuffer;
    if(func != nullptr)
    {
        return func(target);
    }
    else
    {
        return GL_FALSE;
    }
}

GLenum GL_APIENTRY glGetError(void)
{
    auto func = GLESSymbols::get().glGetError;
    if(func != nullptr)
    {
        return func();
    }
    else
    {
        return GL_NO_ERROR;
    }
}

void GL_APIENTRY glGetActiveUniformBlockiv(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params)
{
    auto func = GLESSymbols::get().glGetActiveUniformBlockiv;
    if(func != nullptr)
    {
        return func(program, uniformBlockIndex, pname, params);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glUniformBlockBinding(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding)
{
    auto func = GLESSymbols::get().glUniformBlockBinding;
    if(func != nullptr)
    {
        return func(program, uniformBlockIndex, uniformBlockBinding);
    }
    else
    {
        return;
    }
}

GLuint GL_APIENTRY glGetUniformBlockIndex(GLuint program, const GLchar *uniformBlockName)
{
    auto func = GLESSymbols::get().glGetUniformBlockIndex;
    if(func != nullptr)
    {
        return func(program, uniformBlockName);
    }
    else
    {
        return GL_INVALID_INDEX;
    }
}

void GL_APIENTRY glGenTextures(GLsizei n, GLuint *textures)
{
    auto func = GLESSymbols::get().glGenTextures;
    if(func != nullptr)
    {
        return func(n, textures);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glDeleteTextures(GLsizei n, const GLuint *textures)
{
    auto func = GLESSymbols::get().glDeleteTextures;
    if(func != nullptr)
    {
        return func(n, textures);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glBindTexture(GLenum target, GLuint texture)
{
    auto func = GLESSymbols::get().glBindTexture;
    if(func != nullptr)
    {
        return func(target, texture);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels)
{
    auto func = GLESSymbols::get().glTexImage2D;
    if(func != nullptr)
    {
        return func(target, level, internalformat, width, height, border, format, type, pixels);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glGenFramebuffers(GLsizei n, GLuint *framebuffers)
{
    auto func = GLESSymbols::get().glGenFramebuffers;
    if(func != nullptr)
    {
        return func(n, framebuffers);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers)
{
    auto func = GLESSymbols::get().glDeleteFramebuffers;
    if(func != nullptr)
    {
        return func(n, framebuffers);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glBindFramebuffer(GLenum target, GLuint framebuffer)
{
    auto func = GLESSymbols::get().glBindFramebuffer;
    if(func != nullptr)
    {
        return func(target, framebuffer);
    }
    else
    {
        return;
    }
}

void GL_APIENTRY glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    auto func = GLESSymbols::get().glFramebufferTexture2D;
    if(func != nullptr)
    {
        return func(target, attachment, textarget, texture, level);
    }
    else
    {
        return;
    }
}
