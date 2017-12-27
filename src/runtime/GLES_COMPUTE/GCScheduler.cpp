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

#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"

#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"

using namespace arm_compute;

std::once_flag GCScheduler::_initialize_symbols;

GCScheduler::GCScheduler()
    : _display(EGL_NO_DISPLAY), _context(EGL_NO_CONTEXT)
{
}

GCScheduler::~GCScheduler()
{
    eglDestroyContext(_display, _context);
    eglTerminate(_display);

    _context = EGL_NO_CONTEXT;
    _display = EGL_NO_DISPLAY;
}

void GCScheduler::default_init()
{
    setup_context();

    GCKernelLibrary::get().init("./cs_shaders/", _display, _context);
}

void GCScheduler::init(EGLDisplay dpy, EGLContext ctx)
{
    GCKernelLibrary::get().init("./cs_shaders/", dpy, ctx);
}

GCScheduler &GCScheduler::get()
{
    std::call_once(_initialize_symbols, opengles31_is_available);
    static GCScheduler scheduler;
    return scheduler;
}

void GCScheduler::dispatch(IGCKernel &kernel, bool flush)
{
    kernel.run(kernel.window());
    if(flush)
    {
        ARM_COMPUTE_GL_CHECK(glFlush());
    }
}

void GCScheduler::memory_barrier()
{
    ARM_COMPUTE_GL_CHECK(glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT));
}

void GCScheduler::setup_context()
{
    EGLBoolean res;
    _display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    ARM_COMPUTE_ERROR_ON_MSG(_display == EGL_NO_DISPLAY, "Failed to get display: 0x%x.", eglGetError());

    res = eglInitialize(_display, nullptr, nullptr);

    ARM_COMPUTE_ERROR_ON_MSG(res == EGL_FALSE, "Failed to initialize egl: 0x%x.", eglGetError());
    ARM_COMPUTE_UNUSED(res);

    const char *egl_extension_st = eglQueryString(_display, EGL_EXTENSIONS);
    ARM_COMPUTE_ERROR_ON_MSG((strstr(egl_extension_st, "EGL_KHR_create_context") == nullptr), "Failed to query EGL_KHR_create_context");
    ARM_COMPUTE_ERROR_ON_MSG((strstr(egl_extension_st, "EGL_KHR_surfaceless_context") == nullptr), "Failed to query EGL_KHR_surfaceless_context");
    ARM_COMPUTE_UNUSED(egl_extension_st);

    const EGLint config_attribs[] =
    {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_NONE
    };
    EGLConfig cfg;
    EGLint    count;

    res = eglChooseConfig(_display, config_attribs, &cfg, 1, &count);

    ARM_COMPUTE_ERROR_ON_MSG(res == EGL_FALSE, "Failed to choose config: 0x%x.", eglGetError());
    ARM_COMPUTE_UNUSED(res);

    res = eglBindAPI(EGL_OPENGL_ES_API);

    ARM_COMPUTE_ERROR_ON_MSG(res == EGL_FALSE, "Failed to bind api: 0x%x.", eglGetError());

    const EGLint attribs[] =
    {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };
    _context = eglCreateContext(_display,
                                cfg,
                                EGL_NO_CONTEXT,
                                attribs);

    ARM_COMPUTE_ERROR_ON_MSG(_context == EGL_NO_CONTEXT, "Failed to create context: 0x%x.", eglGetError());
    ARM_COMPUTE_UNUSED(res);

    res = eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, _context);

    ARM_COMPUTE_ERROR_ON_MSG(res == EGL_FALSE, "Failed to make current: 0x%x.", eglGetError());
    ARM_COMPUTE_UNUSED(res);
}
