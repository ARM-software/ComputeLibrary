#include <stdio.h>

#define PRINT_STUB_ERROR printf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nERROR: %s from stub libEGL.so library called! This library can be used to resolve OpenGL ES symbols at compile time but must *not* be in your runtime path (You need to use a real OpenGL ES implementation, this one is empty)\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", __func__)

void eglBindAPI(void) { PRINT_STUB_ERROR; return; }
void eglBindTexImage(void) { PRINT_STUB_ERROR; return; }
void eglChooseConfig(void) { PRINT_STUB_ERROR; return; }
void eglCopyBuffers(void) { PRINT_STUB_ERROR; return; }
void eglCreateContext(void) { PRINT_STUB_ERROR; return; }
void eglCreateImageKHR (void) { PRINT_STUB_ERROR; return; }
void eglCreatePbufferFromClientBuffer(void) { PRINT_STUB_ERROR; return; }
void eglCreatePbufferSurface(void) { PRINT_STUB_ERROR; return; }
void eglCreatePixmapSurface(void) { PRINT_STUB_ERROR; return; }
void eglCreateWindowSurface(void) { PRINT_STUB_ERROR; return; }
void eglDestroyContext(void) { PRINT_STUB_ERROR; return; }
void eglDestroyImageKHR (void) { PRINT_STUB_ERROR; return; }
void eglDestroySurface(void) { PRINT_STUB_ERROR; return; }
void eglGetConfigAttrib(void) { PRINT_STUB_ERROR; return; }
void eglGetConfigs(void) { PRINT_STUB_ERROR; return; }
void eglGetCurrentContext(void) { PRINT_STUB_ERROR; return; }
void eglGetCurrentDisplay(void) { PRINT_STUB_ERROR; return; }
void eglGetCurrentSurface(void) { PRINT_STUB_ERROR; return; }
void eglGetDisplay(void) { PRINT_STUB_ERROR; return; }
void eglGetError(void) { PRINT_STUB_ERROR; return; }
void eglGetProcAddress(void) { PRINT_STUB_ERROR; return; }
void eglInitialize(void) { PRINT_STUB_ERROR; return; }
void eglMakeCurrent(void) { PRINT_STUB_ERROR; return; }
void eglQueryAPI(void) { PRINT_STUB_ERROR; return; }
void eglQueryContext(void) { PRINT_STUB_ERROR; return; }
void eglQueryString(void) { PRINT_STUB_ERROR; return; }
void eglQuerySurface(void) { PRINT_STUB_ERROR; return; }
void eglReleaseTexImage(void) { PRINT_STUB_ERROR; return; }
void eglReleaseThread(void) { PRINT_STUB_ERROR; return; }
void eglSurfaceAttrib(void) { PRINT_STUB_ERROR; return; }
void eglSwapBuffers(void) { PRINT_STUB_ERROR; return; }
void eglSwapInterval(void) { PRINT_STUB_ERROR; return; }
void eglTerminate(void) { PRINT_STUB_ERROR; return; }
void eglWaitClient(void) { PRINT_STUB_ERROR; return; }
void eglWaitGL(void) { PRINT_STUB_ERROR; return; }
void eglWaitNative(void) { PRINT_STUB_ERROR; return; }
