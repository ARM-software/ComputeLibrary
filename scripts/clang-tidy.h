#include <arm_neon.h>

inline float16x8_t vcvtq_f16_u16(uint16x8_t)
{
  return vdupq_n_f16(0);
}

inline uint16x8_t vcvtq_u16_f16(float16x8_t)
{
  return vdupq_n_u16(0);
}

inline int16x8_t vcvtq_s16_f16(float16x8_t)
{
  return vdupq_n_s16(0);
}

inline float16x8_t vaddq_f16(float16x8_t, float16x8_t)
{
  return vdupq_n_f16(0);
}

inline float16x8_t vsubq_f16(float16x8_t, float16x8_t)
{
  return vdupq_n_f16(0);
}

inline float16x8_t vmulq_f16(float16x8_t, float16x8_t)
{
  return vdupq_n_f16(0);
}

inline float16x8_t vmulq_n_f16(float16x8_t, float16_t)
{
  return vdupq_n_f16(0);
}

inline float16x8_t vfmaq_f16(float16x8_t, float16x8_t, float16x8_t)
{
  return vdupq_n_f16(0);
}

inline uint16x8_t vcgeq_f16(float16x8_t, float16x8_t)
{
  return vdupq_n_u16(0);
}

inline uint16x8_t vcgtq_f16(float16x8_t, float16x8_t)
{
  return vdupq_n_u16(0);
}

inline float16x8_t vbslq_f16 (uint16x8_t, float16x8_t, float16x8_t)
{
  return vdupq_n_f16(0);;
}

inline float16x8_t vextq_f16(float16x8_t, float16x8_t, int)
{
  return vdupq_n_f16(0);
}

inline float16x8_t vabsq_f16(float16x8_t)
{
  return vdupq_n_f16(0);
}

inline uint16x8_t vcvtq_f16_s16(float16x8_t)
{
  return vdupq_n_s16(0);
}

inline float16x4_t vbsl_f16 (uint16x4_t,float16x4_t, float16x4_t)
{
  return vdup_n_f16(0);
}

inline float16x8_t vrsqrteq_f16(float16x8_t)
{
   return vdupq_n_f16(0);
}

inline float16x8_t vfmsq_f16 (float16x8_t, float16x8_t, float16x8_t)
{
   return vdupq_n_f16(0);
}

inline float16x8_t vrecpeq_f16 (float16x8_t)
{
   return vdupq_n_f16(0);
}

inline float16x8_t vrecpsq_f16 (float16x8_t, float16x8_t)
{
   return vdupq_n_f16(0);
}

inline float16x8_t vmaxq_f16 (float16x8_t, float16x8_t)
{
   return vdupq_n_f16(0);
}

inline float16x8_t vminq_f16 (float16x8_t, float16x8_t)
{
   return vdupq_n_f16(0);
}

inline uint16x8_t vcltq_f16(float16x8_t, float16x8_t)
{
   return vdupq_n_u16(0);
}

