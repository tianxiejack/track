#ifndef NEON_STD_H
#define NEON_STD_H

#include "arm_neon.h"

static inline float sum_float_neon(float *arr, int len)
{
	int dim4 = (len >> 2);
	int left4 = (len & 3);
	float sum;

	float32x4_t sum_vec = vdupq_n_f32(0.0f);

	for(; dim4>0; dim4--, arr+=4){
		float32x4_t d_vec = vld1q_f32(arr);
		sum_vec = vaddq_f32(sum_vec, d_vec);
	}

	sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) + vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);

	for(; left4>0; left4--, arr++)
		sum += (*arr);

	return sum;
}

static inline void add_n_float_neon(float *result, float *arr, float alph, int len)
{
	int dim4 = (len >> 2);
	int left4 = (len & 3);

	float32x4_t alph_vec = vdupq_n_f32(alph);

	for(; dim4>0; dim4--, arr+=4, result+=4){
		float32x4_t tmp = vaddq_f32(vld1q_f32(arr), alph_vec);
		vst1q_f32(result, tmp);
	}

	for(; left4>0; left4--, arr++, result++)
		*result = (*arr)+alph;
}

static inline void mul_n_float_neon(float *result, float *arr, float alph, int len)
{
	int dim4 = (len >> 2);
	int left4 = (len & 3);

	for(; dim4>0; dim4--, arr+=4, result+=4){
		float32x4_t tmp = vmulq_n_f32(vld1q_f32(arr), alph);
		vst1q_f32(result, tmp);
	}

	for(; left4>0; left4--, arr++, result++)
		*result = (*arr)*alph;
}

static inline void add_float_neon(float *result, float *arr1, float *arr2, int len)
{
	int dim4 = (len >> 2);
	int left4 = (len & 3);

	for(; dim4>0; dim4--, arr1+=4, arr2+=4, result+=4){
		float32x4_t tmp = vaddq_f32(vld1q_f32(arr1), vld1q_f32(arr2));
		vst1q_f32(result, tmp);
	}

	for(; left4>0; left4--, arr1++, arr2++, result++)
		*result = (*arr1) + (*arr2);
}

static inline void sub_float_neon(float *result, float *arr1, float *arr2, int len)
{
	int dim4 = (len >> 2);
	int left4 = (len & 3);

	for(; dim4>0; dim4--, arr1+=4, arr2+=4, result+=4){
		float32x4_t tmp = vsubq_f32(vld1q_f32(arr1), vld1q_f32(arr2));
		vst1q_f32(result, tmp);
	}

	for(; left4>0; left4--, arr1++, arr2++, result++)
		*result = (*arr1) - (*arr2);
}

static inline void mul_float_neon(float *result, float *arr1, float *arr2, int len)
{
	int dim4 = (len >> 2);
	int left4 = (len & 3);

	for(; dim4>0; dim4--, arr1+=4, arr2+=4, result+=4){
		float32x4_t tmp = vmulq_f32(vld1q_f32(arr1), vld1q_f32(arr2));
		vst1q_f32(result, tmp);
	}

	for(; left4>0; left4--, arr1++, arr2++, result++)
		*result = (*arr1) * (*arr2);
}

static inline void sub_u8_float_neon(float *result, unsigned char *arr1, unsigned char *arr2, int len)
{
	int dim8 = (len >> 3);
	int left8 = (len & 7);

	for(; dim8>0; dim8--, arr1+=8, arr2+=8, result+=8){
		uint16x8_t v1_16 = vmovl_u8(vld1_u8(arr1));
		uint16x8_t v2_16 = vmovl_u8(vld1_u8(arr2));
		uint32x4_t v1_u32_low = vmovl_u16(vget_low_u16(v1_16));
		uint32x4_t v1_u32_high = vmovl_u16(vget_high_u16(v1_16));
		uint32x4_t v2_u32_low = vmovl_u16(vget_low_u16(v2_16));
		uint32x4_t v2_u32_high = vmovl_u16(vget_high_u16(v2_16));
		float32x4_t tmp = vsubq_f32(vcvtq_f32_u32(v1_u32_low), vcvtq_f32_u32(v2_u32_low));
		vst1q_f32(result, tmp);
		tmp = vsubq_f32(vcvtq_f32_u32(v1_u32_high), vcvtq_f32_u32(v2_u32_high));
		vst1q_f32(result+4, tmp);
	}

	for(; left8>0; left8--, arr1++, arr2++, result++)
		*result = (*arr1) - (*arr2);
}


#endif

