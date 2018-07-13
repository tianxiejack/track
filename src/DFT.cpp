#include "PCTracker.h"
#include "DFT.h"
#include "malloc_align.h"
#include "SSE2NEON.h"

extern void* cvAlignPtr( const void* ptr, int align);
extern int	DFTFactorize( int n, int* factors );
extern void	DFTInit( int n0, int nf, int* factors, int* itab, int elem_size, void* _wave, int inv_itab );

typedef void (*DFTFunc)(
	const void* src, void* dst, int n, int nf, int* factors,
	const int* itab, const void* wave, int tab_size,
	void* buf, int inv, double scale );

enum { CR_DFT_NO_PERMUTE=256,	CR_DFT_COMPLEX_INPUT_OR_OUTPUT=512 };


#if 1	/*CV_SSE3*/

// optimized radix-4 transform
typedef union CR32suf
{
	int i;
	unsigned u;
	float f;
}CR32suf;

//返回一个__m128的寄存器，Sets the lower two single-precision, floating-point
//values with 64 bits of data loaded from the address p; the upper two values
//are passed through from a
//r0=*_P0, r1=*_P1, r2=_A2, r3=_A3
static __m128	_mm_loadl_pi(__m128 __A, __m64 const *__P)
{
	float32x2_t f32_h = vget_high_f32 ( __A);
	float32x2_t f32_l = vld1_f32((float32_t*)__P);//*__P;
	return  vcombine_f32 (f32_l, f32_h);
}
//返回一个__m128的寄存器，Sets the upper two single-precision, floating-point
//values with 64 bits of data loaded from the address p; the lower two values
//are passed through from a
//r0=_A0, r1=_A1, r2=*_P0, r3=*_P1
static __m128	_mm_loadh_pi(__m128 __A, __m64 const *__P)
{
	float32x2_t f32_l = vget_low_f32 ( __A);
	float32x2_t f32_h = vld1_f32((float32_t*)__P);//*__P;
	return  vcombine_f32 (f32_l, f32_h);
}
//返回一个__m128的寄存器，Moves the lower two single-precision, floating-point
//values of b to the upper two single-precision, floating-point values of the result
//r3=_B1, r2=_B0, r1=_A1, r0=_A0
static __m128 _mm_movelh_ps(__m128 __A, __m128 __B)
{
	float32x2_t f32_l = vget_low_f32 ( __A);
	float32x2_t f32_h = vget_low_f32 ( __B);
	return  vcombine_f32 (f32_l, f32_h);
}
//返回为空，Stores the lower two single-precision, floating-point values of a
	//to the address p, *_P0=_A0, *_P1=_A1
static void _mm_storel_pi(__m64 *__P, __m128 __A)
{
//	*__P = vget_low_f32 ( __A);
	vst1_f32 ((float32_t *) __P, vget_low_f32 ( __A));
}
//返回为空，Stores the upper two single-precision, floating-point values of a
//to the address p, *_P0=_A2, *_P1=_A3
static void _mm_storeh_pi(__m64 *__P, __m128 __A)
{
//	*__P = vget_high_f32 ( __A);
	vst1_f32 ((float32_t *) __P, vget_high_f32 ( __A));
}
//a=(a0, a1, a2, a3), 则r0=a0, r1=a0, r2=a2, r3=a2
static __m128 _mm_moveldup_ps(__m128 a)
{
	float __attribute__((aligned(16)))	f32[4];
	vst1q_lane_f32 (f32, a, 0);
	vst1q_lane_f32 (f32+1, a, 0);
	vst1q_lane_f32 (f32+2, a, 2);
	vst1q_lane_f32 (f32+3, a, 2);
	return vld1q_f32(f32);
}
//a=(a0, a1, a2, a3), 则r0=a1, r1=a1, r2=a3, r3=a3
static __m128 _mm_movehdup_ps(__m128 a)
{
	float __attribute__((aligned(16)))	f32[4];
	vst1q_lane_f32 (f32, a, 1);
	vst1q_lane_f32 (f32+1, a, 1);
	vst1q_lane_f32 (f32+2, a, 3);
	vst1q_lane_f32 (f32+3, a, 3);
	return vld1q_f32(f32);
}
//a=(a0, a1, a2, a3), b=(b0, b1, b2, b3)
//则r0=a0-b0, r1=a1+b1, r2=a2-b2, r3=a3+b3
static __m128 _mm_addsub_ps(__m128 a, __m128 b)
{
	float32_t		b0, b2;
	b0 = vgetq_lane_f32 ( b, 0);
	b2 = vgetq_lane_f32 ( b, 2);
	b0 = -b0;	b2= -b2;
	vsetq_lane_f32 (b0, b, 0);
	vsetq_lane_f32 (b2, b, 2);

	return vaddq_f32 ( a,  b);
}

// optimized radix-4 transform
static int DFT_VecR4(ComplexCR* dst, int N, int n0, int* _pDw0, const ComplexCR* wave)
{
	int n = 1, i, j, nx, dw, dw0 = *_pDw0;
	__m128 z = _mm_setzero_ps(), x02=z, x13=z, w01=z, w23=z, y01, y23, t0, t1,temp;
	CR32suf t; t.i = 0x80000000;
	__m128 neg0_mask = _mm_load_ss(&t.f);
	__m128 neg3_mask = _mm_shuffle_ps(neg0_mask, neg0_mask, _MM_SHUFFLE(0,1,2,3));

	for( ; n*4 <= N; )
	{
		nx = n;
		n *= 4;
		dw0 /= 4;

		for( i = 0; i < n0; i += n )
		{
			ComplexCR *v0, *v1;

			v0 = dst + i;
			v1 = v0 + nx*2;

			x02 = _mm_loadl_pi(x02, (const __m64*)&v0[0]);
			x13 = _mm_loadl_pi(x13, (const __m64*)&v0[nx]);
			x02 = _mm_loadh_pi(x02, (const __m64*)&v1[0]);
			x13 = _mm_loadh_pi(x13, (const __m64*)&v1[nx]);

			y01 = _mm_add_ps(x02, x13);
			y23 = _mm_sub_ps(x02, x13);
			t1 = _mm_xor_ps(_mm_shuffle_ps(y01, y23, _MM_SHUFFLE(2,3,3,2)), neg3_mask);

			t0 = _mm_movelh_ps(y01, y23);
			y01 = _mm_add_ps(t0, t1);
			y23 = _mm_sub_ps(t0, t1);

			_mm_storel_pi((__m64*)&v0[0], y01);
			_mm_storeh_pi((__m64*)&v0[nx], y01);
			_mm_storel_pi((__m64*)&v1[0], y23);
			_mm_storeh_pi((__m64*)&v1[nx], y23);

			for( j = 1, dw = dw0; j < nx; j++, dw += dw0 )
			{
				v0 = dst + i + j;
				v1 = v0 + nx*2;

				x13 = _mm_loadl_pi(x13, (const __m64*)&v0[nx]);
				w23 = _mm_loadl_pi(w23, (const __m64*)&wave[dw*2]);
				x13 = _mm_loadh_pi(x13, (const __m64*)&v1[nx]); // x1, x3 = r1 i1 r3 i3
				w23 = _mm_loadh_pi(w23, (const __m64*)&wave[dw*3]); // w2, w3 = wr2 wi2 wr3 wi3

				t0 = _mm_mul_ps(_mm_moveldup_ps(x13), w23);
				t1 = _mm_mul_ps(_mm_movehdup_ps(x13), _mm_shuffle_ps(w23, w23, _MM_SHUFFLE(2,3,0,1)));
				x13 = _mm_addsub_ps(t0, t1);
				// re(x1*w2), im(x1*w2), re(x3*w3), im(x3*w3)
				x02 = _mm_loadl_pi(x02, (const __m64*)&v1[0]); // x2 = r2 i2
				w01 = _mm_loadl_pi(w01, (const __m64*)&wave[dw]); // w1 = wr1 wi1
				x02 = _mm_shuffle_ps(x02, x02, _MM_SHUFFLE(0,0,1,1));
				w01 = _mm_shuffle_ps(w01, w01, _MM_SHUFFLE(1,0,0,1));
				x02 = _mm_mul_ps(x02, w01);
				x02 = _mm_addsub_ps(x02, _mm_movelh_ps(x02, x02));
				// re(x0) im(x0) re(x2*w1), im(x2*w1)
				x02 = _mm_loadl_pi(x02, (const __m64*)&v0[0]);

				y01 = _mm_add_ps(x02, x13);
				y23 = _mm_sub_ps(x02, x13);
				t1 = _mm_xor_ps(_mm_shuffle_ps(y01, y23, _MM_SHUFFLE(2,3,3,2)), neg3_mask);
				t0 = _mm_movelh_ps(y01, y23);
				y01 = _mm_add_ps(t0, t1);
				y23 = _mm_sub_ps(t0, t1);

				_mm_storel_pi((__m64*)&v0[0], y01);
				_mm_storeh_pi((__m64*)&v0[nx], y01);
				_mm_storel_pi((__m64*)&v1[0], y23);
				_mm_storeh_pi((__m64*)&v1[nx], y23);
			}
		}
	}
	*_pDw0 = dw0;
	return n;
}

#endif

// mixed-radix complex discrete Fourier transform: float-precision version
static void
DFT( const ComplexCR* src, ComplexCR* dst, int n,
	int nf, const int* factors, const int* itab,
	const ComplexCR* wave, int tab_size,
	ComplexCR* buf,
	int flags, double _scale )
{
	static const float sin_120 = (float)0.86602540378443864676372317075294;
	static const float fft5_2 = (float)0.559016994374947424102293417182819;
	static const float fft5_3 = (float)-0.951056516295153572116439333379382;
	static const float fft5_4 = (float)-1.538841768587626701285145288018455;
	static const float fft5_5 = (float)0.363271264002680442947733378740309;

	int n0 = n, f_idx, nx;
	int inv = flags & CR_DFT_INVERSE;
	int dw0 = tab_size, dw;
	int i, j, k;
	ComplexCR t;
	float scale = (float)_scale;
	int tab_step;

	tab_step = tab_size == n ? 1 : tab_size == n*2 ? 2 : tab_size/n;

	// 0. shuffle data
	if( dst != src )
	{
		assert( (flags & CR_DFT_NO_PERMUTE) == 0 );
		if( !inv )
		{
			for( i = 0; i <= n - 2; i += 2, itab += 2*tab_step )
			{
				int k0 = itab[0], k1 = itab[tab_step];
				assert( (unsigned)k0 < (unsigned)n && (unsigned)k1 < (unsigned)n );
				dst[i] = src[k0]; dst[i+1] = src[k1];
			}

			if( i < n )
				dst[n-1] = src[n-1];
		}
		else
		{
			for( i = 0; i <= n - 2; i += 2, itab += 2*tab_step )
			{
				int k0 = itab[0], k1 = itab[tab_step];
				assert( (unsigned)k0 < (unsigned)n && (unsigned)k1 < (unsigned)n );
				t.re = src[k0].re; t.im = -src[k0].im;
				dst[i] = t;
				t.re = src[k1].re; t.im = -src[k1].im;
				dst[i+1] = t;
			}

			if( i < n )
			{
				t.re = src[n-1].re; t.im = -src[n-1].im;
				dst[i] = t;
			}
		}
	}
	else
	{
		if( (flags & CR_DFT_NO_PERMUTE) == 0 )
		{
			assert( factors[0] == factors[nf-1] );
			if( nf == 1 )
			{
				if( (n & 3) == 0 )
				{
					int n2 = n/2;
					ComplexCR* dsth = dst + n2;

					for( i = 0; i < n2; i += 2, itab += tab_step*2 )
					{
						j = itab[0];
						assert( (unsigned)j < (unsigned)n2 );

						CR_SWAP(dst[i+1], dsth[j], t);
						if( j > i )
						{
							CR_SWAP(dst[i], dst[j], t);
							CR_SWAP(dsth[i+1], dsth[j+1], t);
						}
					}
				}
				// else do nothing
			}
			else
			{
				for( i = 0; i < n; i++, itab += tab_step )
				{
					j = itab[0];
					assert( (unsigned)j < (unsigned)n );
					if( j > i )
						CR_SWAP(dst[i], dst[j], t);
				}
			}
		}

		if( inv )
		{
			for( i = 0; i <= n - 2; i += 2 )
			{
				float t0 = -dst[i].im;
				float t1 = -dst[i+1].im;
				dst[i].im = t0; dst[i+1].im = t1;
			}

			if( i < n )
				dst[n-1].im = -dst[n-1].im;
		}
	}

	n = 1;
	// 1. power-2 transforms
	if( (factors[0] & 1) == 0 )
	{
//		Fast Fouier Transform
 /**
		if( factors[0] >= 4)
 		{
 			n = DFT_VecR4(dst, factors[0], n0, &dw0, wave);
 		}
/**/
		// radix-4 transform
		for( ; n*4 <= factors[0]; )
		{
			nx = n;
			n *= 4;
			dw0 /= 4;

			for( i = 0; i < n0; i += n )
			{
				ComplexCR *v0, *v1;
				float r0, i0, r1, i1, r2, i2, r3, i3, r4, i4;

				v0 = dst + i;
				v1 = v0 + nx*2;

				r0 = v1[0].re; i0 = v1[0].im;
				r4 = v1[nx].re; i4 = v1[nx].im;

				r1 = r0 + r4; i1 = i0 + i4;
				r3 = i0 - i4; i3 = r4 - r0;

				r2 = v0[0].re; i2 = v0[0].im;
				r4 = v0[nx].re; i4 = v0[nx].im;

				r0 = r2 + r4; i0 = i2 + i4;
				r2 -= r4; i2 -= i4;

				v0[0].re = r0 + r1; v0[0].im = i0 + i1;
				v1[0].re = r0 - r1; v1[0].im = i0 - i1;
				v0[nx].re = r2 + r3; v0[nx].im = i2 + i3;
				v1[nx].re = r2 - r3; v1[nx].im = i2 - i3;

				for( j = 1, dw = dw0; j < nx; j++, dw += dw0 )
				{
					v0 = dst + i + j;
					v1 = v0 + nx*2;

					r2 = v0[nx].re*wave[dw*2].re - v0[nx].im*wave[dw*2].im;
					i2 = v0[nx].re*wave[dw*2].im + v0[nx].im*wave[dw*2].re;
					r0 = v1[0].re*wave[dw].im + v1[0].im*wave[dw].re;
					i0 = v1[0].re*wave[dw].re - v1[0].im*wave[dw].im;
					r3 = v1[nx].re*wave[dw*3].im + v1[nx].im*wave[dw*3].re;
					i3 = v1[nx].re*wave[dw*3].re - v1[nx].im*wave[dw*3].im;

					r1 = i0 + i3; i1 = r0 + r3;
					r3 = r0 - r3; i3 = i3 - i0;
					r4 = v0[0].re; i4 = v0[0].im;

					r0 = r4 + r2; i0 = i4 + i2;
					r2 = r4 - r2; i2 = i4 - i2;

					v0[0].re = r0 + r1; v0[0].im = i0 + i1;
					v1[0].re = r0 - r1; v1[0].im = i0 - i1;
					v0[nx].re = r2 + r3; v0[nx].im = i2 + i3;
					v1[nx].re = r2 - r3; v1[nx].im = i2 - i3;
				}
			}
		}

		for( ; n < factors[0]; )
		{
			// do the remaining radix-2 transform
			nx = n;
			n *= 2;
			dw0 /= 2;

			for( i = 0; i < n0; i += n )
			{
				ComplexCR* v = dst + i;
				float r0 = v[0].re + v[nx].re;
				float i0 = v[0].im + v[nx].im;
				float r1 = v[0].re - v[nx].re;
				float i1 = v[0].im - v[nx].im;
				v[0].re = r0; v[0].im = i0;
				v[nx].re = r1; v[nx].im = i1;

				for( j = 1, dw = dw0; j < nx; j++, dw += dw0 )
				{
					v = dst + i + j;
					r1 = v[nx].re*wave[dw].re - v[nx].im*wave[dw].im;
					i1 = v[nx].im*wave[dw].re + v[nx].re*wave[dw].im;
					r0 = v[0].re; i0 = v[0].im;

					v[0].re = r0 + r1; v[0].im = i0 + i1;
					v[nx].re = r0 - r1; v[nx].im = i0 - i1;
				}
			}
		}
	}

	// 2. all the other transforms
	for( f_idx = (factors[0]&1) ? 0 : 1; f_idx < nf; f_idx++ )
	{
		int factor = factors[f_idx];
		nx = n;
		n *= factor;
		dw0 /= factor;

		if( factor == 3 )
		{
			// radix-3
			for( i = 0; i < n0; i += n )
			{
				ComplexCR* v = dst + i;

				float r1 = v[nx].re + v[nx*2].re;
				float i1 = v[nx].im + v[nx*2].im;
				float r0 = v[0].re;
				float i0 = v[0].im;
				float r2 = sin_120*(v[nx].im - v[nx*2].im);
				float i2 = sin_120*(v[nx*2].re - v[nx].re);
				v[0].re = r0 + r1; v[0].im = i0 + i1;
				r0 -= (float)0.5*r1; i0 -= (float)0.5*i1;
				v[nx].re = r0 + r2; v[nx].im = i0 + i2;
				v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;

				for( j = 1, dw = dw0; j < nx; j++, dw += dw0 )
				{
					v = dst + i + j;
					r0 = v[nx].re*wave[dw].re - v[nx].im*wave[dw].im;
					i0 = v[nx].re*wave[dw].im + v[nx].im*wave[dw].re;
					i2 = v[nx*2].re*wave[dw*2].re - v[nx*2].im*wave[dw*2].im;
					r2 = v[nx*2].re*wave[dw*2].im + v[nx*2].im*wave[dw*2].re;
					r1 = r0 + i2; i1 = i0 + r2;

					r2 = sin_120*(i0 - r2); i2 = sin_120*(i2 - r0);
					r0 = v[0].re; i0 = v[0].im;
					v[0].re = r0 + r1; v[0].im = i0 + i1;
					r0 -= (float)0.5*r1; i0 -= (float)0.5*i1;
					v[nx].re = r0 + r2; v[nx].im = i0 + i2;
					v[nx*2].re = r0 - r2; v[nx*2].im = i0 - i2;
				}
			}
		}
		else if( factor == 5 )
		{
			// radix-5
			for( i = 0; i < n0; i += n )
			{
				for( j = 0, dw = 0; j < nx; j++, dw += dw0 )
				{
					ComplexCR* v0 = dst + i + j;
					ComplexCR* v1 = v0 + nx*2;
					ComplexCR* v2 = v1 + nx*2;

					float r0, i0, r1, i1, r2, i2, r3, i3, r4, i4, r5, i5;

					r3 = v0[nx].re*wave[dw].re - v0[nx].im*wave[dw].im;
					i3 = v0[nx].re*wave[dw].im + v0[nx].im*wave[dw].re;
					r2 = v2[0].re*wave[dw*4].re - v2[0].im*wave[dw*4].im;
					i2 = v2[0].re*wave[dw*4].im + v2[0].im*wave[dw*4].re;

					r1 = r3 + r2; i1 = i3 + i2;
					r3 -= r2; i3 -= i2;

					r4 = v1[nx].re*wave[dw*3].re - v1[nx].im*wave[dw*3].im;
					i4 = v1[nx].re*wave[dw*3].im + v1[nx].im*wave[dw*3].re;
					r0 = v1[0].re*wave[dw*2].re - v1[0].im*wave[dw*2].im;
					i0 = v1[0].re*wave[dw*2].im + v1[0].im*wave[dw*2].re;

					r2 = r4 + r0; i2 = i4 + i0;
					r4 -= r0; i4 -= i0;

					r0 = v0[0].re; i0 = v0[0].im;
					r5 = r1 + r2; i5 = i1 + i2;

					v0[0].re = r0 + r5; v0[0].im = i0 + i5;

					r0 -= (float)0.25*r5; i0 -= (float)0.25*i5;
					r1 = fft5_2*(r1 - r2); i1 = fft5_2*(i1 - i2);
					r2 = -fft5_3*(i3 + i4); i2 = fft5_3*(r3 + r4);

					i3 *= -fft5_5; r3 *= fft5_5;
					i4 *= -fft5_4; r4 *= fft5_4;

					r5 = r2 + i3; i5 = i2 + r3;
					r2 -= i4; i2 -= r4;

					r3 = r0 + r1; i3 = i0 + i1;
					r0 -= r1; i0 -= i1;

					v0[nx].re = r3 + r2; v0[nx].im = i3 + i2;
					v2[0].re = r3 - r2; v2[0].im = i3 - i2;

					v1[0].re = r0 + r5; v1[0].im = i0 + i5;
					v1[nx].re = r0 - r5; v1[nx].im = i0 - i5;
				}
			}
		}
		else
		{
			// radix-"factor" - an odd number
			int p, q, factor2 = (factor - 1)/2;
			int d, dd, dw_f = tab_size/factor;
			ComplexCR* a = buf;
			ComplexCR* b = buf + factor2;

			for( i = 0; i < n0; i += n )
			{
				for( j = 0, dw = 0; j < nx; j++, dw += dw0 )
				{
					ComplexCR* v = dst + i + j;
					ComplexCR v_0 = v[0];
					ComplexCR vn_0 = v_0;

					if( j == 0 )
					{
						for( p = 1, k = nx; p <= factor2; p++, k += nx )
						{
							float r0 = v[k].re + v[n-k].re;
							float i0 = v[k].im - v[n-k].im;
							float r1 = v[k].re - v[n-k].re;
							float i1 = v[k].im + v[n-k].im;

							vn_0.re += r0; vn_0.im += i1;
							a[p-1].re = r0; a[p-1].im = i0;
							b[p-1].re = r1; b[p-1].im = i1;
						}
					}
					else
					{
						const ComplexCR* wave_ = wave + dw*factor;
						d = dw;

						for( p = 1, k = nx; p <= factor2; p++, k += nx, d += dw )
						{
							float r2 = v[k].re*wave[d].re - v[k].im*wave[d].im;
							float i2 = v[k].re*wave[d].im + v[k].im*wave[d].re;

							float r1 = v[n-k].re*wave_[-d].re - v[n-k].im*wave_[-d].im;
							float i1 = v[n-k].re*wave_[-d].im + v[n-k].im*wave_[-d].re;

							float r0 = r2 + r1;
							float i0 = i2 - i1;
							r1 = r2 - r1;
							i1 = i2 + i1;

							vn_0.re += r0; vn_0.im += i1;
							a[p-1].re = r0; a[p-1].im = i0;
							b[p-1].re = r1; b[p-1].im = i1;
						}
					}

					v[0] = vn_0;

					for( p = 1, k = nx; p <= factor2; p++, k += nx )
					{
						ComplexCR s0 = v_0, s1 = v_0;
						d = dd = dw_f*p;

						for( q = 0; q < factor2; q++ )
						{
							float r0 = wave[d].re * a[q].re;
							float i0 = wave[d].im * a[q].im;
							float r1 = wave[d].re * b[q].im;
							float i1 = wave[d].im * b[q].re;

							s1.re += r0 + i0; s0.re += r0 - i0;
							s1.im += r1 - i1; s0.im += r1 + i1;

							d += dd;
							d -= -(d >= tab_size) & tab_size;
						}

						v[k] = s0;
						v[n-k] = s1;
					}
				}
			}
		}
	}

	if( scale != 1 )
	{
		float re_scale = scale, im_scale = scale;
		if( inv )
			im_scale = -im_scale;

		for( i = 0; i < n0; i++ )
		{
			float t0 = dst[i].re*re_scale;
			float t1 = dst[i].im*im_scale;
			dst[i].re = t0;
			dst[i].im = t1;
		}
	}
	else if( inv )
	{
		for( i = 0; i <= n0 - 2; i += 2 )
		{
			float t0 = -dst[i].im;
			float t1 = -dst[i+1].im;
			dst[i].im = t0;
			dst[i+1].im = t1;
		}

		if( i < n0 )
			dst[n0-1].im = -dst[n0-1].im;
	}
}

static void
	CopyColumn( const unsigned char* _src, size_t src_step,
	unsigned char* _dst, size_t dst_step,
	int len, size_t elem_size )
{
	int i, t0, t1;
	const int* src = (const int*)_src;
	int* dst = (int*)_dst;
	src_step /= sizeof(src[0]);
	dst_step /= sizeof(dst[0]);

	if( elem_size == sizeof(int) )
	{
		for( i = 0; i < len; i++, src += src_step, dst += dst_step )
			dst[0] = src[0];
	}
	else if( elem_size == sizeof(int)*2 )
	{
		for( i = 0; i < len; i++, src += src_step, dst += dst_step )
		{
			t0 = src[0]; t1 = src[1];
			dst[0] = t0; dst[1] = t1;
		}
	}
	else if( elem_size == sizeof(int)*4 )
	{
		for( i = 0; i < len; i++, src += src_step, dst += dst_step )
		{
			t0 = src[0]; t1 = src[1];
			dst[0] = t0; dst[1] = t1;
			t0 = src[2]; t1 = src[3];
			dst[2] = t0; dst[3] = t1;
		}
	}
}

static void
	CopyFrom2Columns( const unsigned char* _src, size_t src_step,
	unsigned char* _dst0, unsigned char* _dst1,
	int len, size_t elem_size )
{
	int i, t0, t1;
	const int* src = (const int*)_src;
	int* dst0 = (int*)_dst0;
	int* dst1 = (int*)_dst1;
	src_step /= sizeof(src[0]);

	if( elem_size == sizeof(int) )
	{
		for( i = 0; i < len; i++, src += src_step )
		{
			t0 = src[0]; t1 = src[1];
			dst0[i] = t0; dst1[i] = t1;
		}
	}
	else if( elem_size == sizeof(int)*2 )
	{
		for( i = 0; i < len*2; i += 2, src += src_step )
		{
			t0 = src[0]; t1 = src[1];
			dst0[i] = t0; dst0[i+1] = t1;
			t0 = src[2]; t1 = src[3];
			dst1[i] = t0; dst1[i+1] = t1;
		}
	}
	else if( elem_size == sizeof(int)*4 )
	{
		for( i = 0; i < len*4; i += 4, src += src_step )
		{
			t0 = src[0]; t1 = src[1];
			dst0[i] = t0; dst0[i+1] = t1;
			t0 = src[2]; t1 = src[3];
			dst0[i+2] = t0; dst0[i+3] = t1;
			t0 = src[4]; t1 = src[5];
			dst1[i] = t0; dst1[i+1] = t1;
			t0 = src[6]; t1 = src[7];
			dst1[i+2] = t0; dst1[i+3] = t1;
		}
	}
}

static void
	CopyTo2Columns( const unsigned char* _src0, const unsigned char* _src1,
	unsigned char* _dst, size_t dst_step,
	int len, size_t elem_size )
{
	int i, t0, t1;
	const int* src0 = (const int*)_src0;
	const int* src1 = (const int*)_src1;
	int* dst = (int*)_dst;
	dst_step /= sizeof(dst[0]);

	if( elem_size == sizeof(int) )
	{
		for( i = 0; i < len; i++, dst += dst_step )
		{
			t0 = src0[i]; t1 = src1[i];
			dst[0] = t0; dst[1] = t1;
		}
	}
	else if( elem_size == sizeof(int)*2 )
	{
		for( i = 0; i < len*2; i += 2, dst += dst_step )
		{
			t0 = src0[i]; t1 = src0[i+1];
			dst[0] = t0; dst[1] = t1;
			t0 = src1[i]; t1 = src1[i+1];
			dst[2] = t0; dst[3] = t1;
		}
	}
	else if( elem_size == sizeof(int)*4 )
	{
		for( i = 0; i < len*4; i += 4, dst += dst_step )
		{
			t0 = src0[i]; t1 = src0[i+1];
			dst[0] = t0; dst[1] = t1;
			t0 = src0[i+2]; t1 = src0[i+3];
			dst[2] = t0; dst[3] = t1;
			t0 = src1[i]; t1 = src1[i+1];
			dst[4] = t0; dst[5] = t1;
			t0 = src1[i+2]; t1 = src1[i+3];
			dst[6] = t0; dst[7] = t1;
		}
	}
}

static void
	ExpandCCS( unsigned char* _ptr, int n, int elem_size )
{
	int i;
	if( elem_size == (int)sizeof(float) )
	{
		float* p = (float*)_ptr;
		for( i = 1; i < (n+1)/2; i++ )
		{
			p[(n-i)*2] = p[i*2-1];
			p[(n-i)*2+1] = -p[i*2];
		}
		if( (n & 1) == 0 )
		{
			p[n] = p[n-1];
			p[n+1] = 0.f;
			n--;
		}
		for( i = n-1; i > 0; i-- )
			p[i+1] = p[i];
		p[1] = 0.f;
	}
}

static 
void CR_DFT_32f( const ComplexCR* src, ComplexCR* dst, int n,
	int nf, const int* factors, const int* itab,
	const ComplexCR* wave, int tab_size,
	ComplexCR* buf,
	int flags, double scale )
{
	DFT(src, dst, n, nf, factors, itab, wave, tab_size, buf, flags, scale);
}


void dftCR( IMG_MAT_FLOAT _src0, 
	IMG_MAT_FLOAT _dst,
	/*void *memT,*/
	int flags, int nonzero_rows )
{
	static DFTFunc dft_tbl[6] =
	{
		(DFTFunc)CR_DFT_32f,
	};

	void *spec = 0;
	int prev_len = 0, stage = 0;
	bool inv = (flags & CR_DFT_INVERSE) != 0;
	int nf = 0, real_transform = _src0.channels == 1 || (inv && (flags & CR_DFT_REAL_OUTPUT)!=0);
	int elem_size = sizeof(float), complex_elem_size = elem_size*2;
	int factors[34];
	bool inplace_transform = false;
	IMG_MAT_FLOAT src, dst;
	unsigned char* ptr = NULL, *ptrbak = NULL;

	src = _src0; 
	dst = _dst;
	
	if( !real_transform )
		elem_size = complex_elem_size;

	if( _src0.width == 1 && nonzero_rows > 0 )
		return;
	
	// determine, which transform to do first - row-wise
	// (stage 0) or column-wise (stage 1) transform
	if( !(flags & CR_DFT_ROWS) && _src0.height > 1 && (_src0.width == 1)  )
		stage = 1;

	for(;;)
	{
		float scale = 1;
		unsigned char* wave = 0;
		int* itab = 0;
		int i, len, count, sz = 0;
		int use_buf = 0, odd_real = 0;
		DFTFunc dft_func;

		if( stage == 0 ) // row-wise transform
		{
			len = !inv ? src.width : dst.width;
			count = src.height;
			if( len == 1 && !(flags & CR_DFT_ROWS) )
			{
				len = !inv ? src.height : dst.height;
				count = 1;
			}
			odd_real = real_transform && (len & 1);
		}
		else
		{
			len = dst.height;
			count = !inv ? _src0.width : dst.width;
			sz = 2*len*complex_elem_size;
		}

		{
			//if( len != prev_len )
			{
				nf = DFTFactorize( len, factors );
				if(ptrbak != NULL){
					FreeAlign(ptrbak);
					ptrbak = NULL;
				}
			}

			inplace_transform = factors[0] == factors[nf-1];
			sz += len*(complex_elem_size + sizeof(int));
			i = nf > 1 && (factors[0] & 1) == 0;
			if( (factors[i] & 1) != 0 && factors[i] > 5 )
				sz += (factors[i]+1)*complex_elem_size;

			if( (stage == 0 && ((src.data == dst.data && !inplace_transform) || odd_real)) ||
				(stage == 1 && !inplace_transform) )
			{
				use_buf = 1;
				sz += len*complex_elem_size;
			}
		}

		if(ptrbak == NULL)
			ptrbak = (unsigned char*)MallocAlign(sz + 32);

		ptr = ptrbak;
		// force recalculation of
		// twiddle factors and permutation table
		if( !spec )
		{
			wave = ptr;
			ptr += len*complex_elem_size;
			itab = (int*)ptr;
			ptr = (unsigned char*)cvAlignPtr( ptr + len*sizeof(int), 16 );

			if( len != prev_len || (!inplace_transform && inv && real_transform))
				DFTInit( len, nf, factors, itab, complex_elem_size, wave, stage == 0 && inv && real_transform );
			// otherwise reuse the tables calculated on the previous stage
		}

		if( stage == 0 )
		{
			unsigned char* tmp_buf = 0;
			int dptr_offset = 0;
			int dst_full_len = len*elem_size;
			int _flags = (int)inv + (src.channels != dst.channels ?CR_DFT_COMPLEX_INPUT_OR_OUTPUT : 0);
			if( use_buf )
			{
				tmp_buf = ptr;
				ptr += len*complex_elem_size;
				if( odd_real && !inv && len > 1 && !(_flags & CR_DFT_COMPLEX_INPUT_OR_OUTPUT))
					dptr_offset = elem_size;
			}

			if( !inv && (_flags & CR_DFT_COMPLEX_INPUT_OR_OUTPUT) )
				dst_full_len += (len & 1) ? elem_size : complex_elem_size;

			dft_func = dft_tbl[0];

			if( count > 1 && !(flags & CR_DFT_ROWS) && (!inv || !real_transform) )
				stage = 1;
			else if( flags & CR_DXT_SCALE )
				scale = 1./(len * (flags & CR_DFT_ROWS ? 1 : count));

			if( nonzero_rows <= 0 || nonzero_rows > count )
				nonzero_rows = count;

			for( i = 0; i < nonzero_rows; i++ )
			{
				unsigned char* sptr = (unsigned char*)(src.data + i*src.step[0]);
				unsigned char* dptr0 = (unsigned char*)(dst.data + i*dst.step[0]);
				unsigned char* dptr = dptr0;

				if( tmp_buf )
					dptr = tmp_buf;

				dft_func( sptr, dptr, len, nf, factors, itab, wave, len, ptr, _flags, scale );
				if( dptr != dptr0 )
					memcpy( dptr0, dptr + dptr_offset, dst_full_len );
			}

			for( ; i < count; i++ )
			{
				unsigned char* dptr0 = (unsigned char*)(dst.data + i*dst.step[0]);
				memset( dptr0, 0, dst_full_len );
			}

			if( stage != 1 )
				break;
			src = dst;
		}
		else
		{
			int a = 0, b = count;
			unsigned char *buf0, *buf1, *dbuf0, *dbuf1;
			unsigned char* sptr0 = (unsigned char*)src.data;
			unsigned char* dptr0 = (unsigned char*)dst.data;
			buf0 = ptr;
			ptr += len*complex_elem_size;
			buf1 = ptr;
			ptr += len*complex_elem_size;
			dbuf0 = buf0, dbuf1 = buf1;

			if( use_buf )
			{
				dbuf1 = ptr;
				dbuf0 = buf1;
				ptr += len*complex_elem_size;
			}

			dft_func = dft_tbl[0];

			if( real_transform && inv && src.width > 1 )
				stage = 0;
			else if( flags & CR_DXT_SCALE )
				scale = 1./(len * count);

			for( i = a; i < b; i += 2 )
			{
				if( i+1 < b )
				{
					CopyFrom2Columns( sptr0, _src0.step[0]*sizeof(float), buf0, buf1, len, complex_elem_size ); 
					dft_func( buf1, dbuf1, len, nf, factors, itab, wave, len, ptr, inv, scale );
				}
				else
					CopyColumn( sptr0, _src0.step[0]*sizeof(float), buf0, complex_elem_size, len, complex_elem_size );

				dft_func( buf0, dbuf0, len, nf, factors, itab, wave, len, ptr, inv, scale );

				if( i+1 < b )
					CopyTo2Columns( dbuf0, dbuf1, dptr0, _dst.step[0]*sizeof(float), len, complex_elem_size );
				else
					CopyColumn( dbuf0, complex_elem_size, dptr0, _dst.step[0]*sizeof(float), len, complex_elem_size );
				sptr0 += 2*complex_elem_size;
				dptr0 += 2*complex_elem_size;
			}

			if( stage != 0 )
			{
				break;
			}
			src = dst;
		}

		//prev_len = len;
	}
	if(ptrbak != NULL){
		FreeAlign(ptrbak);
		ptrbak = NULL;
	}

}

/*
unsigned char *_mempace = NULL;
void dftCR( IMG_MAT_FLOAT _src0, IMG_MAT_FLOAT _dst, int flags, int nonzero_rows )
{
	float *pSrc = (float *)_mempace;
	float *srcbak = _src0.data;
	IMG_MAT_FLOAT matDest;
	
	memcpy(pSrc, _src0.data, _src0.step[0]*_src0.height*sizeof(float));
	_src0.data = pSrc;
	memcpy(&matDest, &_dst, sizeof(IMG_MAT_FLOAT));
	matDest.data = (float *)(_mempace+_src0.step[0]*_src0.height*sizeof(float));
	_dftCR(_src0, matDest, 
		matDest.data+_src0.step[0]*_src0.height,
		flags, nonzero_rows);
	memcpy(_dst.data, matDest.data, _src0.step[0]*_src0.height*sizeof(float));
	_src0.data = srcbak;
}
*/

void idftCR( IMG_MAT_FLOAT src, IMG_MAT_FLOAT dst, int flags, int nonzero_rows )
{
	dftCR( src, dst, flags | CR_DFT_INVERSE, nonzero_rows );
}

void mulSpectrumsCR( IMG_MAT_FLOAT _srcA, IMG_MAT_FLOAT _srcB, IMG_MAT_FLOAT _dst, int flags, bool conjB )
{
	IMG_MAT_FLOAT srcA = _srcA;
	IMG_MAT_FLOAT srcB = _srcB;
	IMG_MAT_FLOAT dst = _dst;
	int rows = srcA.height, cols = srcA.width;
	int j, k;

	int cn, ncols, j0, j1;

	bool is_1d = (flags & CR_DFT_ROWS) || (rows == 1 || cols == 1 );

	if( is_1d && !(flags & CR_DFT_ROWS) )
		cols = cols + rows - 1, rows = 1;

	cn = _srcA.channels;
	ncols = cols*cn;
	j0 = cn == 1;
	j1 = ncols - (cols % 2 == 0 && cn == 1);

	{
		const float* dataA = (const float*)srcA.data;
		const float* dataB = (const float*)srcB.data;
		float* dataC = (float*)dst.data;

		size_t stepA = srcA.step[0];///sizeof(dataA[0]);
		size_t stepB = srcB.step[0];///sizeof(dataB[0]);
		size_t stepC = dst.step[0];///sizeof(dataC[0]);

		if( !is_1d && cn == 1 )
		{
			for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
			{
				if( k == 1 )
					dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
				dataC[0] = dataA[0]*dataB[0];
				if( rows % 2 == 0 )
					dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA]*dataB[(rows-1)*stepB];
				if( !conjB )
					for( j = 1; j <= rows - 2; j += 2 )
					{
						double re = (double)dataA[j*stepA]*dataB[j*stepB] -
												(double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];
						double im = (double)dataA[j*stepA]*dataB[(j+1)*stepB] +
												(double)dataA[(j+1)*stepA]*dataB[j*stepB];
						dataC[j*stepC] = (float)re; dataC[(j+1)*stepC] = (float)im;
					}
				else
					for( j = 1; j <= rows - 2; j += 2 )
					{
						double re = (double)dataA[j*stepA]*dataB[j*stepB] +
												(double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];
						double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] -
												(double)dataA[j*stepA]*dataB[(j+1)*stepB];
						dataC[j*stepC] = (float)re; dataC[(j+1)*stepC] = (float)im;
					}
					if( k == 1 )
						dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
			}
		}

		for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
		{
			if( is_1d && cn == 1 )
			{
				dataC[0] = dataA[0]*dataB[0];
				if( cols % 2 == 0 )
					dataC[j1] = dataA[j1]*dataB[j1];
			}

			if( !conjB )
				for( j = j0; j < j1; j += 2 )
				{
					double re = (double)dataA[j]*dataB[j] - (double)dataA[j+1]*dataB[j+1];
					double im = (double)dataA[j+1]*dataB[j] + (double)dataA[j]*dataB[j+1];
					dataC[j] = (float)re; dataC[j+1] = (float)im;
				}
			else
				for( j = j0; j < j1; j += 2 )
				{
					double re = (double)dataA[j]*dataB[j] + (double)dataA[j+1]*dataB[j+1];
					double im = (double)dataA[j+1]*dataB[j] - (double)dataA[j]*dataB[j+1];
					dataC[j] = (float)re; dataC[j+1] = (float)im;
				}
		}
	}
}

