#include "PCTracker.h"
#include "DFT.h"

/****************************************************************************************\
                               Discrete Fourier Transform
\****************************************************************************************/

void* cvAlignPtr( const void* ptr, int align)
{
	UTILS_assert( (align & (align-1)) == 0 );
	return (void*)( ((size_t)ptr + align - 1) & ~(size_t)(align-1) );
}

static unsigned char bitrevTab[] =
{
  0x00,0x80,0x40,0xc0,0x20,0xa0,0x60,0xe0,0x10,0x90,0x50,0xd0,0x30,0xb0,0x70,0xf0,
  0x08,0x88,0x48,0xc8,0x28,0xa8,0x68,0xe8,0x18,0x98,0x58,0xd8,0x38,0xb8,0x78,0xf8,
  0x04,0x84,0x44,0xc4,0x24,0xa4,0x64,0xe4,0x14,0x94,0x54,0xd4,0x34,0xb4,0x74,0xf4,
  0x0c,0x8c,0x4c,0xcc,0x2c,0xac,0x6c,0xec,0x1c,0x9c,0x5c,0xdc,0x3c,0xbc,0x7c,0xfc,
  0x02,0x82,0x42,0xc2,0x22,0xa2,0x62,0xe2,0x12,0x92,0x52,0xd2,0x32,0xb2,0x72,0xf2,
  0x0a,0x8a,0x4a,0xca,0x2a,0xaa,0x6a,0xea,0x1a,0x9a,0x5a,0xda,0x3a,0xba,0x7a,0xfa,
  0x06,0x86,0x46,0xc6,0x26,0xa6,0x66,0xe6,0x16,0x96,0x56,0xd6,0x36,0xb6,0x76,0xf6,
  0x0e,0x8e,0x4e,0xce,0x2e,0xae,0x6e,0xee,0x1e,0x9e,0x5e,0xde,0x3e,0xbe,0x7e,0xfe,
  0x01,0x81,0x41,0xc1,0x21,0xa1,0x61,0xe1,0x11,0x91,0x51,0xd1,0x31,0xb1,0x71,0xf1,
  0x09,0x89,0x49,0xc9,0x29,0xa9,0x69,0xe9,0x19,0x99,0x59,0xd9,0x39,0xb9,0x79,0xf9,
  0x05,0x85,0x45,0xc5,0x25,0xa5,0x65,0xe5,0x15,0x95,0x55,0xd5,0x35,0xb5,0x75,0xf5,
  0x0d,0x8d,0x4d,0xcd,0x2d,0xad,0x6d,0xed,0x1d,0x9d,0x5d,0xdd,0x3d,0xbd,0x7d,0xfd,
  0x03,0x83,0x43,0xc3,0x23,0xa3,0x63,0xe3,0x13,0x93,0x53,0xd3,0x33,0xb3,0x73,0xf3,
  0x0b,0x8b,0x4b,0xcb,0x2b,0xab,0x6b,0xeb,0x1b,0x9b,0x5b,0xdb,0x3b,0xbb,0x7b,0xfb,
  0x07,0x87,0x47,0xc7,0x27,0xa7,0x67,0xe7,0x17,0x97,0x57,0xd7,0x37,0xb7,0x77,0xf7,
  0x0f,0x8f,0x4f,0xcf,0x2f,0xaf,0x6f,0xef,0x1f,0x9f,0x5f,0xdf,0x3f,0xbf,0x7f,0xff
};

static const double DFTTab[][2] =
{
{ 1.00000000000000000, 0.00000000000000000 },
{-1.00000000000000000, 0.00000000000000000 },
{ 0.00000000000000000, 1.00000000000000000 },
{ 0.70710678118654757, 0.70710678118654746 },
{ 0.92387953251128674, 0.38268343236508978 },
{ 0.98078528040323043, 0.19509032201612825 },
{ 0.99518472667219693, 0.09801714032956060 },
{ 0.99879545620517241, 0.04906767432741802 },
{ 0.99969881869620425, 0.02454122852291229 },
{ 0.99992470183914450, 0.01227153828571993 },
{ 0.99998117528260111, 0.00613588464915448 },
{ 0.99999529380957619, 0.00306795676296598 },
{ 0.99999882345170188, 0.00153398018628477 },
{ 0.99999970586288223, 0.00076699031874270 },
{ 0.99999992646571789, 0.00038349518757140 },
{ 0.99999998161642933, 0.00019174759731070 },
{ 0.99999999540410733, 0.00009587379909598 },
{ 0.99999999885102686, 0.00004793689960307 },
{ 0.99999999971275666, 0.00002396844980842 },
{ 0.99999999992818922, 0.00001198422490507 },
{ 0.99999999998204725, 0.00000599211245264 },
{ 0.99999999999551181, 0.00000299605622633 },
{ 0.99999999999887801, 0.00000149802811317 },
{ 0.99999999999971945, 0.00000074901405658 },
{ 0.99999999999992983, 0.00000037450702829 },
{ 0.99999999999998246, 0.00000018725351415 },
{ 0.99999999999999567, 0.00000009362675707 },
{ 0.99999999999999889, 0.00000004681337854 },
{ 0.99999999999999978, 0.00000002340668927 },
{ 0.99999999999999989, 0.00000001170334463 },
{ 1.00000000000000000, 0.00000000585167232 },
{ 1.00000000000000000, 0.00000000292583616 }
};

#define BitRev(i,shift) \
   ((int)((((unsigned)bitrevTab[(i)&255] << 24)+ \
           ((unsigned)bitrevTab[((i)>> 8)&255] << 16)+ \
           ((unsigned)bitrevTab[((i)>>16)&255] <<  8)+ \
           ((unsigned)bitrevTab[((i)>>24)])) >> (shift)))


int	DFTFactorize( int n, int* factors )
{
	int nf = 0, f, i, j;

	if( n <= 5 )
	{
		factors[0] = n;
		return 1;
	}

	f = (((n - 1)^n)+1) >> 1;
	if( f > 1 )
	{
		factors[nf++] = f;
		n = f == n ? 1 : n/f;
	}

	for( f = 3; n > 1; )
	{
		int d = n/f;
		if( d*f == n )
		{
			factors[nf++] = f;
			n = d;
		}
		else
		{
			f += 2;
			if( f*f > n )
				break;
		}
	}

	if( n > 1 )
		factors[nf++] = n;

	f = (factors[0] & 1) == 0;
	for( i = f; i < (nf+f)/2; i++ )
		CR_SWAP( factors[i], factors[nf-i-1+f], j );

	return nf;
}

void	DFTInit( int n0, int nf, int* factors, int* itab, int elem_size, void* _wave, int inv_itab )
{
	int digits[34], radix[34];
	int n = factors[0], m = 0;
	int* itab0 = itab;
	int i, j, k;
	ComplexCR w, w1, *wave;
	float t;

	if( n0 <= 5 )
	{
		itab[0] = 0;
		itab[n0-1] = n0-1;

		if( n0 != 4 )
		{
			for( i = 1; i < n0-1; i++ )
				itab[i] = i;
		}
		else
		{
			itab[1] = 2;
			itab[2] = 1;
		}
		if( n0 == 5 )
		{
			if( elem_size == sizeof(ComplexCR) ){
				wave = (ComplexCR*)_wave;
				wave[0].re = 1.f;	wave[0].im = 0.f;
			}
		}
		if( n0 != 4 )
			return;
		m = 2;
	}
	else
	{
		// radix[] is initialized from index 'nf' down to zero
		UTILS_assert (nf < 34);
		radix[nf] = 1;
		digits[nf] = 0;
		for( i = 0; i < nf; i++ )
		{
			digits[i] = 0;
			radix[nf-i-1] = radix[nf-i]*factors[nf-i-1];
		}

		if( inv_itab && factors[0] != factors[nf-1] )
			itab = (int*)_wave;

		if( (n & 1) == 0 )
		{
			int a = radix[1], na2 = n*a>>1, na4 = na2 >> 1;
			for( m = 0; (unsigned)(1 << m) < (unsigned)n; m++ )
				;
			if( n <= 2  )
			{
				itab[0] = 0;
				itab[1] = na2;
			}
			else if( n <= 256 )
			{
				int shift = 10 - m;
				for( i = 0; i <= n - 4; i += 4 )
				{
					j = (bitrevTab[i>>2]>>shift)*a;
					itab[i] = j;
					itab[i+1] = j + na2;
					itab[i+2] = j + na4;
					itab[i+3] = j + na2 + na4;
				}
			}
			else
			{
				int shift = 34 - m;
				for( i = 0; i < n; i += 4 )
				{
					int i4 = i >> 2;
					j = BitRev(i4,shift)*a;
					itab[i] = j;
					itab[i+1] = j + na2;
					itab[i+2] = j + na4;
					itab[i+3] = j + na2 + na4;
				}
			}

			digits[1]++;

			if( nf >= 2 )
			{
				for( i = n, j = radix[2]; i < n0; )
				{
					for( k = 0; k < n; k++ )
						itab[i+k] = itab[k] + j;
					if( (i += n) >= n0 )
						break;
					j += radix[2];
					for( k = 1; ++digits[k] >= factors[k]; k++ )
					{
						digits[k] = 0;
						j += radix[k+2] - radix[k];
					}
				}
			}
		}
		else
		{
			for( i = 0, j = 0;; )
			{
				itab[i] = j;
				if( ++i >= n0 )
					break;
				j += radix[1];
				for( k = 0; ++digits[k] >= factors[k]; k++ )
				{
					digits[k] = 0;
					j += radix[k+2] - radix[k];
				}
			}
		}

		if( itab != itab0 )
		{
			itab0[0] = 0;
			for( i = n0 & 1; i < n0; i += 2 )
			{
				int k0 = itab[i];
				int k1 = itab[i+1];
				itab0[k0] = i;
				itab0[k1] = i+1;
			}
		}
	}

	if( (n0 & (n0-1)) == 0 )
	{
		w.re = w1.re = (float)DFTTab[m][0];
		w.im = w1.im = (float)-DFTTab[m][1];
	}
	else
	{
		t = -CR_PI*2/n0;
		w.im = w1.im = (float)sin(t);
		w.re = w1.re = (float)sqrt(1. - w1.im*w1.im);
	}
	n = (n0+1)/2;

	if( elem_size == sizeof(ComplexCR) )
	{
		wave = (ComplexCR*)_wave;

		wave[0].re = 1.f;
		wave[0].im = 0.f;

		if( (n0 & 1) == 0 )
		{
			wave[n].re = -1.f;
			wave[n].im = 0.f;
		}

		for( i = 1; i < n; i++ )
		{
			wave[i].re = (float)w.re;
			wave[i].im = (float)w.im;
			wave[n0-i].re = (float)w.re;
			wave[n0-i].im = (float)-w.im;

			t = w.re*w1.re - w.im*w1.im;
			w.im = w.re*w1.im + w.im*w1.re;
			w.re = t;
		}
	}
}
