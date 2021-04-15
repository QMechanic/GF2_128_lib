///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Inspiration and explanation: https://blog.quarkslab.com/reversing-a-finite-field-multiplication-optimization.html ////
//// I wrote all of this myself from scratch                                                                           ////
//// I also added the square function for exponentiation which I haven't seen mentioned in relation to this            ////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>

using u32   = uint32_t;
using u64   = uint64_t;
using u64x2 = u64 __attribute__ ((vector_size (2*64/8)));
using u64x4 = u64 __attribute__ ((vector_size (4*64/8)));
using v128 = union
{
	__m128i mm;
	u64x2 u64;
};
using v256 = union
{
	__m256i mm;
	u64x4 u64;
};


void print_v128(v128 a) {
	printf("{%016lx, %016lx}\n", a.u64[1], a.u64[0]); // print in big endian
}
void print_v256(v256 a) {
	printf("{%016lx, %016lx, %016lx, %016lx}\n", a.u64[3], a.u64[2], a.u64[1], a.u64[0]); // print in big endian
}

// Stolen from klondike: https://github.com/grocid/weight-4-polynomial-finder/blob/master/polyfinder/gf2_monomial.cpp#L12
inline u32 get_degree(v128 px)
{
	if (px.u64[1] != 0)
		return 127-__builtin_clzll(px.u64[1]);
	else
		return 63-__builtin_clzll(px.u64[0]|1);
}

inline u32 get_degree_256(v256 px)
{
	if (px.u64[3] != 0)
		return 255-__builtin_clzll(px.u64[3]);
	else if (px.u64[2] != 0)
		return 191-__builtin_clzll(px.u64[2]);
	else if (px.u64[1] != 0)
		return 127-__builtin_clzll(px.u64[1]);
	else
		return 63-__builtin_clzll(px.u64[0]|1);
}


// a*b, 128bit input, 256 bit output
inline v256 gf2x_karatsuba(v128 a, v128 b)
{
	v128 z0, z1, z2, t1, t2;
	z0.mm = _mm_clmulepi64_si128(a.mm, b.mm, 0x00);
	z2.mm = _mm_clmulepi64_si128(a.mm, b.mm, 0x11);
	t1.u64[0] = a.u64[0] ^ a.u64[1];
	t2.u64[0] = b.u64[0] ^ b.u64[1];
	z1.mm = _mm_clmulepi64_si128(t1.mm, t2.mm, 0x00);
	z1.u64 ^= z0.u64 ^ z2.u64;
	v256 out;
	out.u64[0] = z0.u64[0];
	out.u64[1] = z0.u64[1] ^ z1.u64[0];
	out.u64[2] = z1.u64[1] ^ z2.u64[0];
	out.u64[3] = z2.u64[1];
	return out;
}
// a², 128 bit input, 256 bit output
inline v256 gf2x_square(v128 a)
{
	v128 z0, z2;
	z0.mm = _mm_clmulepi64_si128(a.mm, a.mm, 0x00);
	z2.mm = _mm_clmulepi64_si128(a.mm, a.mm, 0x11);
	v256 out;
	out.u64[0] = z0.u64[0];
	out.u64[1] = z0.u64[1];
	out.u64[2] = z2.u64[0];
	out.u64[3] = z2.u64[1];
	return out;
}
// right shift, this is really slow and needs to be faster
inline v256 shr_v256(v256 in, u32 n)
{
	for (int i = 0; i < 3; ++i)
	{
		if (n >= 64)
		{
			in.u64[0] = in.u64[1];
			in.u64[1] = in.u64[2];
			in.u64[2] = in.u64[3];
			in.u64[3] = 0;
			n -= 64;
		}
	}
	v256 a  = {0}, b = {0};
	a.mm = _mm256_srli_epi64(in.mm, n);
	b.mm = _mm256_slli_epi64(in.mm, 64-n);
	v256 out = {(long long)b.u64[1], (long long)b.u64[2], (long long)b.u64[3], 0};
	out.u64 |= a.u64;
	return out;
}
// left shift, this is really slow and needs to be faster
inline v256 shl_v256(v256 in, u32 n)
{
	for (int i = 0; i < 3; ++i)
	{
		if (n >= 64)
		{
			in.u64[3] = in.u64[2];
			in.u64[2] = in.u64[1];
			in.u64[1] = in.u64[0];
			in.u64[0] = 0;
			n -= 64;
		}
	}
	v256 a  = {0}, b = {0};
	a.mm = _mm256_slli_epi64(in.mm, n);
	b.mm = _mm256_srli_epi64(in.mm, 64-n);
	v256 out = {0, (long long)b.u64[0], (long long)b.u64[1], (long long)b.u64[2]};
	out.u64 |= a.u64;
	return out;
}

// Calculates a // b (the quotient of a / b)
// This function should only be used once because the modulus is assumed to be constant throughout the program, so speed is not important
inline v128 gf2x_divide(v256 a, v128 b)
{
	const u32 a_deg = get_degree_256(a);
	const u32 b_deg = get_degree(b);

	v256 t = {(long long)b.u64[0], (long long)b.u64[1], 0, 0};
	v128 out = {0, 0};

	t = shl_v256(t, a_deg - b_deg);
	for (int i = a_deg - b_deg; i >= 0; --i)
	{
		if (get_degree_256(a) == get_degree_256(t))
		{
			a.u64 ^= t.u64;
			out.u64[i / 64] |= (u64)1 << (i % 64);
		}
		t = shr_v256(t, 1);
	}
	return out;
}

// This calculates in(x) mod P(x) with just multiplications and additions
// P(x) is assumed to be constant and that's why this can optimize it
inline v128 gf2x_reduce(v256 in, const v128 INV, const v128 P, const u32 p_deg)
{
	v256 t = shr_v256(in, p_deg); // Shift right, needs to be super fast here

	v128 out = {(long long)t.u64[0], (long long)t.u64[1]};
	t = gf2x_karatsuba(out, INV);
	t = shr_v256(t, p_deg); // Shift right, needs to be super fast here

	out.u64[0] = t.u64[0];
	out.u64[1] = t.u64[1];
	t = gf2x_karatsuba(out, P);
	t.u64 ^= in.u64;

	out.u64[0] = t.u64[0];
	out.u64[1] = t.u64[1];

	return out;
}

inline v128 gf2x_exp(v128 base, v128 n, const v128 INV, const v128 P, const u32 p_deg)
{
	v128 out = {1, 0};
	v128 t = {(long long)base.u64[0], (long long)base.u64[1]};

	int i = 0;
	while ((n.u64[1] != 0) || (n.u64[0] != 0))
	{
		if (n.u64[i / 64] & 1)
			out = gf2x_reduce(gf2x_karatsuba(out, t), INV, P, p_deg);
		t = gf2x_reduce(gf2x_square(t), INV, P, p_deg);
		n.u64[i / 64] >>= 1;
		i++;
	}
	return out;
}
