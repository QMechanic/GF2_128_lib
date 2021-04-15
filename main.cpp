#include <bits/stdc++.h>
#include "gf2x_arithmetic.cpp"

#define POLY (v128){(long long)299733420677929, (long long)257}

int main(int argc, char const *argv[])
{
	const u32 P_DEG = get_degree(POLY);

	const v256 POW2 = shl_v256((v256){1, 0, 0, 0}, 2*P_DEG);
	const v128 INV = gf2x_divide(POW2, POLY);

	v128 a = {(long long)0xc3fbd3bcc67fa5cb, (long long)0x00000000000001e6};
	v128 n = {(long long)0xff47ab90b3b81c55, (long long)0x00000000000001d0};

	v128 b = {1, 0};

	for (u32 i = 0; i < 100000000; i++) {
		b = gf2x_reduce(gf2x_mul(b, a), INV, POLY, P_DEG);
	}
	print_v128(b); // {00000000000000f1, 3628488fbc64ebf9}

	for (u32 i = 0; i < 1000000; i++) {
		a = gf2x_exp(a, n, INV, POLY, P_DEG);
	}
	print_v128(a); // {00000000000000bf, e95810315d4271e9}

	return 0;
}
