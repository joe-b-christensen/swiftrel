// swiftrel_fmt.h -- exact, fast replacements for `ostream << double` (= printf "%.6g") as used by
// vcftools --relatedness2.  Part of SwiftRel (LGPL-3.0, derived from vcftools 0.1.17).
//
// fmt_g6(p, v) writes exactly the bytes snprintf(buf, n, "%.6g", v) writes (glibc: exact binary value,
// round-half-even), without the locale / varargs / parsing overhead. Uncommon magnitudes fall back to snprintf.
#ifndef SWIFTREL_FMT_H
#define SWIFTREL_FMT_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace swiftrel {

// two-digit lookup table
static const char kDigits2[201] =
    "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
    "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899";

// Write an unsigned integer < 10^10. Returns the end pointer.
static inline char *fmt_u32(char *p, uint32_t v) {
    char tmp[12];
    char *e = tmp + 12, *q = e;
    while (v >= 100) {
        uint32_t r = v % 100;
        v /= 100;
        q -= 2;
        memcpy(q, kDigits2 + 2 * r, 2);
    }
    if (v >= 10) {
        q -= 2;
        memcpy(q, kDigits2 + 2 * v, 2);
    } else {
        *--q = (char)('0' + v);
    }
    size_t n = (size_t)(e - q);
    memcpy(p, q, n);
    return p + n;
}

static inline char *fmt_g6_slow(char *p, double v) {
    char buf[64];
    int n = snprintf(buf, sizeof buf, "%.6g", v);
    memcpy(p, buf, (size_t)n);
    return p + n;
}

// A count (non-negative integer stored in a double): "%.6g" prints it as an integer below 1e6.
static inline char *fmt_count(char *p, double v) {
    if (v >= 0.0 && v < 1e6) return fmt_u32(p, (uint32_t)v);
    return fmt_g6_slow(p, v);
}

static inline char *fmt_count_u(char *p, uint64_t v) {
    if (v < 1000000u) return fmt_u32(p, (uint32_t)v);
    return fmt_g6_slow(p, (double)v);
}

static const uint64_t kPow10[20] = {1ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull, 10000000ull,
                                    100000000ull, 1000000000ull, 10000000000ull, 100000000000ull,
                                    1000000000000ull, 10000000000000ull, 100000000000000ull,
                                    1000000000000000ull, 10000000000000000ull, 100000000000000000ull,
                                    1000000000000000000ull, 10000000000000000000ull};

// floor(v * 10^k) and the rounding decision, exactly. v = m * 2^e (m < 2^53, e < 0), 0 <= k <= 19.
// Returns false if the shift is out of range (caller falls back).
static inline bool scaled_round6(uint64_t m, int e, int k, uint64_t &q_floor, int &cmp_half, bool &exact_zero_rem) {
    // x = m * 10^k * 2^e ; we need q = floor(x), and the remainder compared to 1/2.
    unsigned __int128 t = (unsigned __int128)m * kPow10[k];
    int s = -e;  // right shift
    if (s <= 0 || s >= 127) return false;
    unsigned __int128 q = t >> s;
    unsigned __int128 r = t - (q << s);
    unsigned __int128 half = (unsigned __int128)1 << (s - 1);
    if (q > (unsigned __int128)UINT64_MAX) return false;
    q_floor = (uint64_t)q;
    cmp_half = r < half ? -1 : (r == half ? 0 : 1);
    exact_zero_rem = (r == 0);
    return true;
}

// Exact "%.6g" for finite non-zero |v| in about [1e-5, 1e6); everything else goes through snprintf.
// No libm and few data-dependent branches: mantissa and exponent come from the IEEE bits, the decimal exponent is
// estimated and corrected with one comparison, the 6 digits come from exact 128-bit scaling (round-half-even on
// the exact binary value, like glibc), trailing zeros are counted without a loop.
static inline char *fmt_g6(char *p, double v) {
    uint64_t bits;
    memcpy(&bits, &v, 8);
    const int bexp = (int)((bits >> 52) & 0x7ff);
    const int e2 = bexp - 1023;  // floor(log2 |v|) for normal numbers
    if (bexp == 0 || bexp == 0x7ff || e2 < -17 || e2 > 19) return fmt_g6_slow(p, v);
    const uint64_t m = (bits & ((1ull << 52) - 1)) | (1ull << 52);  // |v| = m * 2^e
    const int e = bexp - 1075;
    static const double kP10d[14] = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6};
    double a;
    const uint64_t ab = bits & ~(1ull << 63);
    memcpy(&a, &ab, 8);
    int E = (e2 * 78913) >> 18;   // floor(e2 * log10(2)); floor(log10|v|) is E or E+1
    E += (a >= kP10d[E + 8]);     // E in [-6, 6]
    uint64_t q = 0;
    int ch = 0;
    bool z;
    int k = 5 - E;
    if (k < 0 || !scaled_round6(m, e, k, q, ch, z)) return fmt_g6_slow(p, v);
    if (__builtin_expect(q < 100000u || q >= 1000000u, 0)) {
        // the double comparison was off by one (|v| within an ulp of a power of ten): exact search
        for (int iter = 0; iter < 4; iter++) {
            if (q < 100000u) E--;
            else if (q >= 1000000u) E++;
            else break;
            k = 5 - E;
            if (k < 0 || k > 19 || !scaled_round6(m, e, k, q, ch, z)) return fmt_g6_slow(p, v);
        }
        if (q < 100000u || q >= 1000000u) return fmt_g6_slow(p, v);
    }
    uint32_t D = (uint32_t)q + (uint32_t)((ch > 0) | ((ch == 0) & (int)(q & 1)));
    int X = E;
    if (__builtin_expect(D == 1000000u, 0)) {
        D = 100000u;
        X++;
    }
    if (X < -4 || X >= 6) return fmt_g6_slow(p, v);  // scientific notation: rare here
    *p = '-';
    p += (bits >> 63);
    char dg[8];
    const uint32_t hi = D / 10000u, lo = D % 10000u;
    memcpy(dg, kDigits2 + 2 * hi, 2);
    memcpy(dg + 2, kDigits2 + 2 * (lo / 100u), 2);
    memcpy(dg + 4, kDigits2 + 2 * (lo % 100u), 2);
    dg[6] = dg[7] = '0';
    // D >= 100000, so at most 5 trailing zeros
    const int tz = (D % 10u == 0) + (D % 100u == 0) + (D % 1000u == 0) + (D % 10000u == 0) + (D % 100000u == 0);
    const int nd = 6 - tz;
    if (X < 0) {
        // 0.<-X-1 zeros><nd digits>
        memcpy(p, "0.000000", 8);
        memcpy(p + 1 - X, dg, 8);
        return p + 1 - X + nd;
    }
    const int ip = X + 1;  // integer digits
    memcpy(p, dg, 8);
    if (nd <= ip) return p + ip;
    memmove(p + ip + 1, dg + ip, (size_t)(nd - ip));
    p[ip] = '.';
    return p + nd + 1;
}

// phi = num / den with integer num and den (0 < den < 2^32, |num| < 2^31): v must be the double vcftools computes,
// fl(num / den). Writes exactly "%.6g" of v. The 6 significant digits are taken from the rational num/den with
// integer arithmetic: v differs from num/den by at most 2^-53 relative, and unless num/den lies exactly halfway
// between two 6-digit decimals its distance to that boundary is at least 10^-k / (2 den) > 2^-53 * 10^6 * 10^-k
// (den < 4.5e9), so v rounds the same way. Exact ties, |v| outside [1e-4, 1e5) and other inputs use fmt_g6.
static inline char *fmt_phi(char *p, int64_t num, uint64_t den, double v) {
    static const double kP10k[10] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9};
    static const double kP10d[14] = {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6};
    const uint64_t n = (uint64_t)(num < 0 ? -num : num);
    const double a = v < 0 ? -v : v;
    if (!(a >= 1e-4 && a < 1e5) || n >= (1ull << 31) || den == 0 || den >= (1ull << 32)) return fmt_g6(p, v);
    uint64_t bits;
    memcpy(&bits, &a, 8);
    const int e2 = (int)((bits >> 52) & 0x7ff) - 1023;
    int E = (e2 * 78913) >> 18;  // floor(e2 * log10(2)); floor(log10 a) is E or E+1
    E += (a >= kP10d[E + 8]);
    int k = 5 - E;  // want 1e5 * den <= n * 10^k < 1e6 * den
    if (k < 0 || k > 9) return fmt_g6(p, v);
    uint64_t t = n * kPow10[k];
    if (__builtin_expect(t < 100000u * den, 0)) {
        if (++k > 9) return fmt_g6(p, v);
        t = n * kPow10[k];
        E--;
    } else if (__builtin_expect(t >= 1000000u * den, 0)) {
        if (--k < 0) return fmt_g6(p, v);
        t = n * kPow10[k];
        E++;
    }
    // floor(t / den) from the double quotient, corrected by the exact remainder
    int64_t m0 = (int64_t)(a * kP10k[k]);
    int64_t rem = (int64_t)t - m0 * (int64_t)den;
    if (rem < 0) {
        m0--;
        rem += (int64_t)den;
        if (rem < 0) return fmt_g6(p, v);
    } else if (rem >= (int64_t)den) {
        m0++;
        rem -= (int64_t)den;
        if (rem >= (int64_t)den) return fmt_g6(p, v);
    }
    const uint64_t r2 = 2 * (uint64_t)rem;
    if (__builtin_expect(r2 == den, 0)) return fmt_g6(p, v);  // exact tie of num/den: the binary value decides
    uint32_t D = (uint32_t)m0 + (uint32_t)(r2 > den);
    int X = E;
    if (__builtin_expect(D == 1000000u, 0)) {
        D = 100000u;
        X++;
    }
    if (X < -4 || X >= 6) return fmt_g6(p, v);
    *p = '-';
    p += (num < 0);
    char dg[8];
    const uint32_t hi = D / 10000u, lo = D % 10000u;
    memcpy(dg, kDigits2 + 2 * hi, 2);
    memcpy(dg + 2, kDigits2 + 2 * (lo / 100u), 2);
    memcpy(dg + 4, kDigits2 + 2 * (lo % 100u), 2);
    dg[6] = dg[7] = '0';
    const int tz = (D % 10u == 0) + (D % 100u == 0) + (D % 1000u == 0) + (D % 10000u == 0) + (D % 100000u == 0);
    const int nd = 6 - tz;
    if (X < 0) {
        memcpy(p, "0.000000", 8);
        memcpy(p + 1 - X, dg, 8);
        return p + 1 - X + nd;
    }
    const int ip = X + 1;
    memcpy(p, dg, 8);
    if (nd <= ip) return p + ip;
    memmove(p + ip + 1, dg + ip, (size_t)(nd - ip));
    p[ip] = '.';
    return p + nd + 1;
}

}  // namespace swiftrel

#endif
