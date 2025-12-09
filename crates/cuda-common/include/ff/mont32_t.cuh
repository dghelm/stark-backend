/*
 * Source: https://github.com/supranational/sppark (tag=v0.1.12)
 * Status: MODIFIED from sppark/ff/mont32_t.cuh
 * Imported: 2025-08-13 by @gaxiom
 *
 * LOCAL CHANGES (high level):
 * - 2025-09-19: disable asm for static analysis
 */

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(__SPPARK_FF_MONT32_T_CUH__)
#define __SPPARK_FF_MONT32_T_CUH__

# include <cstdint>

// HIP compatibility: include device functions header BEFORE redefining inline
# if defined(__HIPCC__)
#  include <hip/amd_detail/amd_device_functions.h>
# endif

// HIP/CUDA compatibility: inline qualifier for device functions
# if defined(__HIPCC__)
#  define inline __device__ __attribute__((always_inline)) inline
// HIP doesn't support PTX inline assembly - use C++ fallbacks
#  define MONT32_USE_CPP_FALLBACK 1
# else
#  define inline __device__ __forceinline__
# endif

// asm macro for CUDA (HIP uses C++ fallbacks instead)
# ifndef MONT32_USE_CPP_FALLBACK
#  ifdef __clang_analyzer__
#   define asm(...) ({ __builtin_trap(); })
#  elif __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
# endif

// C++ fallback helpers for HIP (portable implementations)
# ifdef MONT32_USE_CPP_FALLBACK
namespace mont32_hip_compat {
    // 32x32 -> 64 bit multiplication, returns {lo, hi}
    __device__ __attribute__((always_inline)) inline
    void mul_wide(uint32_t a, uint32_t b, uint32_t& lo, uint32_t& hi) {
        uint64_t prod = (uint64_t)a * b;
        lo = (uint32_t)prod;
        hi = (uint32_t)(prod >> 32);
    }

    // Multiply-add: {lo, hi} = a * b + c, returns carry
    __device__ __attribute__((always_inline)) inline
    uint32_t mad_wide(uint32_t a, uint32_t b, uint32_t c, uint32_t& lo, uint32_t& hi) {
        uint64_t prod = (uint64_t)a * b + c;
        lo = (uint32_t)prod;
        hi = (uint32_t)(prod >> 32);
        return 0; // No overflow possible with single add
    }

    // Multiply-add with accumulate: {lo, hi} += a * b
    __device__ __attribute__((always_inline)) inline
    void mad_wide_acc(uint32_t a, uint32_t b, uint32_t& lo, uint32_t& hi) {
        uint64_t acc = ((uint64_t)hi << 32) | lo;
        acc += (uint64_t)a * b;
        lo = (uint32_t)acc;
        hi = (uint32_t)(acc >> 32);
    }

    // Funnel shift right: (hi:lo) >> shift
    __device__ __attribute__((always_inline)) inline
    uint32_t shf_r_wrap(uint32_t lo, uint32_t hi, uint32_t shift) {
        uint64_t combined = ((uint64_t)hi << 32) | lo;
        return (uint32_t)(combined >> (shift & 31));
    }

    // Add with carry out
    __device__ __attribute__((always_inline)) inline
    uint32_t add_cc(uint32_t a, uint32_t b, uint32_t& carry) {
        uint64_t sum = (uint64_t)a + b + carry;
        carry = (uint32_t)(sum >> 32);
        return (uint32_t)sum;
    }
}
# endif // MONT32_USE_CPP_FALLBACK

template<const size_t N, const uint32_t MOD, const uint32_t M0,
         const uint32_t RR, const uint32_t ONE>
class mont32_t {
private:
    uint32_t val;

public:
    using mem_t = mont32_t;
    static const uint32_t degree = 1;
    static constexpr size_t __device__ bit_length()     { return N;  }

    inline uint32_t& operator[](size_t i)               { return val; }
    inline uint32_t& operator*()                        { return val; }
    inline const uint32_t& operator[](size_t i) const   { return val; }
    inline uint32_t operator*() const                   { return val; }
    inline size_t len() const                           { return 1;   }

    inline mont32_t() {}
    inline mont32_t(const uint32_t *p)      { val = *p; }
    // this is used in constant declaration, e.g. as mont32_t{11}
    __host__ __device__ constexpr mont32_t(int a)       : val(((uint64_t)a << 32) % MOD) {}
    __host__ __device__ constexpr mont32_t(uint32_t a)  : val(a) {}

    inline operator uint32_t() const        { return mul_by_1(); }
    inline void store(uint32_t *p) const    { *p = mul_by_1();   }
    inline mont32_t& operator=(uint32_t b)  { val = b; to(); return *this; }

    inline mont32_t& operator+=(const mont32_t b)
    {
        val += b.val;
        if (N == 32) {
            if (val < b.val || val >= MOD)  val -= MOD;
        } else {
            if (val >= MOD)                 val -= MOD;
        }

        return *this;
    }
    friend inline mont32_t operator+(mont32_t a, const mont32_t b)
    {   return a += b;   }

    inline mont32_t& operator<<=(uint32_t l)
    {
        if (N == 32) {
            while (l--) {
                bool carry = val >> 31;
                val <<= 1;
                if (carry || val >= MOD)    val -= MOD;
            }
        } else {
            while (l--) {
                val <<= 1;
                if (val >= MOD)             val -= MOD;
            }
        }

        return *this;
    }
    friend inline mont32_t operator<<(mont32_t a, uint32_t l)
    {   return a <<= l;   }

    inline mont32_t& operator>>=(uint32_t r)
    {
        while (r >= 32) {
            val = mul_by_1();
            r -= 32;
	}

        if (r > 2) {
            uint32_t lo, hi, red = (val * M0) & ((1<<r) - 1);

#ifdef MONT32_USE_CPP_FALLBACK
            mont32_hip_compat::mad_wide(red, MOD, val, lo, hi);
            val = mont32_hip_compat::shf_r_wrap(lo, hi, r);
#else
            asm("mad.lo.cc.u32 %0, %2, %3, %4; madc.hi.u32 %1, %2, %3, 0;"
                : "=r"(lo), "=r"(hi) : "r"(red), "r"(MOD), "r"(val));
            asm("shf.r.wrap.b32 %0, %1, %2, %3;"
                : "=r"(val) : "r"(lo), "r"(hi), "r"(r));
#endif
	} else if (N == 32) {
            while (r--) {
                uint32_t tmp = val&1 ? MOD : 0;

#ifdef MONT32_USE_CPP_FALLBACK
                uint32_t carry = 0;
                val = mont32_hip_compat::add_cc(val, tmp, carry);
                val = mont32_hip_compat::shf_r_wrap(val, carry, 1);
#else
                asm("add.cc.u32 %0, %0, %1;"        : "+r"(val) : "r"(tmp));
                asm("addc.u32   %0, 0, 0;"          : "=r"(tmp));
                asm("shf.r.wrap.b32 %0, %0, %1, 1;" : "+r"(val) : "r"(tmp));
#endif
            }
        } else {
            while (r--) {
                if (val&1)  val += MOD;
                val >>= 1;
            }
        }

        return *this;
    }
    friend inline mont32_t operator>>(mont32_t a, uint32_t r)
    {   return a >>= r;   }

    inline mont32_t& operator-=(const mont32_t b)
    {
#ifdef MONT32_USE_CPP_FALLBACK
        if (val < b.val) {
            val = val - b.val + MOD;
        } else {
            val = val - b.val;
        }
#else
        asm("{");
        asm(".reg.pred %brw;");
        asm("setp.lt.u32 %brw, %0, %1;" :: "r"(val), "r"(b.val));
        asm("sub.u32 %0, %0, %1;"       : "+r"(val) : "r"(b.val));
        asm("@%brw add.u32 %0, %0, %1;" : "+r"(val) : "r"(MOD));
        asm("}");
#endif

        return *this;
    }
    friend inline mont32_t operator-(mont32_t a, const mont32_t b)
    {   return a -= b;   }

    inline mont32_t cneg(bool flag)
    {
#ifdef MONT32_USE_CPP_FALLBACK
        if (val != 0 && flag) {
            val = MOD - val;
        }
#else
        asm("{");
        asm(".reg.pred %flag;");
        asm("setp.ne.u32 %flag, %0, 0;" :: "r"(val));
        asm("@%flag setp.ne.u32 %flag, %0, 0;" :: "r"((int)flag));
        asm("@%flag sub.u32 %0, %1, %0;" : "+r"(val) : "r"(MOD));
        asm("}");
#endif

        return *this;
    }
    static inline mont32_t cneg(mont32_t a, bool flag)
    {   return a.cneg(flag);   }
    inline mont32_t operator-() const
    {   return cneg(*this, true);   }

    static inline const mont32_t one()  { return mont32_t{ONE}; }
    inline bool is_one() const          { return val == ONE;    }
    inline bool is_zero() const         { return val == 0;      }
    inline void zero()                  { val = 0;              }

    friend inline mont32_t czero(const mont32_t a, int set_z)
    {
        mont32_t ret;

#ifdef MONT32_USE_CPP_FALLBACK
        ret.val = set_z ? 0 : a.val;
#else
        asm("{");
        asm(".reg.pred %set_z;");
        asm("setp.ne.s32 %set_z, %0, 0;" : : "r"(set_z));
        asm("selp.u32 %0, 0, %1, %set_z;" : "=r"(ret.val) : "r"(a.val));
        asm("}");
#endif

        return ret;
    }

    static inline mont32_t csel(const mont32_t a, const mont32_t b, int sel_a)
    {
        mont32_t ret;

#ifdef MONT32_USE_CPP_FALLBACK
        ret.val = sel_a ? a.val : b.val;
#else
        asm("{");
        asm(".reg.pred %sel_a;");
        asm("setp.ne.s32 %sel_a, %0, 0;" :: "r"(sel_a));
        asm("selp.u32 %0, %1, %2, %sel_a;" : "=r"(ret.val) : "r"(a.val), "r"(b.val));
        asm("}");
#endif

        return ret;
    }

private:
    static inline uint32_t final_sub(uint32_t val)
    {
#ifdef MONT32_USE_CPP_FALLBACK
        // Simple modular reduction - no carry tracking needed for C++ version
        if (val >= MOD) {
            val -= MOD;
        }
        return val;
#else
        asm("{");
        asm(".reg.pred %p;");
        if (N == 32) {
            uint32_t carry;

            asm("addc.u32 %0, 0, 0;"            : "=r"(carry));
            asm("setp.lt.u32 %p, %0, %1;"       :: "r"(val), "r"(MOD));
            asm("@%p  setp.eq.u32 %p, %0, 0;"   :: "r"(carry));
            asm("@!%p sub.u32 %0, %0, %1;"      : "+r"(val) : "r"(MOD));
        } else {
            asm("setp.ge.u32 %p, %0, %1;" :: "r"(val), "r"(MOD));
            asm("@%p sub.u32 %0, %0, %1;" : "+r"(val) : "r"(MOD));
        }
        asm("}");

        return val;
#endif
    }

    inline mont32_t& mul(const mont32_t b)
    {
        uint32_t tmp[2], red;

#ifdef MONT32_USE_CPP_FALLBACK
        // Montgomery multiplication using C++
        mont32_hip_compat::mul_wide(val, b.val, tmp[0], tmp[1]);
        red = tmp[0] * M0;
        mont32_hip_compat::mad_wide_acc(red, MOD, tmp[0], tmp[1]);
#else
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(tmp[0]), "=r"(tmp[1])
            : "r"(val), "r"(b.val));
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M0));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(tmp[0]), "+r"(tmp[1])
            : "r"(red), "r"(MOD));
#endif

        val = final_sub(tmp[1]);

        return *this;
    }

    inline uint32_t mul_by_1() const
    {
        uint32_t tmp[2], red;

#ifdef MONT32_USE_CPP_FALLBACK
        red = val * M0;
        mont32_hip_compat::mad_wide(red, MOD, val, tmp[0], tmp[1]);
#else
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(val), "r"(M0));
        asm("mad.lo.cc.u32 %0, %2, %3, %4; madc.hi.u32 %1, %2, %3, 0;"
            : "=r"(tmp[0]), "=r"(tmp[1])
            : "r"(red), "r"(MOD), "r"(val));
#endif

        return tmp[1];
    }

public:
    friend inline mont32_t operator*(mont32_t a, const mont32_t b)
    {   return a.mul(b);   }
    inline mont32_t& operator*=(const mont32_t a)
    {   return mul(a);   }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline mont32_t& operator^=(uint32_t p)
    {
        mont32_t sqr = *this;
        *this = csel(val, ONE, p&1);

        #pragma unroll 1
        while (p >>= 1) {
            sqr.mul(sqr);
            if (p&1)
                mul(sqr);
        }

        return *this;
    }
    friend inline mont32_t operator^(mont32_t a, uint32_t p)
    {   return a ^= p;   }
    inline mont32_t operator()(uint32_t p)
    {   return *this^p;   }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline mont32_t& operator^=(int p)
    {
        if (p < 2)
#ifdef MONT32_USE_CPP_FALLBACK
            __builtin_trap();
#else
            asm("trap;");
#endif

        if (p == 7) {
            mont32_t temp = sqr_n_mul(*this, 1, *this);
            *this = sqr_n_mul(temp, 1, *this);
            return *this;
        }

        mont32_t sqr = *this;
        if ((p&1) == 0) {
            do {
                sqr.mul(sqr);
                p >>= 1;
            } while ((p&1) == 0);
            *this = sqr;
        }
        for (p >>= 1; p; p >>= 1) {
            sqr.mul(sqr);
            if (p&1)
                mul(sqr);
        }

        return *this;
    }
    friend inline mont32_t operator^(mont32_t a, int p)
    {   return a ^= p;   }
    inline mont32_t operator()(int p)
    {   return *this^p;   }
    friend inline mont32_t sqr(mont32_t a)
    {   return a.sqr();   }
    inline mont32_t& sqr()
    {   return mul(*this);   }

    inline void to()   { mul(RR); }
    inline void from() { val = mul_by_1(); }

    template<size_t T>
    static inline mont32_t dot_product(const mont32_t a[T], const mont32_t b[T])
    {
        uint32_t acc[2];

#ifdef MONT32_USE_CPP_FALLBACK
        mont32_hip_compat::mul_wide(*a[0], *b[0], acc[0], acc[1]);

        for (size_t i = 1; i < T; i++) {
            mont32_hip_compat::mad_wide_acc(*a[i], *b[i], acc[0], acc[1]);
            if (N == 32 || (N != 32 && i == T-1))
                acc[1] = final_sub(acc[1]);
        }

        uint32_t red = acc[0] * M0;
        mont32_hip_compat::mad_wide_acc(red, MOD, acc[0], acc[1]);
#else
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[0]), "=r"(acc[1]) : "r"(*a[0]), "r"(*b[0]));

        if (N == 32) {
            for (size_t i = 1; i < T; i++) {
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[i]));
                acc[1] = final_sub(acc[1]);
            }
        } else {
            size_t i = 1;

            if ((T&1) == 0) {
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[i]));
                i++;
            }
            for (; i < T; i += 2) {
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[i]));
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i+1]), "r"(*b[i+1]));
                acc[1] = final_sub(acc[1]);
            }
        }

        uint32_t red;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M0));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1]) : "r"(red), "r"(MOD));
#endif

        return final_sub(acc[1]);
    }

    template<size_t T>
    static inline mont32_t dot_product(mont32_t a0, mont32_t b0,
                                       const mont32_t a[T-1], const mont32_t *b,
                                       size_t stride_b = 1)
    {
        uint32_t acc[2];

#ifdef MONT32_USE_CPP_FALLBACK
        mont32_hip_compat::mul_wide(*a0, *b0, acc[0], acc[1]);

        for (size_t i = 0; i < T-1; i++, b += stride_b) {
            mont32_hip_compat::mad_wide_acc(*a[i], *b[0], acc[0], acc[1]);
            if (N == 32 || (N != 32 && i == T-2))
                acc[1] = final_sub(acc[1]);
        }

        uint32_t red = acc[0] * M0;
        mont32_hip_compat::mad_wide_acc(red, MOD, acc[0], acc[1]);
#else
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[0]), "=r"(acc[1]) : "r"(*a0), "r"(*b0));

        if (N == 32) {
            for (size_t i = 0; i < T-1; i++, b += stride_b) {
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[0]));
                acc[1] = final_sub(acc[1]);
            }
        } else {
            size_t i = 0;

            if ((T&1) == 0) {
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[0]));
                i++, b += stride_b;
            }
            for (; i < T-1; i += 2) {
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[0]));
                b += stride_b;
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                    : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i+1]), "r"(*b[0]));
                b += stride_b;
                acc[1] = final_sub(acc[1]);
            }
        }

        uint32_t red;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M0));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1]) : "r"(red), "r"(MOD));
#endif

        return final_sub(acc[1]);
    }

    inline mont32_t reciprocal() const
    {   return *this ^ (MOD-2);   }
    friend inline mont32_t operator/(int one, mont32_t a)
    {
#ifdef MONT32_USE_CPP_FALLBACK
        if (one != 1) __builtin_trap();
#else
        if (one != 1) asm("trap;");
#endif
        return a.reciprocal();
    }
    friend inline mont32_t operator/(mont32_t a, mont32_t b)
    {   return a * b.reciprocal();   }
    inline mont32_t& operator/=(const mont32_t a)
    {   return *this *= a.reciprocal();   }

    inline void shfl_bfly(uint32_t laneMask)
    {
#ifdef MONT32_USE_CPP_FALLBACK
        val = __shfl_xor(val, laneMask);
#else
        val = __shfl_xor_sync(0xFFFFFFFF, val, laneMask);
#endif
    }

protected:
    static inline mont32_t sqr_n(mont32_t s, uint32_t n)
    {
#ifdef MONT32_USE_CPP_FALLBACK
        // For HIP, use the generic sqr() approach
        #pragma unroll 4
        while (n--)
            s.sqr();
#else
        if (N == 32 || M0 > MOD) {
            #pragma unroll 4
            while (n--)
                s.sqr();
        } else {    // +20% [for bb31_t::reciprocal()]
            #pragma unroll 4
            while (n--) {
                uint32_t tmp[2], red;

                asm("mul.lo.u32 %0, %2, %2; mul.hi.u32 %1, %2, %2;"
                    : "=r"(tmp[0]), "=r"(tmp[1])
                    : "r"(s.val));
                asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M0));
                asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
                    : "+r"(tmp[0]), "=r"(s.val)
                    : "r"(red), "r"(MOD), "r"(tmp[1]));

                if (n&1)
                    s.val = final_sub(s.val);
            }
        }
#endif

        return s;
    }

    static inline mont32_t sqr_n_mul(mont32_t s, uint32_t n, mont32_t m)
    {
        s = sqr_n(s, n);
        s.mul(m);

        return s;
    }

# undef inline
# ifndef MONT32_USE_CPP_FALLBACK
#  undef asm
# endif

public:
    friend inline bool operator==(mont32_t a, mont32_t b)
    {   return a.val == b.val;   }
    friend inline bool operator!=(mont32_t a, mont32_t b)
    {   return a.val != b.val;   }

# if defined(_GLIBCXX_IOSTREAM) || defined(_IOSTREAM_) // non-standard
    friend std::ostream& operator<<(std::ostream& os, const mont32_t& obj)
    {
        auto f = os.flags();
        uint32_t red = obj.val * M0;
        uint64_t v = obj.val + red * (uint64_t)MOD;
        os << "0x" << std::hex << (uint32_t)(v >> 32);
        os.flags(f);
        return os;
    }
# endif
};

#endif /* __SPPARK_FF_MONT32_T_CUH__ */
