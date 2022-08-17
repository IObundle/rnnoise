#include "opus_types.h"
#include <math.h>
#include <limits.h>

#define PSHR32(a, shift) (SHR32((a)+((EXTEND32(1)<<((shift))>>1)),shift))

#define SHR32(a, shift) ((a)>>(shift))

#define EXTEND32(x) ((opus_val32)(x))

#define SHL32(a, shift) ((opus_int32)((opus_uint32)(a)) << (shift))

#define MULT16_16(a,b) (((opus_val32)(opus_val16)(a))*((opus_val32)(opus_val16)(b)))

#define MULT16_16_Q15(a,b) (SHR(MULT16_16((a),(b)),15))

#define SHR(a, shift) ((a) >> (shift))

#define shr(a, shift) ((a) >> (shift))


#define MAC16_16(c,a,b) (ADD32((c),MULT16_16((a),(b))))

#define ADD32(a,b) ((opus_val32)(a)+(opus_val32)(b))

#define ROUND16(x,a) (EXTRACT16(PSHR32((x),(a))))

#define SROUND16(x,a) EXTRACT16(SATURATE(PSHR32(x,a), 32767))

#define SATURATE(x,a) (((x)>(a) ? (a) : (x)<-(a) ? -(a) : (x)))

#define EXTRACT16(x) ((opus_val16)(x))

#define MULT32_32_Q31(a,b) ADD32(ADD32(SHL(MULT16_16(SHR((a),16),shr((b),16)),1), SHR(MULT16_16SU(SHR((a),16),((b)&0x0000ffff)),15)), SHR(MULT16_16SU(SHR((b),16),((a)&0x0000ffff)),15))


#define MULT16_16SU(a,b) ((opus_val32)(opus_val16)(a)*(opus_val32)(opus_uint16)(b))


#define SHL(a,shift) SHL32(a,shift)

#define MULT16_32_Q15(a,b) ADD32(SHL(MULT16_16((a),SHR((b),16)),1), SHR(MULT16_16SU((a),((b)&0x0000ffff)),15))


#define QCONST16(x,bits) ((opus_val16)(.5+(x)*(((opus_val32)1)<<(bits))))

#define HALF16(x) (SHR16(x,1))

#define HALF32(x) (SHR32(x,1))

#define SHR16(a,shift) ((a) >> (shift))

#define VSHR32(a,shift) (((shift)>0) ? SHR32(a,shift) : SHL32(a, -(shift)))

#define MULT16_32_Q16(a,b) ADD32(MULT16_16((a),SHR((b),16)), SHR(MULT16_16SU((a),((b)&0x0000ffff)),16))

#define SUB32_ovflw(a,b) ((opus_val32)((opus_uint32)(a)-(opus_uint32)(b)))

#define ADD32_ovflw(a,b) ((opus_val32)((opus_uint32)(a)+(opus_uint32)(b)))

#define NEG32_ovflw(a) ((opus_val32)(0-(opus_uint32)(a)))

#define DIV32(a,b) (((opus_val32)(a))/((opus_val32)(b)))

#define PI 3.141592653f
#define celt_sqrt(x) ((float)sqrt(x))
#define celt_rsqrt(x) (1.f/celt_sqrt(x))
#define celt_rsqrt_norm(x) (celt_rsqrt(x))
#define celt_cos_norm(x) ((float)cos((.5f*PI)*(x)))

  static inline opus_val32 celt_maxabs32(opus_val32 *x, int len)
{
   int i;
   opus_val32 maxval = 0;
   for (i=0;i<len;i++)
      maxval = MAX32(maxval, ABS32(x[i]));
   return maxval;
}
static inline opus_val32 celt_maxabs16(const opus_val16 *x, int len)
{
   int i;
   opus_val16 maxval = 0;
   opus_val16 minval = 0;
   for (i=0;i<len;i++)
   {
      maxval = MAX16(maxval, x[i]);
      minval = MIN16(minval, x[i]);
   }
   return MAX32(EXTEND32(maxval),-EXTEND32(minval));
}




#if !defined(_ecintrin_H)
# define _ecintrin_H (1)
/*Some specific platforms may have optimized intrinsic or inline assembly
   versions of these functions which can substantially improve performance.
  We define macros for them to allow easy incorporation of these non-ANSI
   features.*/
/*Note that we do not provide a macro for abs(), because it is provided as a
   library function, which we assume is translated into an intrinsic to avoid
   the function call overhead and then implemented in the smartest way for the
   target platform.
  With modern gcc (4.x), this is true: it uses cmov instructions if the
   architecture supports it and branchless bit-twiddling if it does not (the
   speed difference between the two approaches is not measurable).
  Interestingly, the bit-twiddling method was patented in 2000 (US 6,073,150)
   by Sun Microsystems, despite prior art dating back to at least 1996:
   http://web.archive.org/web/19961201174141/www.x86.org/ftp/articles/pentopt/PENTOPT.TXT
  On gcc 3.x, however, our assumption is not true, as abs() is translated to a
   conditional jump, which is horrible on deeply piplined architectures (e.g.,
   all consumer architectures for the past decade or more) when the sign cannot
   be reliably predicted.*/
/*Modern gcc (4.x) can compile the naive versions of min and max with cmov if
   given an appropriate architecture, but the branchless bit-twiddling versions
   are just as fast, and do not require any special target architecture.
  Earlier gcc versions (3.x) compiled both code to the same assembly
   instructions, because of the way they represented ((_b)>(_a)) internally.*/
# define EC_MINI(_a,_b)      ((_a)+(((_b)-(_a))&-((_b)<(_a))))
/*Count leading zeros.
  This macro should only be used for implementing ec_ilog(), if it is defined.
  All other code should use EC_ILOG() instead.*/
#if defined(_MSC_VER)
# include <intrin.h>
/*In _DEBUG mode this is not an intrinsic by default.*/
# pragma intrinsic(_BitScanReverse)
static __inline int ec_bsr(unsigned long _x){
  unsigned long ret;
  _BitScanReverse(&ret,_x);
  return (int)ret;
}
# define EC_CLZ0    (1)
# define EC_CLZ(_x) (-ec_bsr(_x))
#elif defined(ENABLE_TI_DSPLIB)
# include "dsplib.h"
# define EC_CLZ0    (31)
# define EC_CLZ(_x) (_lnorm(_x))
#elif defined(__GNUC_PREREQ)
# if __GNUC_PREREQ(3,4)
#  if INT_MAX>=2147483647
#   define EC_CLZ0    ((int)sizeof(unsigned)*CHAR_BIT)
#   define EC_CLZ(_x) (__builtin_clz(_x))
#  elif LONG_MAX>=2147483647L
#   define EC_CLZ0    ((int)sizeof(unsigned long)*CHAR_BIT)
#   define EC_CLZ(_x) (__builtin_clzl(_x))
#  endif
# endif
#endif
#if defined(EC_CLZ)
/*Note that __builtin_clz is not defined when _x==0, according to the gcc
   documentation (and that of the BSR instruction that implements it on x86).
  The majority of the time we can never pass it zero.
  When we need to, it can be special cased.*/
# define EC_ILOG(_x) (EC_CLZ0-EC_CLZ(_x))
#else
int ec_ilog(opus_uint32 _v);
# define EC_ILOG(_x) (ec_ilog(_x))
#endif
#endif




static inline opus_int16 celt_ilog2(opus_int32 x)
{
   celt_assert2(x>0, "celt_ilog2() only defined for strictly positive numbers");
   return EC_ILOG(x)-1;
}
