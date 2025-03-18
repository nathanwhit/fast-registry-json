use std::{
    arch::aarch64::{
        uint8x16_t, vandq_u8, vceqq_u8, vdupq_n_u8, veorq_u8, vgetq_lane_u16, vgetq_lane_u64,
        vld1q_u8, vmovq_n_u8, vmvnq_u8, vorrq_u8, vpaddq_u8, vqtbl1q_u8, vreinterpretq_u16_u8,
        vreinterpretq_u64_u8, vshlq_n_u8, vshrq_n_u8, vst1q_u8, vtstq_u8,
    },
    marker::PhantomData,
    ops::{BitAnd, BitOr, BitXor, Not},
};

use arrayref::{array_refs, mut_array_refs};

#[derive(Copy, Clone)]
struct BaseU8<T> {
    _phantom: PhantomData<T>,
    value: uint8x16_t,
}

impl<T> From<uint8x16_t> for BaseU8<T> {
    #[inline(always)]
    fn from(value: uint8x16_t) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

pub trait BaseU8Ops: Sized + BitOr + BitAnd + BitXor + Not + From<uint8x16_t> {}

trait WrapsBaseU8<T> {
    fn base(self) -> BaseU8<T>;
    fn from_base(base: BaseU8<T>) -> Self;
}

macro_rules! impl_wrapper {
    ($t: ty) => {
        impl BitOr for $t {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self::Output {
                Self::from_base(self.base().bitor(rhs.base()))
            }
        }
        impl BitAnd for $t {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self::Output {
                Self::from_base(self.base().bitand(rhs.base()))
            }
        }

        impl BitXor for $t {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self::from_base(self.base().bitxor(rhs.base()))
            }
        }
        impl Not for $t {
            type Output = Self;
            #[inline(always)]
            fn not(self) -> Self::Output {
                Self::from_base(self.base().not())
            }
        }
        impl From<uint8x16_t> for $t {
            #[inline(always)]
            fn from(value: uint8x16_t) -> Self {
                Self::from_base(value.into())
            }
        }

        impl BaseU8Ops for $t {}
    };
}

#[derive(Copy, Clone)]
pub struct Simd8<T> {
    base: BaseU8<T>,
}

impl<T> BaseU8<T> {}

impl<T> BitOr for BaseU8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { vorrq_u8(self.value, rhs.value) }.into()
    }
}

impl<T> BitAnd for BaseU8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { vandq_u8(self.value, rhs.value) }.into()
    }
}

impl<T> BitXor for BaseU8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { veorq_u8(self.value, rhs.value) }.into()
    }
}

impl<T> BitAnd<uint8x16_t> for BaseU8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: uint8x16_t) -> Self::Output {
        unsafe { vandq_u8(self.value, rhs) }.into()
    }
}

impl<T> BitOr<uint8x16_t> for BaseU8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: uint8x16_t) -> Self::Output {
        unsafe { vorrq_u8(self.value, rhs) }.into()
    }
}

impl<T> BitXor<uint8x16_t> for BaseU8<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: uint8x16_t) -> Self::Output {
        unsafe { veorq_u8(self.value, rhs) }.into()
    }
}

impl<T> From<u8> for BaseU8<T> {
    #[inline(always)]
    fn from(value: u8) -> Self {
        Self {
            value: unsafe { vmovq_n_u8(value) },
            _phantom: PhantomData,
        }
    }
}

impl<T> Not for BaseU8<T> {
    type Output = Self;

    fn not(self) -> Self::Output {
        unsafe { vmvnq_u8(self.value) }.into()
    }
}

impl WrapsBaseU8<u8> for Simd8<u8> {
    #[inline(always)]
    fn base(self) -> BaseU8<u8> {
        self.base
    }

    #[inline(always)]
    fn from_base(base: BaseU8<u8>) -> Self {
        Self { base }
    }
}

impl_wrapper!(Simd8<u8>);

impl Simd8<u8> {
    #[inline(always)]
    pub unsafe fn load(values: *const u8) -> uint8x16_t {
        unsafe { vld1q_u8(values) }
    }
    #[inline(always)]
    pub fn zero() -> uint8x16_t {
        unsafe { vdupq_n_u8(0) }
    }
    #[inline(always)]
    pub fn splat(value: u8) -> Self {
        unsafe { vmovq_n_u8(value) }.into()
    }

    #[inline(always)]
    pub fn store(&self, dst: &mut [u8; 16]) {
        unsafe {
            vst1q_u8(dst.as_mut_ptr(), self.base.value);
        }
    }
}

impl From<u8> for Simd8<u8> {
    #[inline(always)]
    fn from(value: u8) -> Self {
        Self::splat(value).into()
    }
}

impl From<&'_ [u8; 16]> for Simd8<u8> {
    #[inline(always)]
    fn from(value: &'_ [u8; 16]) -> Self {
        unsafe { Self::load(value.as_ptr()) }.into()
    }
}
impl From<[u8; 16]> for Simd8<u8> {
    #[inline(always)]
    fn from(value: [u8; 16]) -> Self {
        unsafe { Self::load(value.as_ptr()) }.into()
    }
}

impl Simd8<bool> {
    #[inline(always)]
    pub fn splat(value: bool) -> uint8x16_t {
        unsafe { vmovq_n_u8(if value { 0xFF } else { 0x00 }) }
    }

    #[inline(always)]
    pub fn to_bitmask(&self) -> u32 {
        let bit_mask = make_uint8x16_t(
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
        );
        let m_input = self.base & bit_mask;
        let mut tmp = unsafe { vpaddq_u8(m_input.value, m_input.value) };
        tmp = unsafe { vpaddq_u8(tmp, tmp) };
        tmp = unsafe { vpaddq_u8(tmp, tmp) };
        unsafe { vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0) as u32 }
    }
}

#[inline(always)]
pub fn make_uint8x16_t(
    a: u8,
    b: u8,
    c: u8,
    d: u8,
    e: u8,
    f: u8,
    g: u8,
    h: u8,
    i: u8,
    j: u8,
    k: u8,
    l: u8,
    m: u8,
    n: u8,
    o: u8,
    p: u8,
) -> uint8x16_t {
    let array = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p];
    unsafe { vld1q_u8(array.as_ptr()) }
}
// const SIZE: usize = ;
const NUM_CHUNKS: usize = 64 / size_of::<Simd8<u8>>();

pub struct Simd8x64<T> {
    pub chunks: [Simd8<T>; NUM_CHUNKS],
}

impl<T> Simd8x64<T> {
    #[inline(always)]
    pub fn from_chunks(chunks: [Simd8<T>; NUM_CHUNKS]) -> Self {
        Self { chunks }
    }
}

impl Simd8x64<u8> {
    #[inline(always)]
    pub fn store(&self, buf: &mut [u8; 64]) {
        let (a, b, c, d) = mut_array_refs![buf, 16, 16, 16, 16];
        self.chunks[0].store(a);
        self.chunks[1].store(b);
        self.chunks[2].store(c);
        self.chunks[3].store(d);
    }

    #[inline(always)]
    pub fn load(buf: &[u8; 64]) -> Self {
        let (a, b, c, d) = array_refs![buf, 16, 16, 16, 16];
        Self {
            chunks: [a.into(), b.into(), c.into(), d.into()],
        }
    }
}

pub trait Splat<T> {
    fn splat(value: T) -> Simd8<T>;
}

impl Splat<u8> for Simd8<u8> {
    #[inline(always)]
    fn splat(value: u8) -> Simd8<u8> {
        Simd8::<u8>::splat(value).into()
    }
}

impl Splat<bool> for Simd8<bool> {
    #[inline(always)]
    fn splat(value: bool) -> Simd8<bool> {
        Simd8::<bool>::splat(value).into()
    }
}

impl WrapsBaseU8<bool> for Simd8<bool> {
    #[inline(always)]
    fn base(self) -> BaseU8<bool> {
        self.base
    }

    #[inline(always)]
    fn from_base(base: BaseU8<bool>) -> Self {
        Self { base }
    }
}

impl From<uint8x16_t> for Simd8<bool> {
    #[inline(always)]
    fn from(value: uint8x16_t) -> Self {
        Self::from_base(value.into()).into()
    }
}

impl<T> Simd8<T> {
    #[inline(always)]
    pub fn eq_mask(&self, rhs: &Simd8<T>) -> Simd8<bool> {
        unsafe { vceqq_u8(self.base.value, rhs.base.value) }.into()
    }
}

impl<T> Simd8x64<T>
where
    Simd8<T>: Splat<T>,
{
    #[inline(always)]
    pub fn eq(&self, value: T) -> u64 {
        let mask = Simd8::<T>::splat(value);

        let a = mask.eq_mask(&self.chunks[0]);
        let b = mask.eq_mask(&self.chunks[1]);
        let c = mask.eq_mask(&self.chunks[2]);
        let d = mask.eq_mask(&self.chunks[3]);

        Simd8x64::<bool> {
            chunks: [a, b, c, d],
        }
        .to_bitmask()
    }

    #[inline(always)]
    pub fn to_bitmask(&self) -> u64 {
        let bit_mask = make_uint8x16_t(
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
        );
        unsafe {
            let sum0 = vpaddq_u8(
                vandq_u8(self.chunks[0].base.value, bit_mask),
                vandq_u8(self.chunks[1].base.value, bit_mask),
            );
            let sum1 = vpaddq_u8(
                vandq_u8(self.chunks[2].base.value, bit_mask),
                vandq_u8(self.chunks[3].base.value, bit_mask),
            );
            let sum0 = vpaddq_u8(sum0, sum1);
            let sum0 = vpaddq_u8(sum0, sum0);
            vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0)
        }
    }
}

impl Simd8<u8> {
    #[inline(always)]
    pub fn shr<const N: i32>(&self) -> Self {
        unsafe { vshrq_n_u8(self.base.value, N) }.into()
    }
    #[inline(always)]
    pub fn shl<const N: i32>(&self) -> Self {
        unsafe { vshlq_n_u8(self.base.value, N) }.into()
    }

    #[inline(always)]
    pub fn apply_lookup_16_to(&self, original: Simd8<u8>) -> Simd8<u8> {
        unsafe { vqtbl1q_u8(self.base.value, original.base.value) }.into()
    }

    #[inline(always)]
    pub fn lookup_16_table(&self, table: Simd8<u8>) -> Simd8<u8> {
        table.apply_lookup_16_to(*self)
    }

    #[inline(always)]
    pub fn lookup_16(
        &self,
        a: u8,
        b: u8,
        c: u8,
        d: u8,
        e: u8,
        f: u8,
        g: u8,
        h: u8,
        i: u8,
        j: u8,
        k: u8,
        l: u8,
        m: u8,
        n: u8,
        o: u8,
        p: u8,
    ) -> Simd8<u8> {
        let table = make_uint8x16_t(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p).into();

        self.lookup_16_table(table)
    }

    #[inline(always)]
    pub fn any_bits_set(&self, bits: Simd8<u8>) -> Simd8<bool> {
        unsafe { vtstq_u8(self.base.value, bits.base.value) }.into()
    }
}
