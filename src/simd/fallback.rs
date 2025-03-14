use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{BitAnd, BitOr, BitXor, Not},
};

struct BaseU8<T> {
    value: uint8x16_t,
    _phantom: PhantomData<T>,
}

impl<T> From<[u8; 16]> for BaseU8<T> {
    fn from(value: [u8; 16]) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

pub trait BaseU8Ops: Sized + BitOr + BitAnd + BitXor + Not + From<[u8; 16]> {}

trait WrapsBaseU8<T> {
    fn base(self) -> BaseU8<T>;
    fn from_base(base: BaseU8<T>) -> Self;
}

macro_rules! impl_wrapper {
    ($t: ty) => {
        impl BitOr for $t {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self::Output {
                Self::from_base(self.base().bitor(rhs.base()))
            }
        }
        impl BitAnd for $t {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self::Output {
                Self::from_base(self.base().bitand(rhs.base()))
            }
        }

        impl BitXor for $t {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self::from_base(self.base().bitxor(rhs.base()))
            }
        }
        impl Not for $t {
            type Output = Self;
            fn not(self) -> Self::Output {
                Self::from_base(self.base().not())
            }
        }
        impl From<uint8x16_t> for $t {
            fn from(value: uint8x16_t) -> Self {
                Self::from_base(value.into())
            }
        }

        impl BaseU8Ops for $t {}
    };
}

#[allow(non_camel_case_types)]
pub type uint8x16_t = [u8; 16];

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
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
}

pub struct Simd8<T> {
    base: BaseU8<T>,
}

impl<T> BaseU8<T> {}

macro_rules! array_assume_init {
    ($array:expr, $t: ty, $n:literal) => {
        std::mem::transmute::<[std::mem::MaybeUninit<$t>; $n], [$t; $n]>($array)
    };
}

impl<T> BitOr for BaseU8<T> {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = [MaybeUninit::uninit(); 16];
        for i in 0..16 {
            result[i].write(self.value[i] | rhs.value[i]);
        }
        let result = unsafe { array_assume_init!(result, u8, 16) };
        Self::from(result)
    }
}

impl<T> BitAnd for BaseU8<T> {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = [MaybeUninit::uninit(); 16];
        for i in 0..16 {
            result[i].write(self.value[i] & rhs.value[i]);
        }
        let result = unsafe { array_assume_init!(result, u8, 16) };
        Self::from(result)
    }
}

impl<T> BitXor for BaseU8<T> {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = [MaybeUninit::uninit(); 16];
        for i in 0..16 {
            result[i].write(self.value[i] ^ rhs.value[i]);
        }
        let result = unsafe { array_assume_init!(result, u8, 16) };
        Self::from(result)
    }
}

impl<T> Not for BaseU8<T> {
    type Output = Self;
    fn not(self) -> Self::Output {
        let mut result = [MaybeUninit::uninit(); 16];
        for i in 0..16 {
            result[i].write(!self.value[i]);
        }
        let result = unsafe { array_assume_init!(result, u8, 16) };
        Self::from(result)
    }
}
impl WrapsBaseU8<u8> for Simd8<u8> {
    fn base(self) -> BaseU8<u8> {
        self.base
    }

    fn from_base(base: BaseU8<u8>) -> Self {
        Self { base }
    }
}

impl_wrapper!(Simd8<u8>);

impl Simd8<u8> {
    pub unsafe fn load(values: *const u8) -> uint8x16_t {
        let mut result = [MaybeUninit::uninit(); 16];
        unsafe {
            std::ptr::copy_nonoverlapping(values, result.as_mut_ptr().cast(), 16);
        }
        let result = unsafe { array_assume_init!(result, u8, 16) };
        result
    }
}

impl Simd8<bool> {
    pub fn splat(value: bool) -> Self {
        Self {
            base: BaseU8::from([if value { 0xFF } else { 0x00 }; 16]),
        }
    }

    pub fn to_bitmask(&self) -> u32 {
        let bit_mask = make_uint8x16_t(
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
        )
        .into();
        let m_input = self.base & bit_mask;
        

    }
}

impl Simd8<u8> {
    pub fn splat(value: u8) -> Self {
        Self {
            base: BaseU8::from([value; 16]),
        }
    }
}
impl Simd8<u8> {
    pub fn store(&self, dst: &mut [u8; 16]) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.base.value.as_ptr(), dst.as_mut_ptr().cast(), 16);
        }
    }
}

impl From<u8> for Simd8<u8> {
    fn from(value: u8) -> Self {
        Self::splat(value)
    }
}

impl From<&'_ [u8; 16]> for Simd8<u8> {
    fn from(value: &'_ [u8; 16]) -> Self {
        unsafe { Self::load(value.as_ptr()) }.into()
    }
}
