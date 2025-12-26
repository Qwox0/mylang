use crate::{
    ast::DeclList,
    context::primitives,
    parser::lexer::Code,
    ptr::{OPtr, Ptr},
};
use core::fmt;
use std::{
    hash::{BuildHasher, Hash},
    hint::unreachable_unchecked,
    iter::FusedIterator,
    mem::MaybeUninit,
    path::Path,
};

/// Adds the needed padding bytes to `offset` to align `offset` at `alignment`.
///
/// equivalent to `offset + get_padding(offset, alignment)`
///
/// <https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding>
macro_rules! round_up_to_alignment {
    ($offset:expr, $alignment:expr) => {{
        let alignment = $alignment;
        ($offset + (alignment - 1)) & alignment.wrapping_neg()
    }};
}
pub(crate) use round_up_to_alignment;

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub size: usize,
    pub align: usize,
}

impl Layout {
    #[inline]
    pub fn new(size: usize, align: usize) -> Layout {
        Layout { size, align }
    }
}

#[inline]
pub fn aligned_add(offset: usize, ty_layout: Layout) -> usize {
    round_up_to_alignment!(offset, ty_layout.align) + ty_layout.size
}

/// <https://jameshfisher.com/2018/03/30/round-up-power-2/>
pub fn round_up_to_nearest_power_of_two(x: usize) -> usize {
    1usize.wrapping_shl(usize::BITS - x.wrapping_sub(1).leading_zeros())
}

#[test]
pub fn test_round_up_to_nearest_power_of_two() {
    assert_eq!(round_up_to_nearest_power_of_two(0), 1);
    assert_eq!(round_up_to_nearest_power_of_two(1), 1);
    assert_eq!(round_up_to_nearest_power_of_two(2), 2);
    assert_eq!(round_up_to_nearest_power_of_two(3), 4);
    assert_eq!(round_up_to_nearest_power_of_two(4), 4);
    assert_eq!(round_up_to_nearest_power_of_two(5), 8);
    assert_eq!(round_up_to_nearest_power_of_two(8), 8);
    assert_eq!(round_up_to_nearest_power_of_two(9), 16);
}

#[allow(unused)]
pub struct FileLoc {
    /// 1-indexed
    pub line: usize,
    /// 1-indexed, char count
    pub col: usize,
}

pub fn resolve_file_loc(byte_pos: usize, code: &Code) -> FileLoc {
    //assert_ne!(code.0.as_bytes()[byte_pos], b'\n');
    let mut line = 1;
    let mut last_line_break_pos = 0;
    for (idx, b) in code.0[..byte_pos].as_bytes().iter().copied().enumerate() {
        if b == b'\n' {
            line += 1;
            last_line_break_pos = idx;
        }
    }
    let col = code.0[last_line_break_pos..byte_pos].chars().count() + 1;
    FileLoc { line, col }
}

pub trait UnwrapDebug {
    type Unwrapped;

    /// like [`Option::unwrap`] but UB in release mode.
    #[track_caller]
    fn u(self) -> Self::Unwrapped;
}

impl<T> UnwrapDebug for Option<T> {
    type Unwrapped = T;

    #[inline]
    fn u(self) -> Self::Unwrapped {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
    }
}

impl<T, E: fmt::Debug> UnwrapDebug for Result<T, E> {
    type Unwrapped = T;

    #[inline]
    fn u(self) -> Self::Unwrapped {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
    }
}

impl<T> UnwrapDebug for Ptr<[OPtr<T>]> {
    type Unwrapped = Ptr<[Ptr<T>]>;

    fn u(self) -> Self::Unwrapped {
        debug_assert!(self.iter().all(Option::is_some));
        unsafe { std::mem::transmute::<Self, Self::Unwrapped>(self) }
    }
}

/// like [`unreachable`] but UB in release mode.
#[track_caller]
#[inline]
pub fn unreachable_debug() -> ! {
    if cfg!(debug_assertions) {
        unreachable!()
    } else {
        unsafe { unreachable_unchecked() }
    }
}

/// like [`panic`] but UB in release mode.
macro_rules! panic_debug {
    ($($msg_fmt:expr),* $(,)?) => {
        if cfg!(debug_assertions) {
            panic!($($msg_fmt),*)
        } else {
            unsafe { ::std::hint::unreachable_unchecked() }
        }
    };
}
pub(crate) use panic_debug;

pub trait OptionExt<T> {
    fn set_once(&mut self, val: T) -> &mut T;
}

impl<T> OptionExt<T> for Option<T> {
    #[inline]
    fn set_once(&mut self, val: T) -> &mut T {
        debug_assert!(self.is_none());
        *self = Some(val);
        self.as_mut().u()
    }
}

#[inline]
pub unsafe fn forget_lifetime<'a, T: ?Sized>(r: &T) -> &'a T {
    unsafe { &*(r as *const T) }
}

#[inline]
pub unsafe fn forget_lifetime_mut<'a, T: ?Sized>(r: &mut T) -> &'a mut T {
    unsafe { &mut *(r as *mut T) }
}

pub fn variant_count_to_tag_size_bits(variant_count: usize) -> u32 {
    if variant_count <= 1 { 0 } else { (variant_count - 1).ilog2() + 1 }
}

pub fn variant_count_to_tag_size_bytes(variant_count: usize) -> u32 {
    variant_count_to_tag_size_bits(variant_count).div_ceil(8)
}

#[test]
fn test_variant_count_to_tag_size_bits() {
    assert_eq!(variant_count_to_tag_size_bits(0), 0);
    assert_eq!(variant_count_to_tag_size_bits(1), 0);
    assert_eq!(variant_count_to_tag_size_bits(2), 1);
    assert_eq!(variant_count_to_tag_size_bits(3), 2);
    assert_eq!(variant_count_to_tag_size_bits(4), 2);
    assert_eq!(variant_count_to_tag_size_bits(5), 3);
    assert_eq!(variant_count_to_tag_size_bits(8), 3);
    assert_eq!(variant_count_to_tag_size_bits(9), 4);
    assert_eq!(variant_count_to_tag_size_bits(256), 8);
    assert_eq!(variant_count_to_tag_size_bits(257), 9);
}

pub fn transmute_unchecked<T, U>(val: T) -> U {
    let ptr = &val as *const T;
    debug_assert!(ptr.is_aligned());
    let ptr = ptr as *const U;
    debug_assert!(ptr.is_aligned());
    let u = unsafe { std::ptr::read(ptr) };
    std::mem::forget(val);
    u
}

pub fn is_simple_enum(variants: DeclList) -> bool {
    variants.iter().all(|v| v.var_ty == primitives().void_ty)
}

/// better [`bool::then`]
macro_rules! then {
    ($b:expr => $some:expr) => {
        if $b { Some($some) } else { None }
    };
}
pub(crate) use then;

#[derive(Debug)]
pub enum IteratorOneError {
    NoItems,
    TooManyItems,
}

pub trait IteratorExt: Iterator + Sized {
    fn one(self) -> Result<Self::Item, IteratorOneError>
    where Self: FusedIterator;

    /// Expects the iterator to have exactly one item.
    fn expect_one(self) -> Self::Item
    where Self: FusedIterator {
        self.one().unwrap_or_else(|e| {
            let count = match e {
                IteratorOneError::NoItems => "0",
                IteratorOneError::TooManyItems => "multiple",
            };
            panic!("Expected the iterator to have exactly one item but got {count} items instead");
        })
    }

    fn join(self, sep: impl AsRef<str>) -> String
    where Self::Item: std::fmt::Display;
}

impl<I: FusedIterator> IteratorExt for I {
    fn one(mut self) -> Result<Self::Item, IteratorOneError>
    where Self: FusedIterator {
        let Some(item) = self.next() else { return Err(IteratorOneError::NoItems) };
        if self.next().is_some() {
            return Err(IteratorOneError::TooManyItems);
        }
        Ok(item)
    }

    fn join(mut self, sep: impl AsRef<str>) -> String
    where Self::Item: std::fmt::Display {
        let sep = sep.as_ref();
        let acc = self.next().map(|i| i.to_string()).unwrap_or_else(String::new);
        self.fold(acc, |mut acc, item| {
            acc.push_str(sep);
            use std::fmt::Write;
            write!(acc, "{item}").u();
            acc
        })
    }
}

pub fn is_canonical(path: &Path) -> bool {
    path.canonicalize().is_ok_and(|p| p.as_path() == path)
}

pub const fn concat_arr_impl<T: Copy, const A: usize, const B: usize, const C: usize>(
    a: [T; A],
    b: [T; B],
) -> [T; C] {
    const { assert!(A + B == C) };

    let mut result = [const { MaybeUninit::uninit() }; C];

    let mut i = 0;
    while i < A {
        result[i].write(a[i]);
        i += 1;
    }

    while i < A + B {
        result[i].write(b[i - A]);
        i += 1;
    }

    unsafe { MaybeUninit::array_assume_init(result) }
}

macro_rules! concat_arr {
    ($arr1:expr, $arr2:expr $(,)?) => {
        $crate::util::concat_arr_impl::<_, _, _, { $arr1.len() + $arr2.len() }>($arr1, $arr2)
    };
}
pub(crate) use concat_arr;

pub fn hash_val(h: &impl BuildHasher, val: impl Hash) -> u64 {
    h.hash_one(val)
}
