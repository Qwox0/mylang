use crate::{
    ast::DeclList, context::primitives, parser::lexer::Code, ptr::Ptr, scratch_allocator::TmpPtr,
};
use core::fmt;
use std::{
    fmt::Write,
    hash::{BuildHasher, Hash},
    hint::unreachable_unchecked,
    iter::FusedIterator,
    mem::{MaybeUninit, size_of, transmute},
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
use num::BigInt;
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

impl<T> UnwrapDebug for Ptr<[Option<T>]> {
    type Unwrapped = Ptr<[T]>;

    fn u(self) -> Self::Unwrapped {
        debug_assert!(self.iter().all(Option::is_some));
        const { assert!(size_of::<Option<T>>() == size_of::<T>()) };
        unsafe { std::mem::transmute::<Self, Self::Unwrapped>(self) }
    }
}

impl<T> UnwrapDebug for TmpPtr<[Option<T>]> {
    type Unwrapped = TmpPtr<[T]>;

    fn u(self) -> Self::Unwrapped {
        debug_assert!(self.iter().all(Option::is_some));
        const { assert!(size_of::<Option<T>>() == size_of::<T>()) };
        unsafe { std::mem::transmute::<Self, Self::Unwrapped>(self) }
    }
}

/// like [`unreachable`] but UB in release mode.
#[track_caller]
#[inline]
pub const fn unreachable_debug() -> ! {
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
    fn set_once(&mut self, val: T) -> &mut T
    where T: fmt::Debug;

    fn set_or_expect(&mut self, val: T)
    where T: PartialEq + fmt::Debug;

    fn display(&self) -> impl fmt::Display
    where T: fmt::Display;
}

impl<T> OptionExt<T> for Option<T> {
    #[inline]
    #[track_caller]
    fn set_once(&mut self, val: T) -> &mut T
    where T: fmt::Debug {
        debug_assert!(self.is_none(), "called set_once on {:?}", self);
        *self = Some(val);
        self.as_mut().u()
    }

    fn set_or_expect(&mut self, val: T)
    where T: PartialEq + fmt::Debug {
        debug_assert!(self.as_ref().is_none_or(|s| *s == val));
        *self = Some(val)
    }

    fn display(&self) -> impl std::fmt::Display
    where T: std::fmt::Display {
        struct DisplayOption<'a, T>(&'a Option<T>);
        impl<'a, T: std::fmt::Display> std::fmt::Display for DisplayOption<'a, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.0 {
                    Some(t) => write!(f, "{t}"),
                    None => write!(f, "None"),
                }
            }
        }
        DisplayOption(self)
    }
}

pub trait VecExt<T> {
    fn pop_expect_opt(&mut self, t: Option<T>) -> Option<T>
    where T: std::cmp::PartialEq + std::fmt::Debug;

    fn pop_expect(&mut self, t: T) -> Option<T>
    where T: std::cmp::PartialEq + std::fmt::Debug {
        self.pop_expect_opt(Some(t))
    }
}

impl<T> VecExt<T> for Vec<T> {
    fn pop_expect_opt(&mut self, t: Option<T>) -> Option<T>
    where T: std::cmp::PartialEq + std::fmt::Debug {
        let val = self.pop();
        debug_assert_eq!(val, t);
        val
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

pub trait IteratorExt: Iterator + Sized {
    #[allow(unused)]
    fn join(mut self, sep: impl AsRef<str>) -> String
    where
        Self: FusedIterator,
        Self::Item: std::fmt::Display,
    {
        let mut buf = String::new();
        self.join_into(sep, &mut buf);
        buf
    }

    fn join_into(mut self, sep: impl AsRef<str>, buf: &mut String)
    where
        Self: FusedIterator,
        Self::Item: std::fmt::Display,
    {
        use std::fmt::Write;
        let sep = sep.as_ref();
        let Some(first) = self.next() else { return };
        write!(buf, "{first}").u();
        for item in self {
            write!(buf, "{sep}{item}").u();
        }
    }

    /// Returns "{item1}, {item2}, ..., {itemN-1} {last_sep} {itemN}"
    ///                                          ^ no comma here (because I don't like it)
    fn join_fancy_list(mut self, last_sep: &str) -> String
    where
        Self: DoubleEndedIterator + FusedIterator,
        Self::Item: std::fmt::Display,
    {
        let mut buf =
            String::with_capacity((self.size_hint().0.saturating_sub(1)) * 2 + last_sep.len());

        let Some(last) = self.next_back() else { return buf };
        self.join_into(", ", &mut buf);
        write!(&mut buf, " {last_sep} {last}").u();
        buf
    }

    fn zip_exact<O>(self, other: impl IntoIterator<IntoIter = O>) -> std::iter::Zip<Self, O>
    where
        Self: ExactSizeIterator,
        O: ExactSizeIterator,
    {
        let other = other.into_iter();
        debug_assert_eq!(self.len(), other.len(), "Expected iterators to have same length");
        self.zip(other)
    }
}

impl<I: Iterator> IteratorExt for I {}

pub trait StrExt {
    fn capitalize(&mut self) -> &mut Self;
}

impl StrExt for str {
    fn capitalize(&mut self) -> &mut Self {
        if let Some(first_char) = self.get_mut(..1) {
            first_char.make_ascii_uppercase();
        }
        self
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

macro_rules! assert_has_field {
    ($ty:ty, $field:ident : $f_ty:ty) => {
        const {
            fn _f(x: $ty) -> $f_ty {
                x.$field
            }
        }
    };
}
pub(crate) use assert_has_field;

/// Like [`debug_assert`] but is only type checked in debug_mode
macro_rules! debug_only_assert {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        assert!($($arg)*)
    };
}
pub(crate) use debug_only_assert;

/// Like [`debug_assert_eq`] but is only type checked in debug_mode
macro_rules! debug_only_assert_eq {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        assert_eq!($($arg)*)
    };
}
pub(crate) use debug_only_assert_eq;

pub trait BigIntExt {
    type Inner;

    fn is_negative(&self) -> bool;

    fn is_zero(&self) -> bool;

    fn inner(&self) -> &Self::Inner;
}

impl BigIntExt for num::BigInt {
    type Inner = BigIntInner;

    fn is_negative(&self) -> bool {
        matches!(self.sign(), num::bigint::Sign::Minus)
    }

    fn is_zero(&self) -> bool {
        *self == num::BigInt::ZERO
    }

    fn inner(&self) -> &Self::Inner {
        let ptr = self as *const _;

        const { assert!(cfg!(target_pointer_width = "64")) }
        const { assert!(size_of::<num::BigInt>() == size_of::<BigIntInner>()) }
        unsafe { &*transmute::<*const num::BigInt, *const BigIntInner>(ptr) }
    }
}

impl BigIntExt for num::BigUint {
    type Inner = BigUintInner;

    fn is_negative(&self) -> bool {
        false
    }

    fn is_zero(&self) -> bool {
        *self == num::BigUint::ZERO
    }

    fn inner(&self) -> &Self::Inner {
        let ptr = self as *const _;

        const { assert!(cfg!(target_pointer_width = "64")) }
        const { assert!(size_of::<num::BigUint>() == size_of::<BigUintInner>()) }
        unsafe { &*transmute::<*const num::BigUint, *const BigUintInner>(ptr) }
    }
}

#[derive(Debug)]
pub struct BigIntInner {
    #[allow(unused)]
    pub sign: num::bigint::Sign,
    pub uint: num::BigUint,
}

#[derive(Debug)]
pub struct BigUintInner {
    pub data: Vec<u64>,
}

#[track_caller]
pub fn ui<'a, T>(big_int: &'a BigInt) -> T
where
    T: num::Unsigned + num::Integer + num::traits::WrappingNeg + TryFrom<&'a num::BigUint>,
    T::Error: fmt::Debug,
{
    let a = T::try_from(&big_int.inner().uint).expect("value too big");
    if big_int.is_negative() { a.wrapping_neg() } else { a }
}

pub fn to_f64(val: &BigInt) -> f64 {
    num::ToPrimitive::to_f64(val).expect("can't fail")
}

macro_rules! wrap_display {
    ($fmt:expr, $val:expr) => {{
        struct F<T>(T);

        impl<T: ::std::fmt::Display> ::std::fmt::Display for F<T> {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, $fmt, self.0)
            }
        }

        F($val)
    }};
}
pub(crate) use wrap_display;

pub const BITFLAGS_DEBUG_ALL: bool = false;

pub trait BitFlags: Copy + Eq + std::fmt::Debug {
    type Repr;

    fn get(&self, mask: Self::Repr) -> bool;
    fn set(&mut self, mask: Self::Repr);
    fn unset(&mut self, mask: Self::Repr);
}

macro_rules! bitflags {
    ($ty_name:ident : $repr:ty { $( $(#[$attr:meta])* $flag_name:ident),* $(,)? }) => {
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub struct $ty_name {
            pub data: $repr,
        }

        impl ::std::fmt::Debug for $ty_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use $crate::util::BitFlags;
                if $crate::util::BITFLAGS_DEBUG_ALL {
                    f.debug_struct(stringify!($ty_name))
                        $(.field(stringify!($flag_name), &self.get($ty_name::$flag_name)))*
                        .finish()
                } else {
                    let mut t = f.debug_tuple(stringify!(DeclFlags));
                    t.field_with(|f| write!(f, "{:01$b}", self.data, Self::_FLAG_COUNT));
                    $(if self.get($ty_name::$flag_name) { t.field(&stringify!($flag_name)); })*
                    t.finish_non_exhaustive()
                }
            }
        }

        impl $ty_name {
            $crate::util::bitflags! { _flags: $repr, 0, $($(#[$attr])* $flag_name,)* }

            pub const fn default() -> Self {
                Self { data: 0 }
            }
        }

        impl $crate::util::BitFlags for $ty_name {
            type Repr = $repr;

            fn get(&self, mask: Self::Repr) -> bool {
                self.data & mask != 0
            }

            fn set(&mut self, mask: Self::Repr) {
                self.data |= mask;
            }

            fn unset(&mut self, mask: Self::Repr) {
                self.data &= !mask;
            }
        }
    };
    (_flags: $repr:ty, $idx:expr,) => {
        pub const _FLAG_COUNT: usize = $idx;
    };
    (_flags: $repr:ty, $idx:expr, $(#[$attr:meta])* $flag_name:ident, $($rem:tt)*) => {
        $(#[$attr])*
        pub const $flag_name: $repr = 1 << $idx;
        $crate::util::bitflags! { _flags: $repr, $idx + 1, $($rem)* }
    };
}
pub(crate) use bitflags;

macro_rules! macro_orelse {
    (; $default:expr) => {
        $default
    };
    ($val:expr; $default:expr) => {
        $val
    };
}
pub(crate) use macro_orelse;

pub trait Extends<T> {
    fn base(self: Ptr<Self>) -> Ptr<T>;
}
impl<T> Extends<T> for T {
    #[inline]
    fn base(self: Ptr<Self>) -> Ptr<T> {
        self
    }
}
