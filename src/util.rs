use crate::{ast::DeclList, context::primitives, parser::lexer::Code};
use core::fmt;
use std::{
    hint::unreachable_unchecked,
    io::{self, Read},
    path::Path,
};

/// Calculates the padding bytes needed to align `offset` at `alignment`.
///
/// <https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding>
macro_rules! get_padding {
    ($offset:expr, $alignment:expr) => {
        $offset.wrapping_neg() & ($alignment - 1)
    };
}
pub(crate) use get_padding;

/// Adds the needed padding bytes to `offset` to align `offset` at `alignment`.
///
/// equivalent to `offset + get_padding(offset, alignment)`
///
/// <https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding>
macro_rules! get_aligned_offset {
    ($offset:expr, $alignment:expr) => {{
        let alignment = $alignment;
        ($offset + (alignment - 1)) & alignment.wrapping_neg()
    }};
}
pub(crate) use get_aligned_offset;

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
    get_aligned_offset!(offset, ty_layout.align) + ty_layout.size
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
    type Inner;

    /// like [`Option::unwrap`] but UB in release mode.
    #[track_caller]
    fn u(self) -> Self::Inner;
}

impl<T> UnwrapDebug for Option<T> {
    type Inner = T;

    fn u(self) -> Self::Inner {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
    }
}

impl<T, E: fmt::Debug> UnwrapDebug for Result<T, E> {
    type Inner = T;

    fn u(self) -> Self::Inner {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
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
#[track_caller]
#[inline]
pub fn panic_debug(msg: &str) -> ! {
    if cfg!(debug_assertions) {
        panic!("{}", msg)
    } else {
        unsafe { unreachable_unchecked() }
    }
}

pub fn collect_all_result_errors<T, E>(
    i: impl IntoIterator<Item = Result<T, E>>,
) -> Result<Vec<T>, Vec<E>> {
    let iter = i.into_iter();
    let mut res = Ok(Vec::with_capacity(iter.size_hint().0));
    for x in iter {
        match (x, &mut res) {
            (Ok(t), Ok(ok_list)) => ok_list.push(t),
            (Ok(_), Err(_)) => continue,
            (Err(err), Ok(_)) => res = Err(vec![err]),
            (Err(e), Err(err_list)) => err_list.push(e),
        }
    }
    res
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

pub fn transmute_unchecked<T, U>(val: &T) -> U {
    let ptr = val as *const T;
    debug_assert!(ptr.is_aligned());
    let ptr = ptr as *const U;
    debug_assert!(ptr.is_aligned());
    unsafe { std::ptr::read(ptr) }
}

pub fn any_as_bytes<T: Sized>(p: &T) -> &[u8; core::mem::size_of::<T>()] {
    //let slice = unsafe { core::slice::from_raw_parts((p as *const T) as *const u8, core::mem::size_of::<T>()) };
    unsafe { (p as *const T as *const [u8; core::mem::size_of::<T>()]).as_ref_unchecked() }
}

pub fn read_file_to_buf(path: impl AsRef<Path>, buf: &mut String) -> io::Result<()> {
    std::fs::OpenOptions::new().read(true).open(path)?.read_to_string(buf)?;
    Ok(())
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
