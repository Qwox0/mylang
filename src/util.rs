use crate::{
    error::SpannedError,
    parser::lexer::{Code, Span},
    type_::Type,
};
use core::fmt;
use std::{
    hint::unreachable_unchecked,
    io::{self, Read},
    ops::Try,
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

#[inline]
pub fn aligned_add(offset: usize, ty: &Type) -> usize {
    get_aligned_offset!(offset, ty.alignment()) + ty.size()
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

pub fn display_spanned_error(err: &impl SpannedError, code: &Code) {
    eprintln!("ERROR: {}", err.get_text());
    display_span_in_code(err.span(), code)
}

pub fn display_span_in_code(span: Span, code: &Code) {
    display_span_in_code_with_label(span, code, "")
}

pub fn display_span_in_code_with_label(span: Span, code: &Code, label: impl fmt::Display) {
    let start_offset = if code.0[..span.start].ends_with("\n") {
        0
    } else {
        code.0[..span.start].lines().last().map(str::len).unwrap_or(span.start)
    };
    if span == Span::pos(code.0.len()) {
        eprintln!("|");
        eprintln!("| {}", code.lines().last().unwrap_or(""));
        let offset = " ".repeat(start_offset);
        eprintln!("| {offset}{} {label}", "^".repeat(span.len()));
        eprintln!("|");
        return;
    }
    let end_offset = code.0[span.end..].lines().next().map(str::len).unwrap_or(0);
    let line = &code.0[span.start - start_offset..span.end + end_offset];

    let linecount_in_span = code[span].lines().count();
    eprintln!("|");
    eprintln!("| {}", line.lines().intersperse("\\n").collect::<String>());
    let offset = " ".repeat(start_offset);
    eprintln!("| {offset}{} {label}", "^".repeat(span.len() + linecount_in_span - 1));
    eprintln!("|");
}

pub fn debug_span_in_code(span: Span, code: &Code) {
    let line_num = code.0[..=span.start].lines().count();
    let linecount_in_span = code[span].lines().count();
    println!("{:?}", &code.0[..span.end]);
    println!(
        " {}{} {span:?}",
        " ".repeat(span.start + line_num.saturating_sub(1)),
        "^".repeat(span.len() + linecount_in_span.saturating_sub(1))
    );
}

pub trait UnwrapDebug {
    type Inner;

    /// like [`Option::unwrap`] but UB in release mode.
    #[track_caller]
    fn unwrap_debug(self) -> Self::Inner;
}

impl<T> UnwrapDebug for Option<T> {
    type Inner = T;

    fn unwrap_debug(self) -> Self::Inner {
        if cfg!(debug_assertions) {
            self.unwrap()
        } else {
            unsafe { self.unwrap_unchecked() }
        }
    }
}

impl<T, E: fmt::Debug> UnwrapDebug for Result<T, E> {
    type Inner = T;

    fn unwrap_debug(self) -> Self::Inner {
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

pub trait OkOrWithTry<T> {
    fn ok_or2<Result>(self, err: Result::Residual) -> Result
    where Result: Try<Output = T>;

    fn ok_or_else2<Result>(self, err: impl FnOnce() -> Result::Residual) -> Result
    where Result: Try<Output = T>;
}

impl<T> OkOrWithTry<T> for Option<T> {
    #[inline]
    fn ok_or2<Result>(self, err: Result::Residual) -> Result
    where Result: Try<Output = T> {
        match self {
            Some(t) => Result::from_output(t),
            None => Result::from_residual(err),
        }
    }

    #[inline]
    fn ok_or_else2<Result>(self, err: impl FnOnce() -> Result::Residual) -> Result
    where Result: Try<Output = T> {
        match self {
            Some(t) => Result::from_output(t),
            None => Result::from_residual(err()),
        }
    }
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
    unsafe { std::ptr::read(val as *const T as *const U) }
}

pub fn write_file_to_string(path: impl AsRef<Path>, buf: &mut String) -> io::Result<()> {
    std::fs::OpenOptions::new().read(true).open(path)?.read_to_string(buf)?;
    Ok(())
}
