use crate::{
    error::SpannedError,
    parser::lexer::{Code, Span},
};
use core::fmt;

pub fn display_spanned_error(err: &impl SpannedError, code: &Code) {
    eprintln!("ERROR: {:?}", err);
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
    let end_offset = code.0[span.end..].lines().next().map(str::len).unwrap_or(span.end);
    let line = &code.0[span.start - start_offset..span.end + end_offset];

    let linecount_in_span = code[span].lines().count();
    eprintln!(" {}", line.lines().intersperse("\\n").collect::<String>());
    let offset = " ".repeat(start_offset);
    eprintln!(" {offset}{} {label}", "^".repeat(span.len() + linecount_in_span - 1));
}

pub trait UnwrapDebug {
    type Inner;

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
