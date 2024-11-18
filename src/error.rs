use crate::{
    codegen::llvm::CodegenError,
    parser::{ParseError, lexer::Span},
    sema::SemaError,
    util::panic_debug,
};
use std::fmt::Debug;

#[derive(Debug)]
pub enum Error {
    Parsing(ParseError),
    Sema(SemaError),
    Codegen(CodegenError),
}

macro_rules! impl_convert_method {
    ($fn_name:ident, $variant:ident -> $err_ty:ty) => {
        pub fn $fn_name(self) -> $err_ty {
            match self {
                Error::$variant(err) => err,
                _ => panic_debug("called"),
            }
        }
    };
}

impl Error {
    impl_convert_method! { parse, Parsing -> ParseError }

    impl_convert_method! { sema, Sema -> SemaError }

    impl_convert_method! { codegen, Codegen -> CodegenError }
}

macro_rules! impl_from {
    ($variant:ident($err_ty:ident)) => {
        impl From<$err_ty> for Error {
            fn from(err: $err_ty) -> Self {
                Error::$variant(err)
            }
        }
    };
}

impl_from! { Parsing(ParseError) }
impl_from! { Sema(SemaError) }
impl_from! { Codegen(CodegenError) }

pub trait SpannedError: Debug {
    fn span(&self) -> Span;

    fn get_text(&self) -> String;
}

impl SpannedError for ParseError {
    fn span(&self) -> Span {
        self.span
    }

    fn get_text(&self) -> String {
        #[cfg(debug_assertions)]
        return format!("{:?} ({:#})", self.kind, self.context);
        #[cfg(not(debug_assertions))]
        return format!("{:?}", self.kind);
    }
}

impl SpannedError for SemaError {
    fn span(&self) -> Span {
        self.span
    }

    fn get_text(&self) -> String {
        format!("{:?}", self.kind)
    }
}

impl SpannedError for Error {
    fn span(&self) -> Span {
        match self {
            Error::Parsing(e) => e.span(),
            Error::Sema(e) => e.span(),
            Error::Codegen(_) => todo!(),
        }
    }

    fn get_text(&self) -> String {
        match self {
            Error::Parsing(e) => e.get_text(),
            Error::Sema(e) => e.get_text(),
            Error::Codegen(_) => todo!(),
        }
    }
}
