use crate::{
    codegen::llvm::CodegenError,
    parser::{lexer::Span, ParseError},
    sema::SemaError,
};
use std::fmt::Debug;

#[derive(Debug)]
pub enum Error {
    Parsing(ParseError),
    Sema(SemaError),
    Codegen(CodegenError),
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
        return format!("{:?} ({})", self.kind, self.context);
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
