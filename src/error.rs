use crate::{
    parser::lexer::Span,
    sema::{SemaError, SemaErrorKind},
};
use std::fmt::Debug;

pub trait SpannedError: Debug {
    fn span(&self) -> Span;

    fn get_text(&self) -> String;

    fn was_already_handled(&self) -> bool;
}

impl SpannedError for SemaError {
    fn span(&self) -> Span {
        self.span
    }

    fn get_text(&self) -> String {
        format!("{}", self.kind)
    }

    fn was_already_handled(&self) -> bool {
        matches!(self.kind, SemaErrorKind::HandledErr)
    }
}
