use crate::{
    parser::{ParseError, lexer::Span},
    sema::{SemaError, SemaErrorKind},
    util::unreachable_debug,
};
use std::fmt::Debug;

pub trait SpannedError: Debug {
    fn span(&self) -> Span;

    fn get_text(&self) -> String;

    fn was_already_handled(&self) -> bool;
}

impl SpannedError for ParseError {
    fn span(&self) -> Span {
        unreachable_debug()
    }

    fn get_text(&self) -> String {
        unreachable_debug()
    }

    fn was_already_handled(&self) -> bool {
        true
    }
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
