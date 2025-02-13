use super::{SemaResult, value::SemaValue};
use crate::{ast::Ast, ptr::Ptr, util::OkOrWithTry};

/// Errors are considered a finished symbol of type [`Type::Never`].

impl SemaSymbol {
    pub fn err_symbol() -> SemaSymbol {
        SemaSymbol::Finished(SemaValue::never())
    }
}
