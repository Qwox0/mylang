use super::{value::SemaValue, SemaResult};
use crate::{type_::Type, util::OkOrWithTry};

/// Errors are considered a finished symbol of type [`Type::Never`].
#[derive(Debug)]
pub enum SemaSymbol {
    Finished(SemaValue),
    /// The [`Type`] of a symbol may be known even if the symbol wasn't fully
    /// analyzed yet.
    NotFinished(Option<Type>),
}

impl SemaSymbol {
    pub fn preload_symbol() -> SemaSymbol {
        SemaSymbol::NotFinished(None)
    }

    pub fn err_symbol() -> SemaSymbol {
        SemaSymbol::Finished(SemaValue::never())
    }

    pub fn get_type(&self) -> SemaResult<Type> {
        match self {
            SemaSymbol::Finished(val) => SemaResult::Ok(val.ty),
            SemaSymbol::NotFinished(ty) => ty.ok_or2(SemaResult::NotFinished),
        }
    }
}
