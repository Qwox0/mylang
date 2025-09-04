use inkwell::execution_engine::FunctionLookupError;

#[derive(Debug)]
#[allow(unused)]
pub enum JitError {
    FunctionLookupError(FunctionLookupError),
    MustAddAModuleFirst,
}

impl From<FunctionLookupError> for JitError {
    fn from(e: FunctionLookupError) -> Self {
        JitError::FunctionLookupError(e)
    }
}
