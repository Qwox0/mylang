use crate::{
    arena_allocator::Arena,
    ast::Scope,
    parser::lexer::{Code, Span},
    ptr::Ptr,
};
use std::{ops, path::Path, range::Range};

#[derive(Debug)]
pub struct SourceFile {
    pub path: Ptr<Path>,
    pub code: Ptr<Code>,

    /// This is only [`None`] iff the `code` hasn't been parse yet.
    pub stmt_range: Option<Range<usize>>,
    /// not available during parsing
    pub scope: Option<Scope>,
}

impl SourceFile {
    /// The callee has to make sure that the provided combination of `path` and `code` makes sense.
    #[inline]
    pub fn new(path: Ptr<Path>, code: Ptr<Code>) -> SourceFile {
        Self { path, code, stmt_range: None, scope: None }
    }

    #[inline]
    pub fn set_stmt_range(&mut self, stmt_range: ops::Range<usize>) {
        self.stmt_range = Some(stmt_range.into())
    }

    pub fn read(path: Ptr<Path>, alloc: &Arena) -> std::io::Result<SourceFile> {
        let code = std::fs::read_to_string(path.as_ref())?;
        let code = Ptr::from_ref(alloc.0.alloc_str(&code).as_ref());
        Result::Ok(Self::new(path, code))
    }

    pub fn has_been_parsed(&self) -> bool {
        self.stmt_range.is_some()
    }
}

impl Ptr<SourceFile> {
    pub fn full_span(self) -> Span {
        Span::new(0..self.code.len(), Some(self))
    }
}
