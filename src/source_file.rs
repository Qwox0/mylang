use crate::{
    arena_allocator::Arena,
    ast,
    parser::lexer::{Code, Span},
    ptr::Ptr,
    symbol_table::linear_search_symbol,
    util::UnwrapDebug,
};
use std::{ops, path::Path, range::Range};

#[derive(Debug, Clone, Copy)]
pub struct SourceFile {
    pub path: Ptr<Path>,
    pub code: Ptr<Code>,

    /// This is only [`None`] iff the `code` hasn't been parse yet.
    ///
    /// not owned.
    pub stmt_range: Option<Range<usize>>,
    //pub root_scope: OPtr<ast::Block>,
}

impl SourceFile {
    /// The callee has to make sure that the provided combination of `path` and `code` makes sense.
    #[inline]
    pub fn new(path: Ptr<Path>, code: Ptr<Code>) -> SourceFile {
        Self { path, code, stmt_range: None }
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

    pub fn find_symbol(
        self,
        name: &str,
        all_stmts: Ptr<[Ptr<ast::Ast>]>,
    ) -> Option<Ptr<ast::Decl>> {
        linear_search_symbol(&all_stmts[self.stmt_range.u()], name)
    }
}

impl Ptr<SourceFile> {
    pub fn full_span(self) -> Span {
        Span::new(0..self.code.len(), Some(self))
    }
}
