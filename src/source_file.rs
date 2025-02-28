use crate::{
    arena_allocator::Arena,
    ast::{self, Ast},
    parser::lexer::Code,
    ptr::{OPtr, Ptr},
    symbol_table::linear_search_symbol,
    util::UnwrapDebug,
};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct SourceFile {
    pub path: Ptr<Path>,
    pub code: Ptr<Code>,

    /// This is only [`None`] iff the `code` hasn't been parse yet.
    ///
    /// not owned.
    pub stmts: OPtr<[Ptr<Ast>]>,
    //pub root_scope: OPtr<ast::Block>,
}

impl SourceFile {
    /// The callee has to make sure that the provided combination of `path` and `code` makes sense.
    pub fn new(path: Ptr<Path>, code: Ptr<Code>) -> SourceFile {
        Self { path, code, stmts: None }
    }

    pub fn read(path: Ptr<Path>, alloc: &Arena) -> Result<SourceFile, std::io::Error> {
        let code = std::fs::read_to_string(path.as_ref())?;
        let code = Ptr::from_ref(alloc.0.alloc_str(&code).as_ref());
        Result::Ok(Self::new(path, code))
    }

    pub fn has_been_parsed(&self) -> bool {
        self.stmts.is_some()
    }

    pub fn find_symbol(self, name: &str) -> Option<Ptr<ast::Decl>> {
        linear_search_symbol(&self.stmts.u(), name)
    }
}
