use crate::{ast, ptr::Ptr, scoped_stack::ScopedStack};

pub type CodegenSymbolTable<Symbol> = ScopedStack<(Ptr<ast::Decl>, Symbol)>;

impl<Symbol> CodegenSymbolTable<Symbol> {
    #[inline]
    pub fn insert(&mut self, name: Ptr<ast::Decl>, value: Symbol) {
        self.push((name, value))
    }

    #[inline]
    pub fn insert_no_duplicate(&mut self, name: Ptr<ast::Decl>, value: Symbol) -> Result<(), ()> {
        if self.get_cur_scope().iter().any(|(n, _)| *n == name) {
            Err(())
        } else {
            self.insert(name, value);
            Ok(())
        }
    }

    pub fn get(&self, name: Ptr<ast::Decl>) -> Option<&Symbol> {
        // `rev()` because of shadowing
        Some(&self.iter_scopes().flat_map(|s| s.iter().rev()).find(|(n, _)| *n == name)?.1)
    }
}

/// contains [`ast::Decl`] and [`ast::Extern`]
pub type SemaSymbolTable = ScopedStack<Ptr<ast::Decl>>;

impl SemaSymbolTable {
    #[inline]
    pub fn insert_no_duplicate(&mut self, val: Ptr<ast::Decl>) -> Result<(), Ptr<ast::Decl>> {
        if let Some(dup) = self.get_cur_scope().iter().find(|d| *d.ident.text == *val.ident.text) {
            Err(*dup)
        } else {
            self.push(val);
            Ok(())
        }
    }

    pub fn get(&self, name: &str) -> Option<Ptr<ast::Decl>> {
        self.iter_scopes()
            .flat_map(|s| s.iter().rev()) // `rev()` because of shadowing
            .copied()
            .find(|d| *d.ident.text == *name)
    }
}

impl Ptr<ast::Block> {
    pub fn find_symbol(self, name: &str) -> Option<Ptr<ast::Decl>> {
        linear_search_symbol(&self.stmts, name)
    }
}

#[inline]
pub fn linear_search_symbol(stmts: &[Ptr<ast::Ast>], name: &str) -> Option<Ptr<ast::Decl>> {
    stmts
        .iter()
        .copied()
        .rev() // `rev()` because of shadowing
        .filter_map(|a| a.try_downcast::<ast::Decl>())
        .filter(|d| d.on_type.is_none())
        .find(|d| *d.ident.text == *name)
}
