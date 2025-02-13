use crate::{
    ast::{self},
    ptr::Ptr,
    scoped_stack::ScopedStack,
};

pub type SymbolTable<Symbol> = ScopedStack<(Ptr<str>, Symbol)>;

impl<Symbol> SymbolTable<Symbol> {
    #[inline]
    pub fn insert(&mut self, name: Ptr<str>, value: Symbol) {
        self.push((name, value))
    }

    #[inline]
    pub fn insert_no_duplicate(&mut self, name: Ptr<str>, value: Symbol) -> Result<(), ()> {
        if self.get_cur_scope().iter().any(|(n, _)| **n == *name) {
            Err(())
        } else {
            self.insert(name, value);
            Ok(())
        }
    }

    pub fn get(&self, name: &str) -> Option<&Symbol> {
        // `rev()` because of shadowing
        Some(&self.iter_scopes().flat_map(|s| s.iter().rev()).find(|(n, _)| **n == *name)?.1)
    }
}

/// contains [`ast::Decl`] and [`ast::Extern`]
pub type SymbolTable2 = ScopedStack<Ptr<ast::Decl>>;

impl SymbolTable2 {
    #[inline]
    pub fn insert_no_duplicate(&mut self, val: Ptr<ast::Decl>) -> Result<(), ()> {
        if self.get_cur_scope().iter().any(|d| *d.ident.text == *val.ident.text) {
            Err(())
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
