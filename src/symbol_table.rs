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
