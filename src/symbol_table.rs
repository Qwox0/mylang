use crate::{ptr::Ptr, scoped_stack::ScopedStack};

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
