use super::Symbol;
use std::collections::HashMap;

#[derive(Debug)]
pub struct SymbolTable<'ctx>(Vec<HashMap<String, Symbol<'ctx>>>);

impl<'ctx> SymbolTable<'ctx> {
    pub fn with_one_scope() -> SymbolTable<'ctx> {
        SymbolTable(vec![HashMap::new()])
    }

    pub fn inner(&self) -> &[HashMap<String, Symbol<'ctx>>] {
        &self.0
    }

    pub fn open_scope(&mut self) {
        self.0.push(HashMap::new())
    }

    pub fn close_scope(&mut self) {
        self.0.pop();
    }

    fn get_last_mut(&mut self) -> &mut HashMap<String, Symbol<'ctx>> {
        self.0.last_mut().expect("symbol table has at least one scope")
    }

    /// see [`HashMap::reserve`]
    pub fn reserve(&mut self, additional: usize) {
        self.get_last_mut().reserve(additional)
    }

    pub fn insert(&mut self, name: String, value: Symbol<'ctx>) -> Option<Symbol<'ctx>> {
        self.get_last_mut().insert(name, value)
    }

    pub fn get(&self, name: &str) -> Option<&Symbol<'ctx>> {
        self.0.iter().rev().find_map(|scope| scope.get(name))
    }
}
