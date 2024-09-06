use super::LocalVariable;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct SymbolTable<'ctx>(Vec<HashMap<String, LocalVariable<'ctx>>>);

impl<'ctx> SymbolTable<'ctx> {
    pub fn open_scope(&mut self) {
        self.0.push(HashMap::new())
    }

    pub fn close_scope(&mut self) {
        self.0.pop();
    }

    fn get_last_mut(&mut self) -> &mut HashMap<String, LocalVariable<'ctx>> {
        self.0.last_mut().expect("symbol table has at least one scope")
    }

    /// see [`HashMap::reserve`]
    pub fn reserve(&mut self, additional: usize) {
        self.get_last_mut().reserve(additional)
    }

    pub fn insert(
        &mut self,
        name: String,
        value: LocalVariable<'ctx>,
    ) -> Option<LocalVariable<'ctx>> {
        self.get_last_mut().insert(name, value)
    }

    pub fn get(&self, name: &str) -> Option<&LocalVariable<'ctx>> {
        self.0.iter().rev().find_map(|scope| scope.get(name))
    }
}
