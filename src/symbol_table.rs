use std::collections::HashMap;

#[derive(Debug)]
pub struct SymbolTable<Symbol>(Vec<HashMap<String, Symbol>>);

impl<Symbol> SymbolTable<Symbol> {
    pub fn with_one_scope() -> SymbolTable<Symbol> {
        SymbolTable(vec![HashMap::new()])
    }

    pub fn inner(&self) -> &[HashMap<String, Symbol>] {
        &self.0
    }

    pub fn open_scope(&mut self) {
        self.0.push(HashMap::new())
    }

    pub fn close_scope(&mut self) {
        self.0.pop();
    }

    fn get_last_mut(&mut self) -> &mut HashMap<String, Symbol> {
        self.0.last_mut().expect("symbol table has at least one scope")
    }

    /// see [`HashMap::reserve`]
    pub fn reserve(&mut self, additional: usize) {
        self.get_last_mut().reserve(additional)
    }

    pub fn insert(&mut self, name: impl ToString, value: Symbol) -> Option<Symbol> {
        self.get_last_mut().insert(name.to_string(), value)
    }

    pub fn get(&self, name: &str) -> Option<&Symbol> {
        self.0.iter().rev().find_map(|scope| scope.get(name))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Symbol> {
        self.0.iter().flat_map(|m| m.iter().map(|(_, v)| v))
    }
}
