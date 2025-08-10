use crate::{
    context::ctx,
    ptr::Ptr,
    util::{UnwrapDebug, hash_val},
};
use core::fmt;
use hashbrown::{DefaultHashBuilder, HashMap, hash_map::RawEntryMut};
use std::hash::Hash;

/// An interned Symbol
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol {
    idx: u32,
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Symbol#{}({:?})", self.idx, self.text())
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text())
    }
}

impl Symbol {
    fn new(idx: usize) -> Symbol {
        Symbol { idx: u32::try_from(idx).u() }
    }

    /// Loads the text of the symbol from the [`crate::context::CompilationContext`].
    pub fn text(&self) -> &'static str {
        ctx().symbols.sym_text(*self).as_ref()
    }

    fn as_idx(&self) -> usize {
        self.idx as usize
    }
}

pub struct InternPool {
    strings: Vec<Ptr<str>>,
    dedup: HashMap<Symbol, (), ()>,
    hash_builder: DefaultHashBuilder,
}

impl InternPool {
    pub fn new() -> Self {
        const START_CAP: usize = 512;
        Self {
            strings: Vec::with_capacity(START_CAP),
            dedup: HashMap::with_capacity_and_hasher(START_CAP, ()),
            hash_builder: DefaultHashBuilder::default(),
        }
    }

    pub fn get_or_intern(&mut self, val: Ptr<str>) -> Symbol {
        let hash = hash_val(&self.hash_builder, val.as_ref());
        *match self
            .dedup
            .raw_entry_mut()
            .from_hash(hash, |sym| val.as_ref() == self.strings.get(sym.as_idx()).u().as_ref())
        {
            RawEntryMut::Occupied(val) => val.into_key_value(),
            RawEntryMut::Vacant(slot) => {
                let sym = Symbol::new(self.strings.len());
                self.strings.push(val);
                slot.insert_with_hasher(hash, sym, (), |sym| {
                    hash_val(&self.hash_builder, self.strings.get(sym.as_idx()).u().as_ref())
                })
            },
        }
        .0
    }

    #[inline]
    pub fn sym_text(&self, sym: Symbol) -> Ptr<str> {
        self.strings.get(sym.as_idx()).copied().u()
    }
}
