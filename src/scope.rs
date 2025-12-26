use crate::{
    arena_allocator::{AllocErr, Arena},
    ast::{self, Decl, DeclList},
    context::ctx,
    diagnostics::{DiagnosticReporter, cerror, chint},
    intern_pool::Symbol,
    ptr::{OPtr, Ptr},
    util::{hash_val, unreachable_debug},
};
use hashbrown::{DefaultHashBuilder, HashMap, hash_map::RawEntryMut};

/// The number of [`Decl`]s before this statement in the current [`Scope`].
///
/// ```text
/// stmt pos=0
/// decl pos=0
/// stmt pos=1
/// stmt pos=1
/// decl pos=1
/// stmt pos=2
/// ```
///
/// A [`Decl`] has the same pos as the statement before it because ...
///
/// ```mylang
/// a := a + 1; // the `a` in the init expr shouldn't resolve to this decl.
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScopePos(pub u32);

impl ScopePos {
    pub const UNSET: Self = ScopePos(u32::MAX);

    #[inline]
    pub fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Root,
    File,
    Block,
    ForLoop,
    Fn,
    Struct,
    Union,
    Enum,
}

impl ScopeKind {
    pub fn allows_shadowing(self) -> bool {
        matches!(self, ScopeKind::Block)
    }

    pub fn is_aggregate(self) -> bool {
        matches!(self, ScopeKind::Struct | ScopeKind::Union | ScopeKind::Enum)
    }
}

#[derive(Debug)]
pub struct Scope {
    pub parent: OPtr<Scope>,
    pub pos_in_parent: ScopePos,
    pub decls: DeclList,
    /// used for symbol lookups when this scope has more than [`SMALL_SCOPE_MAX_SIZE`] Decls.
    /// Set at the start of sema, if needed
    ///
    /// Currently only used for unordered scopes because those scopes don't allow shadowing.
    decls_map: Option<UnorderedDeclMap>,
    pub kind: ScopeKind,

    #[cfg(debug_assertions)]
    pub was_checked_for_duplicates: bool,
}

const SMALL_SCOPE_MAX_SIZE: usize = 32;

impl Scope {
    pub fn new(decls: DeclList, kind: ScopeKind) -> Scope {
        debug_assert!(decls.iter().all(|d| d.on_type.is_none()));
        Scope {
            parent: None,
            pos_in_parent: ScopePos::UNSET,
            decls,
            decls_map: None,
            kind,

            #[cfg(debug_assertions)]
            was_checked_for_duplicates: false,
        }
    }

    pub fn from_stmts(
        stmts: &[Ptr<ast::Ast>],
        kind: ScopeKind,
        alloc: &Arena,
    ) -> Result<Scope, AllocErr> {
        // TODO: bench copy vs preallocate `stmts.len`
        let decls = stmts
            .iter()
            .filter_map(|s| s.try_downcast::<Decl>())
            .filter(|d| d.on_type.is_none())
            .collect::<Vec<_>>();
        Ok(Scope::new(alloc.alloc_slice(&decls)?, kind))
    }

    pub fn file(stmts: &[Ptr<ast::Ast>], parent_scope: Ptr<Scope>, alloc: &Arena) -> Scope {
        let mut scope = Scope::from_stmts(stmts, ScopeKind::File, alloc).unwrap();
        scope.parent = Some(parent_scope);
        debug_assert!(!parent_scope.kind.allows_shadowing());
        scope.pos_in_parent = ScopePos(parent_scope.decls.len() as u32);
        scope
    }

    /// also returns the fields as a [`DeclList`].
    pub fn for_aggregate(
        fields: Vec<Ptr<Decl>>,
        consts: Vec<Ptr<Decl>>,
        alloc: &Arena,
        kind: ScopeKind,
    ) -> Result<ScopeAndAggregateInfo, AllocErr> {
        debug_assert!(kind.is_aggregate());
        let fields = alloc.alloc_slice(&fields)?; // fields are allocated twice because `scope_decls` is rearranged during sema.
        let scope_decls = alloc.alloc_uninit_slice(fields.len() + consts.len())?;
        scope_decls.as_mut()[..fields.len()].write_copy_of_slice(&fields);
        scope_decls.as_mut()[fields.len()..].write_copy_of_slice(&consts);
        let scope = Scope::new(scope_decls.assume_init(), kind);
        Ok(ScopeAndAggregateInfo { scope, fields, consts })
    }

    /// also sets up [`Scope::decls_map`] if needed.
    pub fn verify_no_duplicates(&mut self) {
        if !self.kind.allows_shadowing() {
            if let Some(map) = verify_no_duplicates(self.kind, self.decls) {
                self.decls_map = Some(map);
            }
        }
        #[cfg(debug_assertions)]
        {
            self.was_checked_for_duplicates = true;
        }
    }

    pub fn find_decl_norec(
        &self,
        sym: Symbol,
        cur_pos: ScopePos,
        ignore_fields: bool,
    ) -> OPtr<Decl> {
        #[cfg(debug_assertions)]
        debug_assert!(self.was_checked_for_duplicates);
        debug_assert_eq!(self.decls_map.is_some(), self.decls.len() > SMALL_SCOPE_MAX_SIZE);
        let ignore_non_const = ignore_fields && self.kind.is_aggregate();
        if let Some(decls_map) = self.decls_map.as_ref() {
            debug_assert!(ctx().do_abort_compilation() || decls_map.len() == self.decls.len());
            decls_map.get(sym, ignore_non_const)
        } else {
            let decls = if self.kind.allows_shadowing() {
                &self.decls[..cur_pos.0 as usize]
            } else {
                &self.decls
            };
            linear_search_symbol(decls, sym, self.kind.allows_shadowing(), ignore_non_const)
        }
    }

    pub fn find_decl(&self, sym: Symbol, cur_pos: ScopePos, ignore_fields: bool) -> OPtr<Decl> {
        let mut cur_scope = Some(Ptr::from_ref(self));
        let mut cur_pos = cur_pos;
        while let Some(scope) = cur_scope {
            if let Some(sym) = scope.find_decl_norec(sym, cur_pos, ignore_fields) {
                return Some(sym);
            }
            cur_scope = scope.parent;
            cur_pos = scope.pos_in_parent;
        }
        return None;
    }

    /// assumes `self` is an unordered scope. see [`ScopeKind::allows_shadowing`].
    pub fn find_decl_unordered(&self, sym: Symbol, ignore_fields: bool) -> OPtr<Decl> {
        debug_assert!(!self.kind.allows_shadowing());
        self.find_decl(sym, ScopePos::UNSET, ignore_fields)
    }

    /// assumes `self` is an unordered scope. see [`ScopeKind::allows_shadowing`].
    pub fn find_decl_norec_unordered(&self, sym: Symbol, ignore_fields: bool) -> OPtr<Decl> {
        debug_assert!(!self.kind.allows_shadowing());
        self.find_decl_norec(sym, ScopePos::UNSET, ignore_fields)
    }
}

/// `reverse` is needed because of shadowing
fn linear_search_symbol(
    decls: &[Ptr<Decl>],
    sym: Symbol,
    reverse: bool,
    ignore_non_const: bool,
) -> Option<Ptr<Decl>> {
    debug_assert!(decls.iter().all(|d| d.on_type.is_none()));
    let mut d = decls.iter().copied().filter(|d| !ignore_non_const || d.is_const);
    if reverse {
        d.rfind(|d| d.ident.sym == sym)
    } else {
        d.find(|d| d.ident.sym == sym)
    }
}

pub struct ScopeAndAggregateInfo {
    pub scope: Scope,
    pub fields: DeclList,
    pub consts: Vec<Ptr<Decl>>,
}

#[derive(Debug)]
pub struct UnorderedDeclMap {
    map: HashMap<Ptr<ast::Decl>, (), ()>,
    hash_builder: DefaultHashBuilder,
}

impl UnorderedDeclMap {
    fn with_capacity(cap: usize) -> UnorderedDeclMap {
        UnorderedDeclMap {
            map: HashMap::with_capacity_and_hasher(cap, ()),
            hash_builder: DefaultHashBuilder::default(),
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn try_insert(&mut self, decl: Ptr<ast::Decl>) -> Result<(), Ptr<Decl>> {
        let hasher = |d: &Ptr<Decl>| hash_val(&self.hash_builder, d.ident.sym);
        let hash = hasher(&decl);
        match self.map.raw_entry_mut().from_hash(hash, |d| d.ident.sym == decl.ident.sym) {
            RawEntryMut::Occupied(val) => Err(*val.get_key_value().0),
            RawEntryMut::Vacant(slot) => {
                slot.insert_with_hasher(hash, decl, (), hasher);
                Ok(())
            },
        }
    }

    fn get(&self, sym: Symbol, ignore_non_const: bool) -> Option<Ptr<Decl>> {
        let hash = hash_val(&self.hash_builder, sym);
        self.map
            .raw_entry()
            .from_hash(hash, |d| d.ident.sym == sym && (!ignore_non_const || d.is_const))
            .map(|(d, ())| *d)
    }
}

fn verify_no_duplicates(
    scope_kind: ScopeKind,
    decls: Ptr<[Ptr<Decl>]>,
) -> Option<UnorderedDeclMap> {
    debug_assert!(!scope_kind.allows_shadowing());
    if decls.len() <= SMALL_SCOPE_MAX_SIZE {
        for (idx, decl) in decls.into_iter().enumerate() {
            debug_assert!(decl.on_type.is_none());
            if let Some(dup) = linear_search_symbol(&decls[..idx], decl.ident.sym, false, false) {
                error_duplicate_in_unordered_scope(scope_kind, decl, dup);
            }
        }
        None
    } else {
        let mut map = UnorderedDeclMap::with_capacity(decls.len());
        for decl in decls {
            debug_assert!(decl.on_type.is_none());
            if let Err(dup) = map.try_insert(decl) {
                error_duplicate_in_unordered_scope(scope_kind, decl, dup);
            }
        }
        Some(map)
    }
}

fn error_duplicate_in_unordered_scope(
    scope_kind: ScopeKind,
    decl: Ptr<ast::Decl>,
    first: Ptr<ast::Decl>,
) {
    match scope_kind {
        ScopeKind::Root | ScopeKind::File => {
            cerror!(decl.ident.span, "duplicate definition in file scope");
        },
        ScopeKind::Struct | ScopeKind::Union | ScopeKind::Enum => {
            let scope_label = match scope_kind {
                ScopeKind::Struct => "struct",
                ScopeKind::Union => "union",
                ScopeKind::Enum => "enum",
                _ => unreachable_debug(),
            };
            if !decl.might_need_precompilation() && !first.might_need_precompilation() {
                let item_label = if scope_kind == ScopeKind::Enum { "variant" } else { "field" };
                cerror!(
                    decl.ident.span,
                    "duplicate {scope_label} {item_label} `{}`",
                    decl.ident.sym
                );
            } else {
                cerror!(
                    decl.ident.span,
                    "duplicate symbol `{}` in {scope_label} scope",
                    decl.ident.sym
                );
            }
        },
        ScopeKind::ForLoop => todo!(),
        ScopeKind::Fn => {
            cerror!(decl.ident.span, "duplicate parameter '{}'", decl.ident.sym);
        },
        ScopeKind::Block => unreachable_debug(),
    }
    chint!(first.ident.span, "first definition of '{}'", decl.ident.sym);
}
