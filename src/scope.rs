use crate::{
    arena_allocator::{AllocErr, Arena},
    ast::{self, Decl, DeclList},
    context::ctx,
    diagnostics::{DiagnosticReporter, cerror, chint},
    intern_pool::Symbol,
    ptr::{OPtr, Ptr},
    util::unreachable_debug,
};
use hashbrown::HashMap;

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

    /// also returns the fields as a [`DeclList`].
    pub fn for_aggregate(
        fields: Vec<Ptr<Decl>>,
        consts: Vec<Ptr<Decl>>,
        alloc: &Arena,
        kind: ScopeKind,
    ) -> Result<ScopeAndAggregateInfo, AllocErr> {
        debug_assert!(kind.is_aggregate());
        let fields = alloc.alloc_slice(&fields)?;
        let scope = Scope::new(alloc.alloc_slice(&consts)?, kind);
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

    pub fn find_decl_norec(&self, sym: Symbol, cur_pos: ScopePos) -> OPtr<Decl> {
        #[cfg(debug_assertions)]
        debug_assert!(self.was_checked_for_duplicates);
        //debug_assert_eq!(self.decls_map.is_some(), self.decls.len() > SMALL_SCOPE_MAX_SIZE);
        if let Some(decls_map) = self.decls_map.as_ref() {
            debug_assert!(ctx().do_abort_compilation() || decls_map.len() == self.decls.len());
            decls_map.get(sym)
        } else {
            let decls = if self.kind.allows_shadowing() {
                &self.decls[..cur_pos.0 as usize]
            } else {
                &self.decls
            };
            linear_search_symbol(decls, sym, true)
        }
    }

    pub fn find_decl(&self, sym: Symbol, cur_pos: ScopePos) -> OPtr<Decl> {
        let mut cur_scope = Some(Ptr::from_ref(self));
        let mut cur_pos = cur_pos;
        while let Some(scope) = cur_scope {
            if let Some(sym) = scope.find_decl_norec(sym, cur_pos) {
                return Some(sym);
            }
            cur_scope = scope.parent;
            cur_pos = scope.pos_in_parent;
        }
        return None;
    }

    /// assumes `self` is an unordered scope. see [`ScopeKind::allows_shadowing`].
    pub fn find_decl_unordered(&self, sym: Symbol) -> OPtr<Decl> {
        debug_assert!(!self.kind.allows_shadowing());
        self.find_decl(sym, ScopePos::UNSET)
    }
}

/// `reverse` is needed because of shadowing
fn linear_search_symbol(decls: &[Ptr<Decl>], sym: Symbol, reverse: bool) -> Option<Ptr<Decl>> {
    debug_assert!(decls.iter().all(|d| d.on_type.is_none()));
    if reverse {
        decls.iter().copied().rfind(|d| d.ident.sym == sym)
    } else {
        decls.iter().copied().find(|d| d.ident.sym == sym)
    }
}

pub struct ScopeAndAggregateInfo {
    pub scope: Scope,
    pub fields: DeclList,
    pub consts: Vec<Ptr<Decl>>,
}

#[derive(Debug)]
pub struct UnorderedDeclMap {
    map: HashMap<Symbol, Ptr<ast::Decl>>,
}

impl UnorderedDeclMap {
    fn len(&self) -> usize {
        self.map.len()
    }

    fn get(&self, sym: Symbol) -> Option<Ptr<Decl>> {
        self.map.get(&sym).copied()
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
            if let Some(dup) = linear_search_symbol(&decls[..idx], decl.ident.sym, false) {
                error_duplicate_in_unordered_scope(scope_kind, decl, dup);
            }
        }
        None
    } else {
        let mut map = HashMap::with_capacity(decls.len());
        for decl in decls {
            debug_assert!(decl.on_type.is_none());
            if let Err(e) = map.try_insert(decl.ident.sym, decl) {
                let dup = *e.entry.get();
                error_duplicate_in_unordered_scope(scope_kind, decl, dup);
            }
        }
        Some(UnorderedDeclMap { map })
    }
}

fn error_duplicate_in_unordered_scope(
    scope_kind: ScopeKind,
    decl: Ptr<ast::Decl>,
    first: Ptr<ast::Decl>,
) {
    match scope_kind {
        ScopeKind::Root | ScopeKind::File => {
            cerror!(decl.ident.span, "Duplicate definition in file scope");
            chint!(first.ident.span, "First definition here");
        },
        ScopeKind::Struct | ScopeKind::Union | ScopeKind::Enum => todo!(),
        ScopeKind::ForLoop => todo!(),
        ScopeKind::Fn => {
            cerror!(decl.ident.span, "duplicate parameter '{}'", decl.ident.sym);
            chint!(first.ident.span, "first definition of '{}'", decl.ident.sym);
        },
        ScopeKind::Block => unreachable_debug(),
    }
}
