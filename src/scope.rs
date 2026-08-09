use crate::{
    arena_allocator::{AllocErr, Arena},
    ast::{self, Ast, AstKind, Decl, DeclFlags, DeclList, UpcastToAst},
    context::{ctx, primitives},
    diagnostics::{DiagnosticReporter, HandledErr, cerror, chint},
    intern_pool::Symbol,
    ptr::{OPtr, Ptr},
    util::{
        BitFlags, OptionExt, UnwrapDebug, bitflags, debug_only_assert, hash_val, panic_debug,
        unreachable_debug,
    },
};
use hashbrown::{DefaultHashBuilder, HashMap, hash_map::RawEntryMut};
use std::assert_matches::debug_assert_matches;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Root,
    File,
    Block,
    ForLoop,
    SwitchCase,
    FnParams,
    Struct,
    Union,
    Enum,

    Generics,
}

impl ScopeKind {
    pub fn allows_shadowing(self) -> bool {
        matches!(self, ScopeKind::Block)
    }

    pub fn is_aggregate(self) -> bool {
        matches!(self, ScopeKind::Struct | ScopeKind::Union | ScopeKind::Enum)
    }

    pub fn for_container(container_kind: AstKind) -> ScopeKind {
        match container_kind {
            AstKind::Block => ScopeKind::Block,
            AstKind::For => ScopeKind::ForLoop,
            AstKind::While => todo!(),
            AstKind::StructDef => ScopeKind::Struct,
            AstKind::UnionDef => ScopeKind::Union,
            AstKind::EnumDef => ScopeKind::Enum,
            AstKind::Fn => ScopeKind::FnParams,
            k => panic_debug!("{k:?} doesn't contain a scope"),
        }
    }
}

#[derive(Debug)]
pub struct Scope {
    pub kind: ScopeKind,
    pub flags: ScopeFlags,
    pub parent: OPtr<Scope>,
    /// TODO: use `struct { ptr: *mut T, len: u32, cap: u32 }` instead
    pub decls: Vec<Ptr<Decl>>,
    /// used for symbol lookups when this scope has more than [`SMALL_SCOPE_MAX_SIZE`] Decls.
    /// Set at the start of sema, if needed
    ///
    /// Currently only used for unordered scopes because those scopes don't allow shadowing.
    decls_map: Option<UnorderedDeclMap>,
}

bitflags!(ScopeFlags: u8 {
    WAS_CHECKED_FOR_DUPLICATES,
    WAS_SETUP,
});

const SMALL_SCOPE_MAX_SIZE: usize = 32;

impl Scope {
    pub fn new(decls: Vec<Ptr<Decl>>, kind: ScopeKind) -> Scope {
        debug_assert!(decls.iter().all(|d| d.on_type.is_none()));
        Scope { parent: None, decls, decls_map: None, kind, flags: ScopeFlags::default() }
    }

    pub fn for_generics(generics: Vec<Ptr<Decl>>) -> Scope {
        debug_assert!(generics.iter().all(|p| p.flags.get(DeclFlags::IS_GENERIC)));
        debug_assert!(generics.iter().all(|p| p.is_const));
        Scope::new(generics, ScopeKind::Generics)
    }

    pub fn setup(&mut self, parent: Ptr<Scope>) {
        if self.flags.get(ScopeFlags::WAS_SETUP) {
            debug_assert!(self.parent.u() == parent);
            debug_assert!(self.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES));
            return;
        }
        self.flags.set(ScopeFlags::WAS_SETUP);
        self.parent.set_once(parent);
        let _ignore = self.verify_no_duplicates();
    }

    pub fn from_stmts(stmts: &[Ptr<ast::Ast>], kind: ScopeKind) -> Scope {
        let decls = stmts
            .iter()
            .filter_map(|s| s.try_downcast::<Decl>())
            .filter(|d| d.on_type.is_none())
            .collect::<Vec<_>>();
        Scope::new(decls, kind)
    }

    pub fn file(stmts: &[Ptr<ast::Ast>], parent_scope: Ptr<Scope>) -> Scope {
        let mut scope = Scope::from_stmts(stmts, ScopeKind::File);
        scope.parent = Some(parent_scope);
        debug_assert!(!parent_scope.kind.allows_shadowing());
        scope
    }

    /// also returns the fields as a [`DeclList`].
    pub fn for_aggregate(
        decls: Vec<Ptr<Decl>>,
        alloc: &Arena,
        kind: ScopeKind,
    ) -> Result<ScopeAndAggregateInfo, AllocErr> {
        debug_assert!(kind.is_aggregate());
        let mut fields_buf = Vec::with_capacity(decls.len());
        for d in decls.iter().copied() {
            if d.is_const {
                d.as_mut().flags.set(DeclFlags::IS_CONST_MEMBER);
            } else {
                d.as_mut().flags.set(DeclFlags::IS_DATA_MEMBER);
                fields_buf.push(d);
            }
        }
        let fields = alloc.alloc_slice(&fields_buf)?; // fields are allocated twice because `scope.decls` is rearranged during sema.
        Ok(ScopeAndAggregateInfo { scope: Scope::new(decls, kind), fields })
    }

    /// also sets up [`Scope::decls_map`] if needed.
    pub fn verify_no_duplicates(&mut self) -> Result<(), HandledErr> {
        debug_assert!(!self.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES));
        let mut err = false;
        if !self.kind.allows_shadowing() {
            if self.decls.len() <= SMALL_SCOPE_MAX_SIZE {
                for (idx, decl) in self.decls.iter().copied().enumerate() {
                    debug_assert!(decl.on_type.is_none());
                    if let Some(dup) =
                        linear_search_symbol(&self.decls[..idx], decl.ident.sym, false, false)
                    {
                        error_duplicate_in_unordered_scope(self.kind, decl, dup);
                        err = true;
                        for d in &self.decls {
                            if d.ident.sym == decl.ident.sym {
                                d.as_mut().var_ty = Some(primitives().err_ty);
                            }
                        }
                    }
                }
                debug_assert_matches!(self.decls_map, None)
            } else {
                let mut map = UnorderedDeclMap::with_capacity(self.decls.len());
                for &decl in self.decls.iter() {
                    debug_assert!(decl.on_type.is_none());
                    if let Err(dup) = map.try_insert_no_dup(decl) {
                        error_duplicate_in_unordered_scope(self.kind, decl, dup);
                        err = true;
                    }
                }
                self.decls_map = Some(map);
            }
        }
        self.flags.set(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES);
        if err { Err(HandledErr) } else { Ok(()) }
    }

    pub fn add_decl_to_block(&mut self, decl: Ptr<Decl>) {
        debug_assert!(self.kind.allows_shadowing());
        debug_assert!(self.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES));
        debug_assert!(decl.on_type.is_none());
        if let Some(map) = self.decls_map.as_mut() {
            map.insert_or_replace(decl);
        }
        self.decls.push(decl);
    }

    pub fn add_decl(&mut self, decl: Ptr<Decl>) -> Result<(), Ptr<Decl>> {
        if self.kind.allows_shadowing() {
            self.add_decl_to_block(decl);
        } else {
            if let Some(map) = self.decls_map.as_mut() {
                let () = map.try_insert_no_dup(decl)?;
            } else if let Some(dup) =
                linear_search_symbol(&self.decls, decl.ident.sym, false, false)
            {
                return Err(dup);
            }
            self.decls.push(decl);
        }
        Ok(())
    }

    pub fn find_decl_norec(&self, sym: Symbol, ignore_fields: bool) -> OPtr<Decl> {
        debug_only_assert!(self.flags.get(ScopeFlags::WAS_CHECKED_FOR_DUPLICATES));
        debug_assert_eq!(self.decls_map.is_some(), self.decls.len() > SMALL_SCOPE_MAX_SIZE);
        // TODO: remove this
        let ignore_non_const =
            ignore_fields && self.kind.is_aggregate() && self.kind != ScopeKind::Enum;
        if let Some(decls_map) = self.decls_map.as_ref() {
            debug_assert!(ctx().do_abort_compilation() || decls_map.len() == self.decls.len());
            decls_map.get(sym, ignore_non_const)
        } else {
            linear_search_symbol(&self.decls, sym, self.kind.allows_shadowing(), ignore_non_const)
        }
    }

    pub fn find_decl(&self, sym: Symbol) -> OPtr<Decl> {
        let mut cur_scope = Some(Ptr::from_ref(self));
        while let Some(scope) = cur_scope {
            if let Some(sym) = scope.find_decl_norec(sym, true) {
                return Some(sym);
            }
            cur_scope = scope.parent;
        }
        return None;
    }

    // Currently only for debugging
    pub fn get_expr(self: Ptr<Self>) -> OPtr<Ast> {
        macro_rules! get_scope_container {
            ($scope:expr, $container_ty:path, $scope_field:ident) => {{
                use crate::ast::AstVariant;
                let scope: Ptr<Scope> = $scope;
                debug_assert_eq!(scope.kind, ScopeKind::for_container(<$container_ty>::KIND));
                crate::util::assert_has_field!($container_ty, $scope_field: Scope);
                scope
                    .byte_sub(std::mem::offset_of!($container_ty, $scope_field))
                    .cast::<$container_ty>()
            }};
        }

        Some(match self.kind {
            ScopeKind::Root | ScopeKind::File => return None,
            ScopeKind::Block => get_scope_container!(self, ast::Block, decl_scope).upcast(),
            ScopeKind::ForLoop => get_scope_container!(self, ast::For, scope).upcast(),
            ScopeKind::SwitchCase => todo!(),
            ScopeKind::FnParams => get_scope_container!(self, ast::Fn, params_scope).upcast(),
            ScopeKind::Struct => get_scope_container!(self, ast::StructDef, scope).upcast(),
            ScopeKind::Union => get_scope_container!(self, ast::UnionDef, scope).upcast(),
            ScopeKind::Enum => get_scope_container!(self, ast::EnumDef, scope).upcast(),
            ScopeKind::Generics => todo!(),
        })
    }
}

pub fn setup_scopes(item_scope: &mut Scope, generics_scope: OPtr<Scope>, parent: Ptr<Scope>) {
    if item_scope.parent.is_none() {
        if let Some(generics_scope) = generics_scope {
            debug_assert_eq!(generics_scope.kind, ScopeKind::Generics);
            generics_scope.as_mut().setup(parent);
            item_scope.setup(generics_scope);
        } else {
            item_scope.setup(parent);
        }
    } else {
        #[cfg(debug_assertions)]
        if let Some(generics_scope) = generics_scope {
            debug_assert!(generics_scope.parent.u() == parent);
            debug_assert!(item_scope.parent.u() == generics_scope);
        } else {
            debug_assert!(item_scope.parent.u() == parent);
        }
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

    fn eq(decl: Ptr<Decl>) -> impl Fn(&Ptr<Decl>) -> bool {
        move |d: &Ptr<Decl>| d.ident.sym == decl.ident.sym
    }

    fn try_insert_no_dup(&mut self, decl: Ptr<ast::Decl>) -> Result<(), Ptr<Decl>> {
        let hasher = |d: &Ptr<Decl>| hash_val(&self.hash_builder, d.ident.sym);
        let hash = hasher(&decl);
        match self.map.raw_entry_mut().from_hash(hash, Self::eq(decl)) {
            RawEntryMut::Occupied(val) => Err(*val.get_key_value().0),
            RawEntryMut::Vacant(slot) => {
                slot.insert_with_hasher(hash, decl, (), hasher);
                Ok(())
            },
        }
    }

    fn insert_or_replace(&mut self, decl: Ptr<Decl>) -> OPtr<Decl> {
        let hasher = |d: &Ptr<Decl>| hash_val(&self.hash_builder, d.ident.sym);
        let hash = hasher(&decl);
        match self.map.raw_entry_mut().from_hash(hash, Self::eq(decl)) {
            RawEntryMut::Occupied(mut val) => Some(val.insert_key(decl)),
            RawEntryMut::Vacant(slot) => {
                slot.insert_with_hasher(hash, decl, (), hasher);
                None
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

#[track_caller]
pub fn error_duplicate_in_unordered_scope(
    scope_kind: ScopeKind,
    decl: Ptr<ast::Decl>,
    first: Ptr<ast::Decl>,
) -> HandledErr {
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
        ScopeKind::SwitchCase => todo!(),
        ScopeKind::FnParams => {
            cerror!(decl.ident.span, "duplicate parameter '{}'", decl.ident.sym);
        },
        ScopeKind::Block => unreachable_debug(),
        ScopeKind::Generics => {
            cerror!(decl.ident.span, "duplicate generic '${}'", decl.ident.sym);
        },
    }
    chint!(first.ident.span, "first definition of '{}'", decl.ident.sym);
    HandledErr
}
