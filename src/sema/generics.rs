use crate::{
    arena_allocator::Arena,
    ast::{
        self, Ast, AstKind, CloneAst, DeclFlags, EnumFlags, FnFlags, GenericSlotFlags, HasAstKind,
        StructFlags, TypeVariant, UpcastToAst, inherit_ast, inherit_type,
    },
    diagnostics::HandledErr,
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    scope::{Scope, ScopeFlags, ScopeKind, error_duplicate_in_unordered_scope},
    sema::accumulate_type_inner,
    type_::{ty_match, ty_match_quiet},
    util::{BitFlags, OptionExt, UnwrapDebug, then, unreachable_debug},
};

/// like [`ty_match`]
pub fn generic_val_match(
    got: Ptr<ast::ConstVal>,
    expected: Ptr<ast::ConstVal>,
    quiet: bool,
) -> bool {
    if let Some(expected) = expected.try_downcast_type() {
        let Some(got) = got.try_downcast_type() else { return false };
        return ty_match_quiet(got, expected, quiet);
    } else {
        generic_val_eq(got, expected)
    }
}

/// used to check if instantiations are equal
pub fn generic_val_eq(got: Ptr<ast::ConstVal>, expected: Ptr<ast::ConstVal>) -> bool {
    if let Some(expected) = expected.try_downcast_type() {
        let Some(got) = got.try_downcast_type() else { return false };
        return got == expected;
    }
    if got.kind != expected.kind {
        return false;
    }
    // currently no coercion allowed
    match expected.matchable2() {
        ast::ConstValMatch::IntVal(expected) => {
            let got = got.downcast::<ast::IntVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::FloatVal(expected) => {
            let got = got.downcast::<ast::FloatVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::BoolVal(expected) => {
            let got = got.downcast::<ast::BoolVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::CharVal(expected) => {
            let got = got.downcast::<ast::CharVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::StrVal(expected) => {
            let got = got.downcast::<ast::StrVal>();
            got.text.as_ref() == expected.text.as_ref()
        },
        ast::ConstValMatch::RawPtrVal(expected) => {
            let got = got.downcast::<ast::RawPtrVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::StaticPtrVal(expected) => {
            let got = got.downcast::<ast::StaticPtrVal>();
            got.sym == expected.sym
        },
        ast::ConstValMatch::EnumVal(expected) => {
            debug_assert!(expected.is_valid);
            let got = got.downcast::<ast::EnumVal>();
            if !got.is_valid {
                return false;
            }

            ty_match_quiet(got.enum_ty.upcast_to_type(), expected.enum_ty.upcast_to_type(), true)
                && got.variant_idx == expected.variant_idx
                && got.data.zip(expected.data).map(|(g, e)| generic_val_eq(g, e)).unwrap_or(true)
        },
        ast::ConstValMatch::OptionalVal(expected) => {
            let got = got.downcast::<ast::OptionalVal>();
            generic_val_eq(got.val.u(), expected.val.u())
        },
        ast::ConstValMatch::AggregateVal(expected) => {
            let got = got.downcast::<ast::AggregateVal>();
            if !ty_match(got.ty.u(), expected.ty.u()) {
                return false;
            }
            debug_assert_eq!(got.elements.len(), expected.elements.len());
            got.elements
                .into_iter()
                .zip(expected.elements)
                .all(|(g, e)| generic_val_eq(g, e))
        },
        _ => todo!(),
    }
}

pub fn accumulate_generic(
    generic: Ptr<ast::GenericSlot>,
    next_val: Ptr<ast::ConstVal>,
    quiet: bool,
) -> bool {
    debug_assert!(!next_val.p_eq(generic));
    match &mut generic.as_mut().cur_inst {
        cur_inst @ None => {
            if !quiet {
                *cur_inst = Some(next_val);
            }
            true
        },
        Some(generic) => {
            if let Some(ty_acc) = generic.try_downcast_type_ref() {
                let Some(next_ty) = next_val.try_downcast_type() else { return false };
                accumulate_type_inner(ty_acc, next_ty, None, quiet).is_ok()
            } else {
                generic_val_eq(next_val, *generic)
            }
        },
    }
}

pub trait PolymorphableType: TypeVariant + CloneAst<Ptr<Self>> + 'static {
    type Flags: BitFlags + 'static;

    const FLAG_IS_GENERIC: Self::Flags;
    const FLAG_IS_INSTANTIATION: Self::Flags;
    const FLAG_IS_INSTANTIATION_WITH_GENERICS: Self::Flags;

    fn main_scope(self: Ptr<Self>) -> &'static mut Scope;

    fn generics_scope(&self) -> OPtr<Scope>;

    fn polymorphs(self: Ptr<Self>) -> &'static mut Vec<Ptr<Self>>;

    fn flags(self: Ptr<Self>) -> &'static mut Self::Flags;

    fn generics(&self) -> &[Ptr<ast::Decl>] {
        self.generics_scope().map(|s| s.as_ref().decls.as_slice()).unwrap_or(&[])
    }

    fn try_get_polymorphs(self: Ptr<Self>) -> OPtr<[Ptr<Self>]> {
        if self.flags().get(Self::FLAG_IS_GENERIC) {
            Some(Ptr::from_ref(&self.polymorphs()[..]))
        } else {
            debug_assert!(self.polymorphs().is_empty());
            None
        }
    }

    fn instantiations(self: &Ptr<Self>) -> &[Ptr<Self>] {
        self.try_get_polymorphs().unwrap_or(Ptr::from_ref(self).as_slice1()).as_ref()
    }
}

inherit_type! {
    #[derive(Debug)]
    struct Polymorphable {}
}

macro_rules! impl_PolymorphableType {
    ($($ty:ident, $flags_ty:ty, $main_scope:ident);* $(;)?) => {
        $(impl PolymorphableType for ast::$ty {
            type Flags = $flags_ty;

            const FLAG_IS_GENERIC: Self::Flags = Self::Flags::IS_GENERIC;
            const FLAG_IS_INSTANTIATION: Self::Flags = Self::Flags::IS_INSTANTIATION;
            const FLAG_IS_INSTANTIATION_WITH_GENERICS: Self::Flags = Self::Flags::IS_INSTANTIATION_WITH_GENERICS;

            #[inline]
            fn main_scope(self: Ptr<Self>) -> &'static mut Scope {
                &mut self.as_mut().$main_scope
            }

            #[inline]
            fn generics_scope(&self) -> OPtr<Scope> {
                self.generics_scope
            }

            #[inline]
            fn polymorphs(self: Ptr<Self>) -> &'static mut Vec<Ptr<Self>> {
                &mut self.as_mut().polymorphs
            }

            #[inline]
            fn flags(self: Ptr<Self>) -> &'static mut Self::Flags {
                &mut self.as_mut().flags
            }
        })*

        pub const POLYMORPHABLE_AST_KINDS: &[AstKind] = &[$(<ast::$ty as ast::AstVariant>::KIND),*];

        unsafe impl UpcastToAst for Polymorphable {}
        impl Polymorphable {
            pub fn main_scope(self: Ptr<Self>) -> &'static mut Scope {
                match self.upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => t.main_scope(),)*
                    _ => unreachable_debug(),
                }
            }

            pub fn generics_scope(&self) -> OPtr<Scope> {
                match Ptr::from_ref(self).upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => t.generics_scope(),)*
                    _ => unreachable_debug(),
                }
            }

            pub fn is_generic(&self) -> bool {
                match Ptr::from_ref(self).upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => t.flags.get(ast::$ty::FLAG_IS_GENERIC),)*
                    _ => unreachable_debug(),
                }
            }

            pub fn is_instantiation(&self) -> bool {
                match Ptr::from_ref(self).upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => t.flags.get(ast::$ty::FLAG_IS_INSTANTIATION),)*
                    _ => unreachable_debug(),
                }
            }

            pub fn is_instantiation_with_generics(&self) -> bool {
                match Ptr::from_ref(self).upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => t.flags.get(ast::$ty::FLAG_IS_INSTANTIATION_WITH_GENERICS),)*
                    _ => unreachable_debug(),
                }
            }

            pub fn try_get_polymorphs(self: Ptr<Self>) -> OPtr<[Ptr<ast::Type>]> {
                Some(match self.upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => t.try_get_polymorphs()?.cast_slice(),)*
                    _ => unreachable_debug(),
                })
            }

            pub fn take_generics(&mut self) -> Ptr<Scope> {
                match Ptr::from_ref(self).upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => {
                        debug_assert!(t.flags.get(ast::$ty::FLAG_IS_GENERIC));
                        t.as_mut().flags.unset(ast::$ty::FLAG_IS_GENERIC);
                        t.as_mut().generics_scope.take().u()
                    },)*
                    _ => unreachable_debug(),
                }
            }
        }

        pub enum PolymorphableMatch {
            $($ty(Ptr<ast::$ty>),)+
        }

        impl Polymorphable {
            pub fn matchable2(self: Ptr<Polymorphable>) -> PolymorphableMatch {
                match self.kind {
                    $(AstKind::$ty => PolymorphableMatch::$ty(self.upcast().flat_downcast::<ast::$ty>()),)+
                    _ => unreachable_debug(),
                }
            }
        }
    };
}
impl_PolymorphableType! {
    Fn, FnFlags, params_scope;
    StructDef, StructFlags, scope;
    EnumDef, EnumFlags, scope;
}

impl ast::Ast {
    pub fn try_downcast_polymorphable(self: Ptr<Self>) -> OPtr<Polymorphable> {
        self.rep().try_flat_downcast_polymorphable()
    }

    pub fn try_flat_downcast_polymorphable(self: Ptr<Self>) -> OPtr<Polymorphable> {
        then!(POLYMORPHABLE_AST_KINDS.contains(&self.kind) => self.downcast_polymorphable())
    }

    pub fn downcast_polymorphable(self: Ptr<Self>) -> Ptr<Polymorphable> {
        self.rep().flat_downcast_polymorphable()
    }

    pub fn flat_downcast_polymorphable(self: Ptr<Self>) -> Ptr<Polymorphable> {
        debug_assert!(POLYMORPHABLE_AST_KINDS.contains(&self.kind));
        self.cast()
    }
}

impl ast::Type {
    pub fn downcast_polymorphable(self: Ptr<Self>) -> Ptr<Polymorphable> {
        debug_assert!(self.replacement.is_none());
        self.upcast().flat_downcast_polymorphable()
    }

    pub fn try_downcast_polymorphable(self: Ptr<Self>) -> OPtr<Polymorphable> {
        debug_assert!(self.replacement.is_none());
        self.upcast().try_flat_downcast_polymorphable()
    }
}

pub fn steal_generics(
    decl_with_generic: Ptr<ast::Decl>,
    target: Ptr<ast::Fn>,
    alloc: &Arena,
) -> Result<(), HandledErr> {
    debug_assert!(decl_with_generic.flags.get(DeclFlags::CONTAINS_GENERIC));

    let ty = decl_with_generic.var_ty.u().downcast_polymorphable();
    debug_assert!(ty.is_generic());
    let generics = ty.as_mut().take_generics();
    for g in generics.decls.iter().copied() {
        debug_assert_eq!(g.scope.u(), generics);
        g.as_mut().scope = None;
        g.generic.u().as_mut().flags.set(GenericSlotFlags::WAS_STOLEN);

        add_generic_to_fn(target, g, alloc)?;
    }

    debug_assert!(target.flags.get(FnFlags::IS_GENERIC));
    Ok(())
}

pub fn add_generic_to_fn(
    f: Ptr<ast::Fn>,
    generic_decl: Ptr<ast::Decl>,
    alloc: &Arena,
) -> Result<(), HandledErr> {
    debug_assert!(generic_decl.flags.get(DeclFlags::IS_GENERIC_DEF));
    let generic = generic_decl.generic.u();
    debug_assert_eq!(generic.name.decl, generic_decl);

    let generics_scope = f.as_mut().generics_scope.get_or_insert_with(|| {
        let s = alloc.alloc(Scope::new(vec![], ScopeKind::FnGenerics)).unwrap();
        s.as_mut().expr.set_once(f.upcast());
        debug_assert!(f.params_scope.flags.get(ScopeFlags::WAS_SETUP));
        s.as_mut().setup(f.params_scope.parent.u());
        f.as_mut().params_scope.parent = Some(s);
        s
    });
    if let Result::Err(dup) = generics_scope.add_decl(generic_decl) {
        return Err(error_duplicate_in_unordered_scope(ScopeKind::FnGenerics, generic_decl, dup));
    }
    debug_assert!(!f.flags.get(FnFlags::IS_INSTANTIATION));
    f.as_mut().flags.set(FnFlags::IS_GENERIC);
    generic.as_mut().flags.set(GenericSlotFlags::WAS_ADDED_TO_SCOPE);

    Ok(())
}
