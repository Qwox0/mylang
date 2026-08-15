use crate::{
    ast::{
        self, Ast, AstKind, CloneAst, EnumFlags, FnFlags, HasAstKind, StructFlags, TypeVariant,
        UpcastToAst, inherit_ast,
    },
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    scope::Scope,
    sema::accumulate_type_inner,
    type_::ty_match,
    util::{BitFlags, UnwrapDebug, then, unreachable_debug},
};

pub fn generic_match(got: Ptr<ast::ConstVal>, expected: Ptr<ast::ConstVal>) -> bool {
    if let Some(expected) = expected.try_downcast_type() {
        let Some(got) = got.try_downcast_type() else { return false };
        return ty_match(got, expected);
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

            ty_match(got.enum_ty.upcast_to_type(), expected.enum_ty.upcast_to_type())
                && got.variant_idx == expected.variant_idx
                && got.data.zip(expected.data).map(|(g, e)| generic_match(g, e)).unwrap_or(true)
        },
        ast::ConstValMatch::OptionalVal(expected) => {
            let got = got.downcast::<ast::OptionalVal>();
            generic_match(got.val.u(), expected.val.u())
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
                .all(|(g, e)| generic_match(g, e))
        },
        _ => todo!(),
    }
}

pub fn accumulate_generic(generic: Ptr<ast::GenericDef>, next_val: Ptr<ast::ConstVal>) -> bool {
    let expr = None; // quiet
    match &mut generic.as_mut().cur_inst {
        cur_inst @ None => {
            *cur_inst = Some(next_val);
            true
        },
        Some(generic) => {
            if let Some(ty_acc) = generic.try_downcast_type_ref() {
                let Some(next_ty) = next_val.try_downcast_type() else { return false };
                accumulate_type_inner(ty_acc, next_ty, expr).is_ok()
            } else {
                generic_match(next_val, *generic)
            }
        },
    }
}

pub trait PolymorphableType: TypeVariant + CloneAst<Ptr<Self>> + 'static {
    type Flags: BitFlags + 'static;

    const FLAG_IS_GENERIC: <Self::Flags as BitFlags>::Repr;
    const FLAG_IS_INSTANTIATION: <Self::Flags as BitFlags>::Repr;

    fn main_scope(self: Ptr<Self>) -> &'static mut Scope;

    fn generics_scope(&self) -> OPtr<Scope>;

    fn instantiations(self: Ptr<Self>) -> &'static mut Vec<Ptr<Self>>;

    fn flags(self: Ptr<Self>) -> &'static mut Self::Flags;

    fn generics(&self) -> &[Ptr<ast::Decl>] {
        self.generics_scope().map(|s| s.as_ref().decls.as_slice()).unwrap_or(&[])
    }

    fn try_get_polymorphs(self: Ptr<Self>) -> OPtr<[Ptr<Self>]> {
        if self.flags().get(Self::FLAG_IS_GENERIC) {
            Some(Ptr::from_ref(&self.instantiations()[..]))
        } else {
            debug_assert!(self.instantiations().is_empty());
            None
        }
    }

    fn polymorphs_or_self(self: &Ptr<Self>) -> &[Ptr<Self>] {
        self.try_get_polymorphs().unwrap_or(Ptr::from_ref(self).as_slice1()).as_ref()
    }
}

inherit_ast! {
    struct Polymorphable {}
}

macro_rules! impl_PolymorphableType {
    ($($ty:ident, $flags_ty:ty, $main_scope:ident);* $(;)?) => {
        $(impl PolymorphableType for ast::$ty {
            type Flags = $flags_ty;

            const FLAG_IS_GENERIC: <Self::Flags as BitFlags>::Repr = Self::Flags::IS_GENERIC;
            const FLAG_IS_INSTANTIATION: <Self::Flags as BitFlags>::Repr =
                Self::Flags::IS_INSTANTIATION;

            #[inline]
            fn main_scope(self: Ptr<Self>) -> &'static mut Scope {
                &mut self.as_mut().$main_scope
            }

            #[inline]
            fn generics_scope(&self) -> OPtr<Scope> {
                self.generics_scope
            }

            #[inline]
            fn instantiations(self: Ptr<Self>) -> &'static mut Vec<Ptr<Self>> {
                &mut self.as_mut().instantiations
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

            pub fn try_get_polymorphs(self: Ptr<Self>) -> OPtr<[Ptr<ast::Type>]> {
                Some(match self.upcast().matchable2() {
                    $(ast::AstMatch::$ty(t) => t.try_get_polymorphs()?.cast_slice(),)*
                    _ => unreachable_debug(),
                })
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
    pub fn try_downcast_polymorphable(self: Ptr<Self>) -> OPtr<Polymorphable> {
        debug_assert!(self.replacement.is_none());
        self.upcast().try_flat_downcast_polymorphable()
    }
}
