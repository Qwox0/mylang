use crate::{
    ast::{Expr, Fn, Ident, VarDecl, VarDeclList, VarDeclListTrait, debug::DebugAst},
    ptr::Ptr,
    util::{
        aligned_add, panic_debug, round_up_to_nearest_power_of_two, variant_count_to_tag_size_bits,
        variant_count_to_tag_size_bytes,
    },
};
use core::fmt;

// TODO: benchmark this
// pub type Type = Ptr<TypeInfo>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Void,
    Never,
    Int {
        bits: u32,
        is_signed: bool,
    },
    Bool,
    Float {
        bits: u32,
    },

    Ptr {
        pointee_ty: Ptr<Type>,
    },
    Slice {
        elem_ty: Ptr<Type>,
    },
    Array {
        len: usize,
        elem_ty: Ptr<Type>,
    },

    Function(Ptr<Fn>),

    Struct {
        fields: VarDeclList,
    },
    Union {
        fields: VarDeclList,
    },
    Enum {
        variants: VarDeclList, // TODO
    },

    Range {
        elem_ty: Ptr<Type>,
    },
    RangeInclusive {
        elem_ty: Ptr<Type>,
    },

    Option {
        ty: Ptr<Type>,
    },

    /// `a :: int`
    /// -> type of `a`: [`Type::Type`]
    /// -> value of `a`: [`Type::Int`]
    Type(Ptr<Type>),

    // --------------------------------
    // compiletime only types:
    //
    IntLiteral,
    FloatLiteral,

    /// ```mylang
    /// MyEnum :: enum { A, B(i64) };
    /// a := MyEnum.A; // variant -> valid val
    /// b1 := MyEnum.B; // variant -> invalid val
    /// b2 := MyEnum.B(5); // variant initialization
    /// ```
    EnumVariant {
        enum_ty: Ptr<Type>,
        idx: usize,
    },

    /// The type was not explicitly set in the original source code and must
    /// still be inferred.
    Unset,
    /// The type was explicitly set in the original source code, but hasn't been
    /// analyzed yet.
    Unevaluated(Ptr<Expr>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Function(arg0) => f.debug_tuple("Function").field(arg0).finish(),
            Type::Unset => write!(f, "Unset"),
            Type::Unevaluated(arg0) => f.debug_tuple("Unevaluated").field(arg0).finish(),
            ti => write!(f, "{}", ti.to_text()),
        }
    }
}

impl Type {
    pub const F32: Type = Type::Float { bits: 32 };
    pub const F64: Type = Type::Float { bits: 64 };
    pub const I64: Type = Type::Int { bits: 64, is_signed: true };
    pub const U64: Type = Type::Int { bits: 64, is_signed: false };

    pub fn try_internal_ty(name: Ptr<str>) -> Option<Self> {
        match name.bytes().next()? {
            b'i' => name[1..].parse().ok().map(|bits| Type::Int { bits, is_signed: true }),
            b'u' => name[1..].parse().ok().map(|bits| Type::Int { bits, is_signed: false }),
            b'f' => name[1..].parse().ok().map(|bits| Type::Float { bits }),
            _ if &*name == "void" => Some(Type::Void),
            _ if &*name == "never" => Some(Type::Never),
            _ => None,
        }
    }

    pub fn ptr_unset() -> Ptr<Type> {
        thread_local!(static TYPE: Type = Type::Unset);
        TYPE.with(|t| Ptr::from(t))
    }

    /// returns `Ptr(Type::Ptr(Type::Void))`
    pub fn ptr_void_ptr() -> Ptr<Type> {
        thread_local!(static TYPE: Type = Type::Ptr { pointee_ty: Type::ptr_void() });
        TYPE.with(|t| Ptr::from(t))
    }

    /// returns `Ptr(Type::Void)`
    pub fn ptr_void() -> Ptr<Type> {
        thread_local!(static TYPE: Type = Type::Void);
        TYPE.with(|t| Ptr::from(t))
    }

    pub fn str_slice() -> Type {
        thread_local!(static TYPE: Type = Type::Int { bits: 8, is_signed: false });
        TYPE.with(|t| Type::Slice { elem_ty: Ptr::from(t) })
    }

    /// size of stack allocation in bytes
    pub fn size(&self) -> usize {
        match self {
            Type::Void | Type::Never => 0,
            Type::Int { bits, .. } | Type::Float { bits } => {
                round_up_to_nearest_power_of_two(*bits as usize).div_ceil(8)
            },
            Type::Bool => 1,
            Type::Function(_) => todo!(),
            Type::Ptr { .. } => 8,
            Type::Slice { .. } => 16,
            Type::Array { len, elem_ty } => elem_ty.size() * len,
            Type::Struct { fields } => fields.iter().map(|f| &f.ty).fold(0, aligned_add),
            Type::Union { fields } => fields.iter().map(|f| f.ty.size()).max().unwrap_or(0),
            Type::Enum { variants } => aligned_add(
                variant_count_to_tag_size_bytes(variants.len()) as usize,
                &Type::Union { fields: *variants },
            ),
            Type::Range { elem_ty } | Type::RangeInclusive { elem_ty } => elem_ty.size() * 2,
            Type::Option { ty } if ty.is_non_null() => ty.size(),
            Type::Option { ty } => aligned_add(1, ty),
            Type::Type(_) => 0,
            Type::IntLiteral
            | Type::FloatLiteral
            | Type::EnumVariant { .. }
            | Type::Unset
            | Type::Unevaluated(_) => panic_debug("cannot find stack size"),
        }
    }

    /// alignment of stack allocation in bytes
    pub fn alignment(&self) -> usize {
        const ZST_ALIGNMENT: usize = 1;
        let alignment = match self {
            Type::Void | Type::Never => ZST_ALIGNMENT,
            Type::Int { .. } => self.size().min(16),
            Type::Bool => 1,
            Type::Float { .. } => self.size().min(16),
            Type::Function(_) => todo!(),
            Type::Ptr { .. } | Type::Slice { .. } => 8,
            Type::Array { elem_ty, .. } => elem_ty.alignment(),
            Type::Struct { fields } => {
                fields.iter().map(|f| f.ty.alignment()).max().unwrap_or(ZST_ALIGNMENT)
            },
            Type::Union { fields } => {
                fields.iter().map(|f| f.ty.alignment()).max().unwrap_or(ZST_ALIGNMENT)
            },
            Type::Enum { variants } => {
                Type::Int { bits: variant_count_to_tag_size_bits(variants.len()), is_signed: false }
                    .alignment()
                    .max(Type::Union { fields: *variants }.alignment())
            },
            Type::Range { elem_ty } | Type::RangeInclusive { elem_ty } => elem_ty.alignment(),
            Type::Option { ty } => ty.alignment(),
            Type::IntLiteral
            | Type::FloatLiteral
            | Type::EnumVariant { .. }
            | Type::Type(..)
            | Type::Unset
            | Type::Unevaluated(_) => panic_debug("cannot find stack alignment"),
        };
        debug_assert!(alignment.is_power_of_two());
        alignment
    }

    /// Checks if the two types equal or can be coerced into a common type.
    ///
    /// For exact equality use `==`.
    pub fn matches(self, other: Self) -> bool {
        self.common_type(other).is_some()
    }

    /// Returns the common type after type coercion or [`None`] if the types
    /// don't match (see [`Type::matches`]).
    pub fn common_type(self, other: Self) -> Option<Type> {
        enum Selection {
            Left,
            Right,
        }
        use Selection::*;

        fn inner(l: Type, r: Type) -> Option<Selection> {
            macro_rules! match_and_select {
                (match $i:expr => $($body:tt)*) => { match_and_select!(@inner $i; $($body)*) };
                (@inner $i:expr;) => {};
                (@inner $i:expr; mirror ($l_pat:pat, $r_pat:pat) $(if $guard:expr)? => Some(Left), $($tail:tt)*) => {
                    if let ($l_pat, $r_pat) = $i $( && $guard )? {
                        return Some(Left)
                    } else if let ($r_pat, $l_pat) = $i $( && $guard )? {
                        return Some(Right)
                    }
                    match_and_select!(@inner $i; $($tail)*)
                };
                (@inner $i:expr; $pat:pat $(if $guard:expr)? => $return:expr, $($tail:tt)*) => {
                    #[allow(irrefutable_let_patterns)]
                    if let $pat = $i $( && $guard )? {
                        return $return;
                    }
                    match_and_select!(@inner $i; $($tail)*)
                };
            }

            match_and_select!(match (l, r) =>
                (l, r) if l == r => Some(Left),
                mirror (_, Type::Never) => Some(Left),
                (Type::Slice { elem_ty: t1 }, Type::Slice { elem_ty: t2 }) => inner(*t1, *t2),
                (Type::Array { len: l1, elem_ty: t1 }, Type::Array { len: l2, elem_ty: t2 })
                    if l1 == l2 => inner(*t1, *t2),

                //(Type::Ptr { pointee_ty: p1 }, Type::Ptr { pointee_ty: p2 }) => inner(*p1, *p2),
                // TODO: remove these rules:
                mirror (Type::Ptr { .. } | Type::Int { bits: 64, is_signed: false } | Type::IntLiteral, Type::Ptr { .. }) => Some(Left), // allows `*T == *U`, `*T == int`
                mirror (Type::Ptr { .. } | Type::Int { bits: 64, is_signed: false } | Type::IntLiteral, Type::Option { ty: p }) if matches!(*p, Type::Ptr { .. }) => Some(Left), // allows `?*T == *U`, `?*T == int`
                //mirror (p1 @ Type::Ptr { .. }, Type::Option { ty: p2 }) if match *p2 { Type::Ptr { .. } => inner(p1, *p2).is_some(), _ => false } => Some(Left), // allows `?*T == *T`
                mirror (Type::U64, Type::I64) => Some(Left),

                mirror (Type::Int { .. } | Type::Float { .. } | Type::FloatLiteral, Type::IntLiteral) => Some(Left),
                mirror (Type::Float { .. }, Type::FloatLiteral) => Some(Left),
                mirror (e @ Type::Enum { variants }, Type::EnumVariant { enum_ty, idx })
                    if *enum_ty == e && variants[idx].ty == Type::Void => Some(Left),
                (Type::Option { ty: t1 }, Type::Option { ty: t2 }) => inner(*t1, *t2),
            );
            None
        }

        inner(self, other).map(|s| match s {
            Selection::Left => self,
            Selection::Right => other,
        })
    }

    /// This might mutate values behind [`Ptr`]s in `self`.
    /// Example: the value behind `elem_ty` on [`TypeInfo::Array`] might change.
    pub fn finalize(self) -> Self {
        match self {
            Type::IntLiteral => Type::I64,
            Type::FloatLiteral => Type::F64,
            Type::Array { mut elem_ty, .. }
            | Type::Range { mut elem_ty }
            | Type::RangeInclusive { mut elem_ty } => {
                *elem_ty = elem_ty.finalize();
                self
            },
            Type::EnumVariant { enum_ty, .. } => *enum_ty,
            Type::Unset | Type::Unevaluated(_) => panic_debug("cannot finalize invalid type"),
            t => t,
        }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        match self {
            Type::Unset | Type::Unevaluated(_) => false,
            _ => true,
        }
    }

    /// if `self` [Type::is_valid] this returns `Some(self)`, otherwise [`None`]
    #[inline]
    pub fn into_valid(self) -> Option<Type> {
        match self {
            Type::Unset | Type::Unevaluated(_) => None,
            t => Some(t),
        }
    }

    pub fn is_non_null(&self) -> bool {
        match self {
            Type::Void => todo!(),
            Type::Never => todo!(),
            Type::Int { .. } | Type::Bool | Type::Float { .. } => false,
            Type::Ptr { .. } | Type::Slice { .. } => true,
            Type::Array { .. } => todo!(),
            Type::Function(..) => todo!(),
            Type::Struct { fields } => fields.as_type_iter().any(|t| t.is_non_null()),
            Type::Union { .. } => todo!(),
            Type::Enum { .. } => todo!(),
            Type::Range { .. } => todo!(),
            Type::RangeInclusive { .. } => todo!(),
            Type::Option { .. } => false,
            Type::Type(..) => todo!(),
            Type::IntLiteral => todo!(),
            Type::FloatLiteral => todo!(),
            Type::EnumVariant { .. } => todo!(),
            Type::Unset => todo!(),
            Type::Unevaluated(..) => todo!(),
        }
    }

    pub(crate) fn slice_fields(elem_ty: Ptr<Type>) -> [VarDecl; 2] {
        [
            VarDecl::new_basic(Ident::from("ptr"), Type::Ptr { pointee_ty: elem_ty }),
            VarDecl::new_basic(Ident::from("len"), Type::U64),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(unused)]
enum SemaType {
    RuntimeType(Type),
    IntLiteral,
    FloatLiteral,
    /// ```mylang
    /// MyEnum :: enum { A, B(i64) };
    /// a := MyEnum.A; // variant -> valid val
    /// b1 := MyEnum.B; // variant -> invalid val
    /// b2 := MyEnum.B(5); // variant initialization
    /// ```
    EnumVariant {
        enum_ty: Ptr<Type>,
        idx: usize,
    },

    /// The type was not explicitly set in the original source code and must
    /// still be inferred.
    Unset,
    /// The type was explicitly set in the original source code, but hasn't been
    /// analyzed yet.
    Unevaluated(Ptr<Expr>),
}

#[test]
fn test_sema_type_transmute_to_type() {
    fn t(sema_type: SemaType) -> Type {
        unsafe { std::mem::transmute::<SemaType, Type>(sema_type) }
    }

    let alloc = bumpalo::Bump::new();
    let void_ty_ptr = Ptr::from(alloc.alloc(Type::Void));
    let void_ptr = Type::Ptr { pointee_ty: void_ty_ptr };
    let bool_ty_ptr = Ptr::from(alloc.alloc(Type::Bool));
    let bool_ptr = Type::Ptr { pointee_ty: bool_ty_ptr };
    let i64_ty = Type::Int { bits: 64, is_signed: true };
    let i32_ty = Type::Int { bits: 32, is_signed: true };
    let u64_ty = Type::Int { bits: 64, is_signed: false };
    let f64_ty = Type::Float { bits: 64 };
    let f32_ty = Type::Float { bits: 32 };

    assert_eq!(t(SemaType::RuntimeType(Type::Void)), Type::Void);
    assert_eq!(t(SemaType::RuntimeType(Type::Never)), Type::Never);

    assert_eq!(t(SemaType::RuntimeType(void_ptr)), void_ptr);
    assert_ne!(t(SemaType::RuntimeType(bool_ptr)), void_ptr);
    assert_ne!(t(SemaType::RuntimeType(void_ptr)), bool_ptr);

    assert_eq!(t(SemaType::RuntimeType(i64_ty)), i64_ty);
    assert_ne!(t(SemaType::RuntimeType(i64_ty)), u64_ty);
    assert_ne!(t(SemaType::RuntimeType(i64_ty)), i32_ty);

    assert_eq!(t(SemaType::RuntimeType(Type::Bool)), Type::Bool);

    assert_eq!(t(SemaType::RuntimeType(f64_ty)), f64_ty);
    assert_ne!(t(SemaType::RuntimeType(f64_ty)), f32_ty);

    /*
    Function(Ptr<Fn>),
    Array {
        len: usize,
        elem_ty: Ptr<Type>,
    },
    Struct {
        fields: VarDeclList,
    },
    Union {
        fields: VarDeclList,
    },
    Enum {
        variants: VarDeclList, // TODO
    },
    */

    assert_eq!(t(SemaType::RuntimeType(Type::Type(void_ty_ptr))), Type::Type(void_ty_ptr));
    assert_ne!(t(SemaType::RuntimeType(Type::Type(void_ty_ptr))), Type::Type(bool_ty_ptr));
}
