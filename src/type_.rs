use crate::{
    ast::{Expr, Fn, VarDeclList, debug::DebugAst},
    ptr::Ptr,
    util::{
        aligned_add, panic_debug, round_up_to_nearest_power_of_two, variant_count_to_tag_size_bits,
        variant_count_to_tag_size_bytes,
    },
};
use core::fmt;
use std::cell::OnceCell;

#[allow(unused)]
fn void(alloc: &bumpalo::Bump) -> Ptr<Type> {
    thread_local!(static PTR: OnceCell<Ptr<Type>> = OnceCell::new());
    PTR.with(|cell| *cell.get_or_init(|| Ptr::from(alloc.alloc(Type::Void))))
}

// TODO: benchmark this
// pub type Type = Ptr<TypeInfo>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Void,
    Never,
    Ptr {
        pointee_ty: Ptr<Type>,
    },
    Int {
        bits: u32,
        is_signed: bool,
    },
    Bool,
    Float {
        bits: u32,
    },

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

    /// `a :: int`
    /// -> type of `a`: [`Type::Type`]
    /// -> value of `a`: [`Type::Int`]
    Type(Ptr<Type>),

    // --------------------------------
    // compiletime only types:
    //
    IntLiteral,
    FloatLiteral,

    /// ```
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
            Type::Never => write!(f, "Never"),
            Type::Ptr { pointee_ty } => write!(f, "*{:?}", &**pointee_ty),
            Type::Function(arg0) => f.debug_tuple("Function").field(arg0).finish(),
            Type::Struct { fields } | Type::Union { fields } => write!(
                f,
                "{}{{{}}}",
                if matches!(self, Type::Struct { .. }) { "struct" } else { "union" },
                fields
                    .iter()
                    .map(|f| format!("{}:{:?}", &*f.ident.text, f.ty))
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            Type::Enum { variants } => write!(
                f,
                "enum {{{}}}",
                variants
                    .iter()
                    .map(|v| format!(
                        "{}{}",
                        &*v.ident.text,
                        if v.ty != Type::Void { format!("({})", v.ty) } else { String::default() }
                    ))
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            Type::Unset => write!(f, "Unset"),
            Type::Unevaluated(arg0) => f.debug_tuple("Unevaluated").field(arg0).finish(),
            ti => write!(f, "{}", ti.to_text()),
        }
    }
}

impl Type {
    /// size of stack allocation in bytes
    pub fn size(&self) -> usize {
        match self {
            Type::Void | Type::Never => 0,
            Type::Ptr { .. } => 8,
            Type::Int { bits, .. } | Type::Float { bits } => {
                round_up_to_nearest_power_of_two(*bits as usize).div_ceil(8)
            },
            Type::Bool => 1,
            Type::Function(_) => todo!(),
            Type::Array { len, elem_ty } => elem_ty.size() * len,
            Type::Struct { fields } => fields.iter().map(|f| &f.ty).fold(0, aligned_add),
            Type::Union { fields } => fields.iter().map(|f| f.ty.size()).max().unwrap_or(0),
            Type::Enum { variants } => aligned_add(
                variant_count_to_tag_size_bytes(variants.len()) as usize,
                &Type::Union { fields: *variants },
            ),
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
            Type::Ptr { .. } => 8,
            Type::Int { .. } => self.size().min(16),
            Type::Bool => 1,
            Type::Float { .. } => self.size().min(16),
            Type::Function(_) => todo!(),
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

        fn inner(s: Type, other: Type) -> Option<Selection> {
            match (s, other) {
                (t, v) if t == v => Some(Left),
                (Type::Never, _) => Some(Right),
                (_, Type::Never) => Some(Left),
                (Type::Array { len: l1, elem_ty: t1 }, Type::Array { len: l2, elem_ty: t2 })
                    if l1 == l2 =>
                {
                    inner(*t1, *t2)
                },
                (Type::Ptr { .. }, Type::Ptr { .. }) => Some(Left), // TODO: remove this
                // (Type::Ptr { pointee_ty: p1 }, Type::Ptr { pointee_ty: p2 }) => inner(*p1, *p2),
                (Type::Int { .. } | Type::Float { .. } | Type::FloatLiteral, Type::IntLiteral) => {
                    Some(Left)
                },
                (Type::IntLiteral, Type::Int { .. } | Type::Float { .. } | Type::FloatLiteral) => {
                    Some(Right)
                },
                (Type::Float { .. }, Type::FloatLiteral) => Some(Left),
                (Type::FloatLiteral, Type::Float { .. }) => Some(Right),
                (Type::EnumVariant { enum_ty, idx }, e @ Type::Enum { variants })
                    if *enum_ty == e && variants[idx].ty == Type::Void =>
                {
                    Some(Right)
                },
                (e @ Type::Enum { variants }, Type::EnumVariant { enum_ty, idx })
                    if *enum_ty == e && variants[idx].ty == Type::Void =>
                {
                    Some(Left)
                },
                _ => None,
            }
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
            Type::IntLiteral => Type::Int { bits: 64, is_signed: true },
            Type::FloatLiteral => Type::Float { bits: 64 },
            Type::Array { mut elem_ty, .. } => {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(unused)]
enum SemaType {
    RuntimeType(Type),
    IntLiteral,
    FloatLiteral,
    /// ```
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
