use crate::{
    ast::{debug::DebugAst, Expr, Fn, VarDeclList},
    ptr::Ptr,
};
use core::fmt;

// TODO: benchmark this
// pub type Type = Ptr<TypeInfo>;
pub type Type = TypeInfo;

#[derive(Clone, Copy, Eq)]
pub enum TypeInfo {
    Void,
    Never,
    Ptr(Ptr<TypeInfo>),
    Int {
        bits: u32,
        is_signed: bool,
    },
    IntLiteral,
    Bool,
    Float {
        bits: u8,
    },
    FloatLiteral,

    Function(Ptr<Fn>),

    Array {
        len: usize,
        elem_ty: Ptr<TypeInfo>,
    },

    Struct {
        fields: VarDeclList,
    },
    Union {
        fields: VarDeclList,
    },
    Enum {
        variants: (), // TODO
    },

    /// `a :: int`
    /// -> type of `a`: [`Type::Type`]
    /// -> value of `a`: [`Type::Int`]
    Type(Ptr<TypeInfo>),

    /// The type was not explicitly set in the original source code and must
    /// still be inferred.
    Unset,
    /// The type was explicitly set in the original source code, but hasn't been
    /// analyzed yet.
    Unevaluated(Ptr<Expr>),
}

impl PartialEq for TypeInfo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ptr(l), Self::Ptr(r)) => **l == **r,
            (Self::Int { bits: lb, is_signed: ls }, Self::Int { bits: rb, is_signed: rs }) => {
                lb == rb && ls == rs
            },
            (Self::Float { bits: l }, Self::Float { bits: r }) => l == r,
            (Self::Function(l), Self::Function(r)) => l == r,
            (
                Self::Array { len: l_len, elem_ty: l_elem },
                Self::Array { len: r_len, elem_ty: r_elem },
            ) => l_len == r_len && **l_elem == **r_elem,
            (Self::Struct { fields: l }, Self::Struct { fields: r }) => l == r,
            (Self::Union { fields: l }, Self::Union { fields: r }) => l == r,
            (Self::Enum { variants: l }, Self::Enum { variants: r }) => l == r,
            (Self::Unevaluated(_), Self::Unevaluated(_)) => false,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl fmt::Debug for TypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeInfo::Never => write!(f, "Never"),
            TypeInfo::Ptr(pointee) => write!(f, "*{:?}", &**pointee),
            TypeInfo::Function(arg0) => f.debug_tuple("Function").field(arg0).finish(),
            TypeInfo::Struct { fields } => write!(
                f,
                "struct{{{}}}",
                fields
                    .iter()
                    .map(|f| format!("{}:{:?}", &*f.ident.text, f.ty,))
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            TypeInfo::Union { .. } => write!(f, "union {{ ... }}"),
            TypeInfo::Enum { .. } => write!(f, "enum {{ ... }}"),
            TypeInfo::Unset => write!(f, "Unset"),
            TypeInfo::Unevaluated(arg0) => f.debug_tuple("Unevaluated").field(arg0).finish(),
            ti => write!(f, "{}", ti.to_text()),
        }
    }
}

impl TypeInfo {
    /// Checks if the two types equal or can be coerced into a common type.
    ///
    /// For exact equality use `==`.
    pub fn matches(self, rhs_ty: TypeInfo) -> bool {
        // TODO: more type coercion
        self == rhs_ty || self == TypeInfo::Never || rhs_ty == TypeInfo::Never
    }

    /// Returns the common type after type coercion or [`None`] if the types
    /// don't match (see [`TypeInfo::matches`]).
    pub fn common_type(self, rhs_ty: TypeInfo) -> Option<TypeInfo> {
        if self == rhs_ty || rhs_ty == TypeInfo::Never {
            Some(self)
        } else if self == TypeInfo::Never {
            Some(rhs_ty)
        } else {
            None
        }
    }

    pub fn assign_matches(target: TypeInfo, val: TypeInfo) -> bool {
        //target == val || val == TypeInfo::Never || (val == TypeInfo::In)
        match (target, val) {
            (t, v) if t == v => true,
            (_, TypeInfo::Never) => true,
            (TypeInfo::Int { .. } | TypeInfo::Float { .. }, Type::IntLiteral) => true,
            (TypeInfo::Float { .. }, TypeInfo::FloatLiteral) => true,
            (
                TypeInfo::Array { len: l1, elem_ty: t1 },
                TypeInfo::Array { len: l2, elem_ty: t2 },
            ) => l1 == l2 && TypeInfo::assign_matches(*t1, *t2),
            _ => false,
        }
    }
}

impl Type {
    #[inline]
    pub fn is_valid(&self) -> bool {
        match self {
            TypeInfo::Unset | TypeInfo::Unevaluated(_) => false,
            _ => true,
        }
    }

    /// if `self` [Type::is_valid] this returns `Some(self)`, otherwise [`None`]
    #[inline]
    pub fn into_valid(self) -> Option<TypeInfo> {
        match self {
            TypeInfo::Unset | TypeInfo::Unevaluated(_) => None,
            t => Some(t),
        }
    }
}

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum MaybeType {
//     Ok(TypeInfo),
//     /// The type was not explicitly set in the original source code and must
//     /// still be inferred.
//     Unset,
//     /// The type was explicitly set in the original source code, but hasn't
// been     /// analyzed yet.
//     Unevaluated(Ptr<Expr>),
// }
//
// impl MaybeType {
//     #[inline]
//     pub fn is_valid(&self) -> bool {
//         matches!(self, MaybeType::Ok(_))
//     }
//
//     /// if `self` [Type::is_valid] this returns `Some(self)`, otherwise
// [`None`]     #[inline]
//     pub fn into_valid(self) -> Option<TypeInfo> {
//         match self {
//             MaybeType::Ok(ty) => Some(ty),
//             _ => None,
//         }
//     }
// }
//
// impl UnwrapDebug for MaybeType {
//     type Inner = TypeInfo;
//
//     #[inline]
//     fn unwrap_debug(self) -> Self::Inner {
//         match self {
//             MaybeType::Ok(ty) => ty,
//             _ => {
//                 if cfg!(debug_assertions) {
//                     panic!("called unwrap_debug on an invalid MaybeType
// variant")                 } else {
//                     unsafe { unreachable_unchecked() }
//                 }
//             },
//         }
//     }
// }
