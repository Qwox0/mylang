use crate::{
    ast::{Expr, Fn, VarDeclList},
    ptr::Ptr,
};
use core::fmt;

// TODO: benchmark this
// pub type Type = Ptr<TypeInfo>;
pub type Type = TypeInfo;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TypeInfo {
    Void,
    Never,
    Ptr(Ptr<TypeInfo>),
    Int {
        bits: u8,
        is_signed: bool,
    },
    IntLiteral,
    Bool,
    Float {
        bits: u8,
    },
    FloatLiteral,

    Function(Ptr<Fn>),

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

impl fmt::Debug for TypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeInfo::Void => write!(f, "Void"),
            TypeInfo::Never => write!(f, "Never"),
            TypeInfo::Ptr(pointee) => write!(f, "*{:?}", &**pointee),
            TypeInfo::Int { bits, is_signed } => {
                write!(f, "{}{}", if *is_signed { "i" } else { "u" }, bits)
            },
            TypeInfo::IntLiteral => write!(f, "int lit"),
            TypeInfo::Bool => write!(f, "bool"),
            TypeInfo::Float { bits } => write!(f, "f{}", bits),
            TypeInfo::FloatLiteral => write!(f, "float lit"),
            TypeInfo::Function(arg0) => f.debug_tuple("Function").field(arg0).finish(),
            // TypeInfo::Literal(kind) => write!(f, "{:?}Lit", kind),
            //TypeInfo::Custom(name) => f.debug_tuple("Custom").field(&&**name).finish(),
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
            TypeInfo::Type(_) => write!(f, "type"),
            TypeInfo::Unset => write!(f, "Unset"),
            TypeInfo::Unevaluated(arg0) => f.debug_tuple("Unevaluated").field(arg0).finish(),
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
