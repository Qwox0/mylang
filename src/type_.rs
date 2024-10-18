use crate::{
    ast::{Expr, Fn, VarDeclList, debug::DebugAst},
    ptr::Ptr,
    util::panic_debug,
};
use core::fmt;

// TODO: benchmark this
// pub type Type = Ptr<TypeInfo>;
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Void,
    Never,
    Ptr(Ptr<Type>),
    Int {
        bits: u32,
        is_signed: bool,
    },
    IntLiteral,
    Bool,
    Float {
        bits: u32,
    },
    FloatLiteral,

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
        variants: (), // TODO
    },

    /// `a :: int`
    /// -> type of `a`: [`Type::Type`]
    /// -> value of `a`: [`Type::Int`]
    Type(Ptr<Type>),

    /// The type was not explicitly set in the original source code and must
    /// still be inferred.
    Unset,
    /// The type was explicitly set in the original source code, but hasn't been
    /// analyzed yet.
    Unevaluated(Ptr<Expr>),
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Never => write!(f, "Never"),
            Type::Ptr(pointee) => write!(f, "*{:?}", &**pointee),
            Type::Function(arg0) => f.debug_tuple("Function").field(arg0).finish(),
            Type::Struct { fields } => write!(
                f,
                "struct{{{}}}",
                fields
                    .iter()
                    .map(|f| format!("{}:{:?}", &*f.ident.text, f.ty,))
                    .intersperse(",".to_string())
                    .collect::<String>()
            ),
            Type::Union { .. } => write!(f, "union {{ ... }}"),
            Type::Enum { .. } => write!(f, "enum {{ ... }}"),
            Type::Unset => write!(f, "Unset"),
            Type::Unevaluated(arg0) => f.debug_tuple("Unevaluated").field(arg0).finish(),
            ti => write!(f, "{}", ti.to_text()),
        }
    }
}

impl Type {
    pub fn stack_size(&self) -> usize {
        match self {
            Type::Void => 0,
            Type::Never => 0,
            Type::Ptr(_) => 8,
            Type::Int { bits, .. } | Type::Float { bits } => (*bits as usize).div_ceil(8),
            Type::Bool => 1,
            Type::Function(_) => todo!(),
            Type::Array { len, elem_ty } => elem_ty.stack_size() * len,
            Type::Struct { fields } => fields.iter().map(|f| f.ty.stack_size()).max().unwrap_or(0),
            // TODO: struct might contain padding
            Type::Union { fields } => fields.iter().map(|f| f.ty.stack_size()).sum(),
            Type::Enum { .. } => todo!(),
            Type::Type(_) => 0,
            _ => panic_debug("cannot find stack size"),
        }
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
                (Type::Int { .. } | Type::Float { .. } | Type::FloatLiteral, Type::IntLiteral) => {
                    Some(Left)
                },
                (Type::IntLiteral, Type::Int { .. } | Type::Float { .. } | Type::FloatLiteral) => {
                    Some(Right)
                },
                (Type::Float { .. }, Type::FloatLiteral) => Some(Left),
                (Type::FloatLiteral, Type::Float { .. }) => Some(Right),
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
            Type::Unset | Type::Unevaluated(_) => panic_debug("cannot finalize invalid type"),
            t => t,
        }
    }
}

impl Type {
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

/*
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaybeType {
    Valid(TypeInfo),

    IntLiteral,
    FloatLiteral,

    /// The type was not explicitly set in the original source code and must
    /// still be inferred.
    Unset,
    /// The type was explicitly set in the original source code, but hasn't been
    /// analyzed yet.
    Unevaluated(Ptr<Expr>),
}

impl MaybeType {
    #[inline]
    pub fn is_valid(&self) -> bool {
        matches!(self, MaybeType::Valid(_))
    }

    /// if `self` [Type::is_valid] this returns `Some(self)`, otherwise [`None`]
    #[inline]
    pub fn into_valid(self) -> Option<TypeInfo> {
        match self {
            MaybeType::Valid(ty) => Some(ty),
            _ => None,
        }
    }

    // a: f64 = 1 + 1.0 + 1
    //          ^           int_lit
    //              ^^^     float_lit
    //          ^^^^^^^     float_lit
    //                    ^ int_lit
    //          ^^^^^^^^^^^ float_lit
    //          ^^^^^^^^^^^ f64

    /// expects both types to be valid
    pub fn common_type_valid(self, other: MaybeType) -> Option<MaybeType> {
        match (self, other) {
            (MaybeType::Valid(a), MaybeType::Valid(b)) => a.common_type(b).map(MaybeType::Valid),
            (MaybeType::Valid(t), MaybeType::IntLiteral)
            | (MaybeType::IntLiteral, MaybeType::Valid(t)) => match t {
                TypeInfo::Never | TypeInfo::Int { .. } | TypeInfo::Float { .. } => {
                    Some(MaybeType::Valid(t))
                },
                _ => None,
            },
            (MaybeType::Valid(t), MaybeType::FloatLiteral)
            | (MaybeType::FloatLiteral, MaybeType::Valid(t)) => match t {
                TypeInfo::Never | TypeInfo::Float { .. } => Some(MaybeType::Valid(t)),
                _ => None,
            },
            (MaybeType::IntLiteral, MaybeType::IntLiteral) => Some(MaybeType::IntLiteral),
            (
                MaybeType::IntLiteral | MaybeType::FloatLiteral,
                MaybeType::IntLiteral | MaybeType::FloatLiteral,
            ) => Some(MaybeType::FloatLiteral),
            (MaybeType::Unset | MaybeType::Unevaluated(_), _)
            | (_, MaybeType::Unset | MaybeType::Unevaluated(_)) => {
                panic_debug("common_type_valid expects valid types")
            },
        }
    }

    /*
    /// expects both types to be valid
    pub fn common_type_valid(self, other: MaybeType) -> Option<MaybeType> {
        match (self, other) {
            (MaybeType::Valid(a), MaybeType::Valid(b)) => a.common_type(b).map(MaybeType::Valid),
            (MaybeType::Valid(t), MaybeType::NumLiteral { kind, real_ty })
            | (MaybeType::NumLiteral { kind, real_ty }, MaybeType::Valid(t)) => {
                if let Some(real_ty) = *real_ty {
                    return real_ty.common_type(t).map(MaybeType::Valid);
                }
                let matches = match kind {
                    NumLiteralKind::Int => {
                        matches!(t, TypeInfo::Int { .. } | TypeInfo::Float { .. })
                    },
                    NumLiteralKind::Float => matches!(t, TypeInfo::Float { .. }),
                };
                if matches {
                    *real_ty.as_mut() = Some(t);
                }
                Some(t)
            },
            (
                MaybeType::NumLiteral { kind: k1, real_ty: t1 },
                MaybeType::NumLiteral { kind: k2, real_ty: t2 },
            ) => match ((k1, *t1), (k2, *t2)) {
                ((_, Some(t1)), (_, Some(t2))) => t1.common_type(t2).map(MaybeType::Valid),
                ((_, Some(t)), (k, None)) | ((k, None), (_, Some(t))) => match (t, k) {
                    (TypeInfo::Void, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Void, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Never, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Never, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Ptr(ptr), NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Ptr(ptr), NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Int { bits, is_signed }, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Int { bits, is_signed }, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Bool, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Bool, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Float { bits }, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Float { bits }, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Function(ptr), NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Function(ptr), NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Array { len, elem_ty }, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Array { len, elem_ty }, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Struct { fields }, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Struct { fields }, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Union { fields }, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Union { fields }, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Enum { variants }, NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Enum { variants }, NumLiteralKind::Float) => todo!(),
                    (TypeInfo::Type(ptr), NumLiteralKind::Int) => todo!(),
                    (TypeInfo::Type(ptr), NumLiteralKind::Float) => todo!(),
                },
                ((k1, None), (k2, None)) => {
                    let is_float = k1 == NumLiteralKind::Float || k2 == NumLiteralKind::Float;
                    let kind = if is_float { NumLiteralKind::Float } else { NumLiteralKind::Int };
                    Some(MaybeType::NumLiteral { kind, real_ty: t1 })
                },
            },
            (MaybeType::Unset | MaybeType::Unevaluated(_), _)
            | (_, MaybeType::Unset | MaybeType::Unevaluated(_)) => {
                panic_debug("common_type_valid expects valid types")
            },
        }
    }
    */
}

impl UnwrapDebug for MaybeType {
    type Inner = TypeInfo;

    #[inline]
    fn unwrap_debug(self) -> Self::Inner {
        match self {
            MaybeType::Valid(ty) => ty,
            _ => panic_debug("called unwrap_debug on an invalid MaybeType variant"),
        }
    }
}
*/
