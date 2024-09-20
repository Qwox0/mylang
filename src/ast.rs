use crate::{
    parser::{lexer::Span, DebugAst},
    ptr::Ptr,
};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum ExprKind {
    Ident(Ptr<str>),
    Literal {
        kind: LitKind,
        code: Ptr<str>,
    },
    /// `true`, `false`
    BoolLit(bool),

    /// `[<val>; <count>]`
    /// both for types and literals
    ArraySemi {
        val: Ptr<Expr>,
        count: Ptr<Expr>,
    },
    /// `[<expr>, <expr>, ..., <expr>,]`
    ArrayComma {
        elements: Ptr<[Expr]>,
    },
    /// `(<expr>, <expr>, ..., <expr>,)`
    /// both for types and literals
    Tuple {
        elements: Ptr<[Expr]>,
    },
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <type> { <body> }`
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <body>`
    /// `-> <type> { <body> }`
    /// `-> <body>`
    /// `^ expr.span`
    Fn(Fn),
    /// `( <expr> )`
    Parenthesis {
        expr: Ptr<Expr>,
    },
    /// `{ <stmt>`*` }`
    Block {
        stmts: Ptr<[Ptr<Expr>]>,
        has_trailing_semicolon: bool,
    },

    /// `struct { a: int, b: String, c: (u8, u32) }`
    StructDef(Ptr<[VarDecl]>),
    /// `union { a: int, b: String, c: (u8, u32) }`
    UnionDef(Ptr<[VarDecl]>),
    /// `enum { ... }`
    EnumDef {},
    /// `?<ty>`
    OptionShort(Type),
    /// `*<ty>`
    /// `*mut <ty>`
    Ptr {
        is_mut: bool,
        ty: Type,
    },

    /// `alloc(MyStruct).{ a = <expr>, b, }`
    Initializer {
        lhs: Option<Ptr<Expr>>,
        fields: Ptr<[(Ident, Option<Ptr<Expr>>)]>,
    },

    /// [`expr`] . [`expr`]
    Dot {
        lhs: Ptr<Expr>,
        rhs: Ident,
    },
    /// examples: `<expr>?`, `<expr>.*`
    PostOp {
        expr: Ptr<Expr>,
        kind: PostOpKind,
    },
    /// `<lhs> [ <idx> ]`
    Index {
        lhs: Ptr<Expr>,
        idx: Ptr<Expr>,
    },

    /*
    /// `<func> < <params> >`
    CompCall {
        func: Ptr<Expr>,
        args: Vec<Expr>,
    },
    */
    /// `<func> ( <expr>, ..., param=<expr>, ... )`
    /// `                                        ^ expr.span`
    Call {
        func: Ptr<Expr>,
        args: Ptr<[Ptr<Expr>]>,
    },

    /// examples: `&<expr>`, `- <expr>`
    /// `          ^`        `^ expr.span`
    PreOp {
        kind: PreOpKind,
        expr: Ptr<Expr>,
    },
    /// `<lhs> op <lhs>`
    /// `      ^^ expr.span`
    BinOp {
        lhs: Ptr<Expr>,
        op: BinOpKind,
        rhs: Ptr<Expr>,
    },
    /// `<lhs> = <lhs>`
    Assign {
        //lhs: Ptr<LValue>,
        lhs: Ptr<Expr>,
        rhs: Ptr<Expr>,
    },
    /// `<lhs> op= <lhs>`
    BinOpAssign {
        //lhs: Ptr<LValue>,
        lhs: Ptr<Expr>,
        op: BinOpKind,
        rhs: Ptr<Expr>,
    },

    /// variable declaration (and optional initialization)
    /// `mut rec <name>: <ty>`
    /// `mut rec <name>: <ty> = <init>`
    /// `mut rec <name>: <ty> : <init>`
    /// `mut rec <name> := <init>`
    /// `mut rec <name> :: <init>`
    /// `expr.span` must describe the entire expression if `default.is_none()`,
    /// otherwise only the start is important
    VarDecl(VarDecl),

    // /// `pub extern my_fn: (a: i32, b: f64) -> bool`
    // ExternDecl {
    //     is_pub: bool,
    //     ident: Ptr<Expr>,
    //     ty: Ptr<Expr>,
    // },
    /// `if <cond> <then>` (`else <else>`)
    /// `^^` expr.span
    If {
        condition: Ptr<Expr>,
        then_body: Ptr<Expr>,
        else_body: Option<Ptr<Expr>>,
    },
    /// `match <val> <body>` (`else <else>`)
    Match {
        val: Ptr<Expr>,
        // TODO
        else_body: Option<Ptr<Expr>>,
    },

    /// TODO: normal syntax
    /// `<source> | for <iter_var> <body>`
    For {
        source: Ptr<Expr>,
        iter_var: Ident,
        body: Ptr<Expr>,
    },
    /// `while <cond> <body>`
    While {
        condition: Ptr<Expr>,
        body: Ptr<Expr>,
    },

    /// `lhs catch ...`
    Catch {
        lhs: Ptr<Expr>,
        // TODO
    },

    /// `lhs | rhs`
    /// Note: `lhs | if ...`, `lhs | match ...`, `lhs | for ...` and
    /// `lhs | while ...` are inlined during parsing
    Pipe {
        lhs: Ptr<Expr>,
        // TODO
    },

    Defer(Ptr<Expr>),

    /// `return <expr>`
    /// `^^^^^^` expr.span
    Return {
        expr: Option<Ptr<Expr>>,
    },

    Semicolon(Option<Ptr<Expr>>),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
    pub ty: Type,
}

impl From<(ExprKind, Span)> for Expr {
    #[inline]
    fn from((kind, span): (ExprKind, Span)) -> Self {
        Expr { kind, span, ty: Type::Unset }
    }
}

impl Expr {
    #[inline]
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span, ty: Type::Unset }
    }

    /// Returns a [`Span`] representing the entire expression.
    pub fn full_span(&self) -> Span {
        #[allow(unused_variables)]
        match self.kind {
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn(Fn { params, ret_type, body }) => self.span.join(body.full_span()),
            ExprKind::StructDef(_) => todo!(),
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs, fields } => todo!(),
            ExprKind::Dot { lhs, rhs } => todo!(),
            ExprKind::PostOp { expr, kind } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            ExprKind::Call { func, args } => func.full_span().join(self.span),
            ExprKind::PreOp { kind: _, expr } => self.span.join(expr.full_span()),
            ExprKind::BinOp { lhs, op: _, rhs }
            | ExprKind::Assign { lhs, rhs }
            | ExprKind::BinOpAssign { lhs, op: _, rhs } => lhs.full_span().join(rhs.full_span()),
            ExprKind::VarDecl(decl) => match &decl.default {
                Some(e) => self.span.join(e.full_span()),
                None => self.span,
            },
            ExprKind::If { condition, then_body, else_body } => {
                self.span.join(else_body.unwrap_or(then_body).full_span())
            },
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => todo!(),
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Return { expr } => match expr {
                Some(expr) => self.span.join(expr.full_span()),
                None => self.span,
            },
            ExprKind::Semicolon(_) => todo!(),
            _ => self.span,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    /// `*`, `*=`
    Mul,
    /// `/`, `/=`
    Div,
    /// `%`, `%=`
    Mod,

    /// `+`, `+=`
    Add,
    /// `-`, `-=`
    Sub,

    /// `<<`, `<<=`
    ShiftL,
    /// `>>`, `>>=`
    ShiftR,

    /// `&`, `&=`
    BitAnd,

    /// `^`, `^=`
    BitXor,

    /// TODO: find a solution for pipe vs bitor (currently bitand, bitxor and
    /// bitor are ignored)
    /// `|`, `|=`
    BitOr,

    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,

    /// `&&`
    And,

    /// `||`
    Or,

    /// `..`
    Range,
    /// `..=`
    RangeInclusive,
}

impl BinOpKind {
    pub fn to_binop_text(self) -> &'static str {
        match self {
            BinOpKind::Mul => "*",
            BinOpKind::Div => "/",
            BinOpKind::Mod => "%",
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::ShiftL => "<<",
            BinOpKind::ShiftR => ">>",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitXor => "^",
            BinOpKind::BitOr => "|",
            BinOpKind::Eq => "==",
            BinOpKind::Ne => "!=",
            BinOpKind::Lt => "<",
            BinOpKind::Le => "<=",
            BinOpKind::Gt => ">",
            BinOpKind::Ge => ">=",
            BinOpKind::And => "&&",
            BinOpKind::Or => "||",
            BinOpKind::Range => "..",
            BinOpKind::RangeInclusive => "..=",
        }
    }

    pub fn to_binop_assign_text(self) -> &'static str {
        match self {
            BinOpKind::Mul => "*=",
            BinOpKind::Div => "/=",
            BinOpKind::Mod => "%=",
            BinOpKind::Add => "+=",
            BinOpKind::Sub => "-=",
            BinOpKind::ShiftL => "<<=",
            BinOpKind::ShiftR => ">>=",
            BinOpKind::BitAnd => "&=",
            BinOpKind::BitXor => "^=",
            BinOpKind::BitOr => "|=",
            k => panic!("Unexpected binop kind: {:?}", k),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PostOpKind {
    /// `<expr>.&`
    AddrOf,
    /// `<expr>.&mut`
    AddrMutOf,
    /// `<expr>.*`
    Deref,
    /// `<expr>?`
    Try,
    /// `<expr>!`
    Force,
    /// `<expr>!unsafe`
    ForceUnsafe,
    // /// `<expr>.type`
    // TypeOf,
}

#[derive(Debug, Clone, Copy)]
pub struct Fn {
    pub params: Ptr<[VarDecl]>,
    pub ret_type: Type,
    pub body: Ptr<Expr>,
}

#[derive(Debug, Clone, Copy)]
pub struct VarDecl {
    pub markers: DeclMarkers,
    pub ident: Ident,
    pub ty: Type,
    /// * default value for fn params, struct fields, ...
    /// * init for local veriable declarations
    pub default: Option<Ptr<Expr>>,
    pub is_const: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DeclMarkers {
    pub is_pub: bool,
    pub is_mut: bool,
    pub is_rec: bool,
}

impl DeclMarkers {
    pub fn is_empty(&self) -> bool {
        !(self.is_pub || self.is_mut || self.is_rec)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclMarkerKind {
    Pub,
    Mut,
    Rec,
}

/*
#[derive(Debug, Clone, Copy)]
pub enum VarDeclKind {
    /// `<name>: <ty>;`
    /// `<name>: <ty> = <init>;`
    WithTy { ty: Ptr<Expr>, init: Option<Ptr<Expr>> },
    /// `<name> := <init>;`
    InferTy { init: Ptr<Expr> },
}

impl VarDeclKind {
    pub fn get_init(&self) -> Option<&Ptr<Expr>> {
        match self {
            VarDeclKind::WithTy { init, .. } => init.as_ref(),
            VarDeclKind::InferTy { init } => Some(init),
        }
    }
}
*/

#[derive(Debug, Clone, Copy)]
pub struct Ident {
    pub(super) text: Ptr<str>,
    pub span: Span,
}

impl Ident {
    pub fn into_expr(self) -> Expr {
        Expr::new(ExprKind::Ident(self.text), self.span)
    }
}

#[allow(unused)]
pub struct Pattern {
    kind: ExprKind, // TODO: own kind enum
    span: Span,
}

#[derive(Debug, Clone, Copy)]
pub enum PreOpKind {
    /// `& <expr>`
    AddrOf,
    /// `&mut <expr>`
    AddrMutOf,
    /// `* <expr>`
    Deref,
    /// `! <expr>`
    Not,
    /// `- <expr>`
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LitKind {
    /// `'a'`
    Char,
    /// `b'a'`
    BChar,
    /// `1`, `-10`, `10_000`
    /// allowed but not recommended: `1_0_0_0`
    Int,
    /// `1.5`
    Float,
    /// `"literal"`
    Str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Void,
    Never,
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
    Custom(Ptr<str>),

    //Literal(LitKind),
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
            Type::Void => write!(f, "Void"),
            Type::Never => write!(f, "Never"),
            Type::Int { bits, is_signed } => {
                write!(f, "{}{}", if *is_signed { "i" } else { "u" }, bits)
            },
            Type::IntLiteral => write!(f, "int lit"),
            Type::Bool => write!(f, "bool"),
            Type::Float { bits } => write!(f, "f{}", bits),
            Type::FloatLiteral => write!(f, "float lit"),
            Type::Function(arg0) => f.debug_tuple("Function").field(arg0).finish(),
            // Type::Literal(kind) => write!(f, "{:?}Lit", kind),
            Type::Custom(name) => f.debug_tuple("Custom").field(&&**name).finish(),
            Type::Unset => write!(f, "Unset"),
            Type::Unevaluated(arg0) => f.debug_tuple("Unevaluated").field(arg0).finish(),
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
        if self.is_valid() { Some(self) } else { None }
    }

    /// Checks if the two types equal or can be coerced into a common type.
    ///
    /// For exact equality use `==`.
    pub fn matches(&self, rhs_ty: &Type) -> bool {
        // TODO: more type coercion
        self == rhs_ty || self == &Type::Never || rhs_ty == &Type::Never
    }

    /// Returns the common type after type coercion or [`None`] if the types
    /// don't match (see [`Type::matches`]).
    pub fn common_type(self, rhs_ty: Type) -> Option<Type> {
        if self == rhs_ty || rhs_ty == Type::Never {
            Some(self)
        } else if self == Type::Never {
            Some(rhs_ty)
        } else {
            None
        }
    }
}
