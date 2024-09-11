use crate::parser::lexer::Span;
use std::ptr::NonNull;

#[derive(Debug, Clone, Copy)]
pub enum ExprKind {
    Ident(NonNull<str>),
    Literal {
        kind: LitKind,
        code: NonNull<str>,
    },
    /// `true`, `false`
    BoolLit(bool),

    /// `[<val>; <count>]`
    /// both for types and literals
    ArraySemi {
        val: NonNull<Expr>,
        count: NonNull<Expr>,
    },
    /// `[<expr>, <expr>, ..., <expr>,]`
    ArrayComma {
        elements: NonNull<[Expr]>,
    },
    /// `(<expr>, <expr>, ..., <expr>,)`
    /// both for types and literals
    Tuple {
        elements: NonNull<[Expr]>,
    },
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <type> { <body> }`
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <body>`
    /// `-> <type> { <body> }`
    /// `-> <body>`
    Fn {
        params: NonNull<[VarDecl]>,
        ret_type: Option<Type>,
        body: NonNull<Expr>,
    },
    /// `( <expr> )`
    Parenthesis {
        expr: NonNull<Expr>,
    },
    /// `{ <stmt>`*` }`
    Block {
        stmts: NonNull<[NonNull<Expr>]>,
        has_trailing_semicolon: bool,
    },

    /// `struct { a: int, b: String, c: (u8, u32) }`
    StructDef(NonNull<[VarDecl]>),
    /// `union { a: int, b: String, c: (u8, u32) }`
    UnionDef(NonNull<[VarDecl]>),
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
        lhs: Option<NonNull<Expr>>,
        fields: NonNull<[(Ident, Option<NonNull<Expr>>)]>,
    },

    /// [`expr`] . [`expr`]
    Dot {
        lhs: NonNull<Expr>,
        rhs: Ident,
    },
    /// examples: `<expr>?`, `<expr>.*`
    PostOp {
        expr: NonNull<Expr>,
        kind: PostOpKind,
    },
    /// `<lhs> [ <idx> ]`
    Index {
        lhs: NonNull<Expr>,
        idx: NonNull<Expr>,
    },

    /*
    /// `<func> < <params> >`
    CompCall {
        func: NonNull<Expr>,
        args: Vec<Expr>,
    },
    */
    /// [`colon`] `(` [`comma_chain`] ([`expr`]) `)`
    Call {
        func: NonNull<Expr>,
        args: NonNull<[NonNull<Expr>]>,
    },

    /// examples: `&<expr>`, `- <expr>`
    PreOp {
        kind: PreOpKind,
        expr: NonNull<Expr>,
    },
    /// `<lhs> op <lhs>`
    BinOp {
        lhs: NonNull<Expr>,
        op: BinOpKind,
        rhs: NonNull<Expr>,
    },
    /// `<lhs> = <lhs>`
    Assign {
        //lhs: NonNull<LValue>,
        lhs: NonNull<Expr>,
        rhs: NonNull<Expr>,
    },
    /// `<lhs> op= <lhs>`
    BinOpAssign {
        //lhs: NonNull<LValue>,
        lhs: NonNull<Expr>,
        op: BinOpKind,
        rhs: NonNull<Expr>,
    },

    /// variable declaration (and optional initialization)
    /// `mut rec <name>: <ty>`
    /// `mut rec <name>: <ty> = <init>`
    /// `mut rec <name>: <ty> : <init>`
    /// `mut rec <name> := <init>`
    /// `mut rec <name> :: <init>`
    VarDecl(VarDecl),

    // /// `pub extern my_fn: (a: i32, b: f64) -> bool`
    // ExternDecl {
    //     is_pub: bool,
    //     ident: NonNull<Expr>,
    //     ty: NonNull<Expr>,
    // },
    /// `if <cond> <then>` (`else <else>`)
    If {
        condition: NonNull<Expr>,
        then_body: NonNull<Expr>,
        else_body: Option<NonNull<Expr>>,
    },
    /// `match <val> <body>` (`else <else>`)
    Match {
        val: NonNull<Expr>,
        // TODO
        else_body: Option<NonNull<Expr>>,
    },

    /// TODO: normal syntax
    /// `<source> | for <iter_var> <body>`
    For {
        source: NonNull<Expr>,
        iter_var: Ident,
        body: NonNull<Expr>,
    },
    /// `while <cond> <body>`
    While {
        condition: NonNull<Expr>,
        body: NonNull<Expr>,
    },

    /// `lhs catch ...`
    Catch {
        lhs: NonNull<Expr>,
        // TODO
    },

    /// `lhs | rhs`
    /// Note: `lhs | if ...`, `lhs | match ...`, `lhs | for ...` and
    /// `lhs | while ...` are inlined during parsing
    Pipe {
        lhs: NonNull<Expr>,
        // TODO
    },

    Return {
        expr: Option<NonNull<Expr>>,
    },

    Semicolon(Option<NonNull<Expr>>),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl From<(ExprKind, Span)> for Expr {
    fn from((kind, span): (ExprKind, Span)) -> Self {
        Expr { kind, span }
    }
}

impl Expr {
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
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
pub struct VarDecl {
    pub markers: DeclMarkers,
    pub ident: Ident,
    pub ty: Option<Type>,
    /// * default value for fn params, struct fields, ...
    /// * init for local veriable declarations
    pub default: Option<NonNull<Expr>>,
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
    WithTy { ty: NonNull<Expr>, init: Option<NonNull<Expr>> },
    /// `<name> := <init>;`
    InferTy { init: NonNull<Expr> },
}

impl VarDeclKind {
    pub fn get_init(&self) -> Option<&NonNull<Expr>> {
        match self {
            VarDeclKind::WithTy { init, .. } => init.as_ref(),
            VarDeclKind::InferTy { init } => Some(init),
        }
    }
}
*/

#[derive(Debug, Clone, Copy)]
pub struct Ident {
    pub(super) text: NonNull<str>,
    pub span: Span,
}

impl Ident {
    pub fn into_expr(self) -> Expr {
        Expr { kind: ExprKind::Ident(self.text), span: self.span }
    }

    pub fn get_text(&self) -> &str {
        unsafe { self.text.as_ref() }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Void,
    Never,
    Float { bits: u8 },
    Function(NonNull<Expr>),

    Unevaluated(NonNull<Expr>),
}

impl Type {
    pub const UNKNOWN: Option<Type> = None;
}

pub trait IsValidType {
    fn is_valid_ty(&self) -> bool;
}

impl IsValidType for Type {
    fn is_valid_ty(&self) -> bool {
        match self {
            Type::Unevaluated(_) => false,
            _ => true,
        }
    }
}

impl IsValidType for Option<Type> {
    fn is_valid_ty(&self) -> bool {
        self.as_ref().is_some_and(Type::is_valid_ty)
    }
}
