use crate::{
    parser::lexer::Span,
    ptr::Ptr,
    type_::Type,
    util::{UnwrapDebug, forget_lifetime, unreachable_debug},
};
use debug::DebugAst;
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

pub mod debug;

#[derive(Debug, Clone, Copy)]
pub struct ExprWithTy {
    pub expr: Ptr<Expr>,
    pub ty: Type,
}

impl ExprWithTy {
    pub fn untyped(expr: Ptr<Expr>) -> ExprWithTy {
        ExprWithTy { expr, ty: Type::Unset }
    }
}

impl Deref for ExprWithTy {
    type Target = Expr;

    fn deref(&self) -> &Self::Target {
        &*self.expr
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExprKind {
    Ident(Ptr<str>),
    Literal {
        kind: LitKind,
        code: Ptr<str>,
    },
    /// `true`, `false`
    BoolLit(bool),

    /// `*<ty>`
    /// `*mut <ty>`
    PtrTy {
        is_mut: bool,
        ty: Type,
    },
    /// `[]T` -> `struct { ptr: *T, len: u64 }`
    SliceTy {
        ty: Type,
    },
    /// `[<count>]ty`
    ArrayTy {
        count: Ptr<Expr>,
        ty: Type,
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
        stmts: Ptr<[ExprWithTy]>,
        has_trailing_semicolon: bool,
    },

    /// `struct { a: int, b: String, c: (u8, u32) }`
    StructDef(VarDeclList),
    /// `union { a: int, b: String, c: (u8, u32) }`
    UnionDef(VarDeclList),
    /// `enum { A, B(i64) }`
    EnumDef(VarDeclList),
    /// `?<ty>`
    OptionShort(Type),

    /// `alloc(MyStruct).( a, b, c = <expr>, )`
    /// `               ^^^^^^^^^^^^^^^^^^^^^^` expr.span
    ///
    /// [`Type`] -> value
    /// `*T` -> `*T`
    PositionalInitializer {
        lhs: Option<Ptr<Expr>>,
        lhs_ty: Type,
        args: Ptr<[Ptr<Expr>]>,
    },
    /// `alloc(MyStruct).{ a = <expr>, b, }`
    /// `               ^^^^^^^^^^^^^^^^^^^` expr.span
    ///
    /// [`Type`] -> value
    /// `*T` -> `*T`
    NamedInitializer {
        lhs: Option<Ptr<Expr>>,
        lhs_ty: Type,
        fields: Ptr<[(Ident, Option<Ptr<Expr>>)]>,
    },
    /// `alloc(MyArray).[<expr>, <expr>, ..., <expr>,]`
    ArrayInitializer {
        lhs: Option<Ptr<Expr>>,
        lhs_ty: Type,
        elements: Ptr<[Ptr<Expr>]>,
    },
    /// `alloc(MyArray).[<expr>; <count>]`
    ArrayInitializerShort {
        lhs: Option<Ptr<Expr>>,
        lhs_ty: Type,
        val: Ptr<Expr>,
        count: Ptr<Expr>,
    },

    /// `expr . ident`, `.ident`
    /// `     ^` `       ^` expr.span
    Dot {
        lhs: Option<Ptr<Expr>>,
        lhs_ty: Type,
        rhs: Ident,
    },
    /// `<lhs> [ <idx> ]`
    /// `              ^` expr.span
    Index {
        lhs: ExprWithTy,
        idx: ExprWithTy,
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
        func: ExprWithTy,
        args: Ptr<[Ptr<Expr>]>,
        /// which argument was piped into this [`ExprKind::Call`]
        pipe_idx: Option<usize>,
    },

    /// examples: `&<expr>`, `<expr>.*`, `- <expr>`
    /// `          ^` `             ^^` ` ^ expr.span`
    UnaryOp {
        kind: UnaryOpKind,
        expr: Ptr<Expr>,
        is_postfix: bool,
    },
    /// `<lhs> op <lhs>`
    /// `      ^^ expr.span`
    BinOp {
        lhs: Ptr<Expr>,
        op: BinOpKind,
        rhs: Ptr<Expr>,
        arg_ty: Type,
    },
    /// `<lhs> = <lhs>`
    Assign {
        lhs: ExprWithTy,
        rhs: Ptr<Expr>,
    },
    /// `<lhs> op= <lhs>`
    BinOpAssign {
        lhs: ExprWithTy,
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
    Extern {
        ident: Ident,
        ty: Type,
    },

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
        was_piped: bool,
    },
    /// `match <val> <body>` (`else <else>`)
    Match {
        val: Ptr<Expr>,
        // TODO
        else_body: Option<Ptr<Expr>>,
        was_piped: bool,
    },

    /// TODO: normal syntax
    /// `<source> | for <iter_var> <body>`
    For {
        source: ExprWithTy,
        iter_var: Ident,
        body: Ptr<Expr>,
        was_piped: bool,
    },
    /// `while <cond> <body>`
    While {
        condition: Ptr<Expr>,
        body: Ptr<Expr>,
        was_piped: bool,
    },

    /// `lhs catch ...`
    Catch {
        lhs: Ptr<Expr>,
        // TODO
    },

    /*
    /// `lhs | rhs`
    /// Note: `lhs | if ...`, `lhs | match ...`, `lhs | for ...` and
    /// `lhs | while ...` are inlined during parsing
    Pipe {
        lhs: Ptr<Expr>,
        // TODO
    },
    */
    Defer(Ptr<Expr>),

    /// `return <expr>`
    /// `^^^^^^` expr.span
    Return {
        expr: Option<ExprWithTy>,
    },
    Break {
        expr: Option<ExprWithTy>,
    },
    Continue,

    Semicolon(Option<Ptr<Expr>>),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    #[inline]
    pub fn new(kind: ExprKind, span: Span) -> Self {
        //Self { kind, span, ty: Type::Unset }
        Self { kind, span }
    }

    /// Returns a [`Span`] representing the entire expression.
    pub fn full_span(&self) -> Span {
        #[allow(unused_variables)]
        match self.kind {
            ExprKind::Fn(Fn { params, ret_type, body }) => self.span.join(body.full_span()),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::PositionalInitializer { lhs, .. }
            | ExprKind::NamedInitializer { lhs, .. } => {
                lhs.map(|e| e.full_span().join(self.span)).unwrap_or(self.span)
            },
            ExprKind::Dot { lhs, lhs_ty: _, rhs } => {
                lhs.map(|l| l.full_span()).unwrap_or(self.span).join(rhs.span)
            },
            ExprKind::Index { lhs, idx } => lhs.full_span().join(self.span),
            ExprKind::Call { func, args, pipe_idx } => match pipe_idx {
                Some(i) => args[i].full_span().join(self.span),
                None => func.full_span().join(self.span),
            },
            ExprKind::UnaryOp { expr, .. } => self.span.join(expr.full_span()),
            ExprKind::BinOp { lhs, rhs, .. }
            | ExprKind::Assign { lhs: ExprWithTy { expr: lhs, .. }, rhs }
            | ExprKind::BinOpAssign { lhs: ExprWithTy { expr: lhs, .. }, rhs, .. } => {
                lhs.full_span().join(rhs.full_span())
            },
            ExprKind::VarDecl(decl) => match &decl.default {
                Some(e) => self.span.join(e.full_span()),
                None => self.span,
            },
            ExprKind::If { condition, then_body, else_body, was_piped } => {
                let r_span = else_body.unwrap_or(then_body).full_span();
                if was_piped { condition.full_span() } else { self.span }.join(r_span)
            },
            ExprKind::Match { val, else_body, was_piped } => todo!(),
            ExprKind::For { source: ExprWithTy { expr: l, .. }, iter_var: _, body, was_piped }
            | ExprKind::While { condition: l, body, was_piped } => {
                if was_piped { l.full_span() } else { self.span }.join(body.full_span())
            },
            ExprKind::Catch { lhs } => todo!(),
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

    /// `&&`, `&&=`
    And,

    /// `||`, `||=`
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
            BinOpKind::And => "&&=",
            BinOpKind::Or => "||=",
            k => panic!("Unexpected binop kind: {:?}", k),
        }
    }

    pub fn finalize_arg_type(&self, arg_ty: Type, out_ty: Type) -> Type {
        match self {
            BinOpKind::Mul
            | BinOpKind::Div
            | BinOpKind::Mod
            | BinOpKind::Add
            | BinOpKind::Sub
            | BinOpKind::ShiftL
            | BinOpKind::ShiftR
            | BinOpKind::BitAnd
            | BinOpKind::BitXor
            | BinOpKind::BitOr
            | BinOpKind::And
            | BinOpKind::Or => out_ty,
            BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge => {
                debug_assert_eq!(out_ty, Type::Bool);
                arg_ty
            },
            BinOpKind::Range | BinOpKind::RangeInclusive => {
                let t = match out_ty {
                    Type::Range { elem_ty } | Type::RangeInclusive { elem_ty } => *elem_ty,
                    _ => unreachable_debug(),
                };
                debug_assert!(arg_ty.matches(t));
                t
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOpKind {
    /// `& <expr>`, `<expr>.&`
    AddrOf,
    /// `&mut <expr>`, `<expr>.&mut`
    AddrMutOf,
    /// `* <expr>`, `<expr>.*`
    Deref,
    /// `! <expr>`
    Not,
    /// `- <expr>`
    Neg,
    /// `<expr>?`
    Try,
    /*
    /// `<expr>!`
    Force,
    /// `<expr>!unsafe`
    ForceUnsafe,
    /// `<expr>.type`
    TypeOf,
    */
}

#[derive(Debug, Clone, Copy)]
pub struct Fn {
    pub params: VarDeclList,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

impl LitKind {
    pub fn parse(
        self,
        code: Ptr<str>,
        alloc: &bumpalo::Bump,
    ) -> Result<Ptr<()>, bumpalo::AllocErr> {
        macro_rules! alloc {
            ($val:expr) => {
                alloc.try_alloc($val).map(Ptr::from).map(Ptr::cast)
            };
        }

        match self {
            LitKind::Char => {
                let mut chars = code.chars();
                let c = chars.next().unwrap_debug();
                debug_assert!(c == '\'');
                let val = chars.next().unwrap_debug();
                let c = chars.next().unwrap_debug();
                debug_assert!(c == '\'');
                alloc!(val)
            },
            LitKind::BChar => todo!(),
            LitKind::Int => alloc!(code.parse::<i128>().unwrap_debug()),
            LitKind::Float => alloc!(code.parse::<f64>().unwrap_debug()),
            LitKind::Str => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarDeclList(pub Ptr<[VarDecl]>);

impl VarDeclList {
    pub fn find_field(self, name: &str) -> Option<(usize, &VarDecl)> {
        unsafe { forget_lifetime(&*self.0) }
            .into_iter()
            .enumerate()
            .find(|(_, f)| &*f.ident.text == name)
    }

    pub fn into_type_iter(&self) -> impl DoubleEndedIterator<Item = Type> + '_ {
        self.iter().map(|decl| decl.ty)
    }
}

impl From<Ptr<[VarDecl]>> for VarDeclList {
    fn from(value: Ptr<[VarDecl]>) -> Self {
        VarDeclList(value)
    }
}

impl Deref for VarDeclList {
    type Target = [VarDecl];

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl DerefMut for VarDeclList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}
