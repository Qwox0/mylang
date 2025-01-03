use crate::{parser::lexer::Span, ptr::Ptr, type_::Type, util::forget_lifetime};
use debug::DebugAst;
use std::{fmt, ops::Deref};

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

    IntLit(Ptr<str>),
    FloatLit(Ptr<str>),
    /// `true`, `false`
    BoolLit(bool),
    /// `'a'`
    CharLit(char),
    /// `b'a'`
    BCharLit(u8),
    /// `"hello world"`
    StrLit(Ptr<str>),

    /// `*<ty>`
    /// `*mut <ty>`
    PtrTy {
        ty: Type,
        is_mut: bool,
    },
    /// `[]T` -> `struct { ptr: *T, len: u64 }`
    /// `[]mut T`
    SliceTy {
        ty: Type,
        is_mut: bool,
    },
    /// `[<count>]ty`
    ArrayTy {
        count: Ptr<Expr>,
        ty: Type,
    },
    /// `?<ty>`
    OptionShort(Type),

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

    /// `expr.as(ty)`
    /// TODO: remove this [`ExprKind`] when implementing method calls.
    Cast {
        lhs: ExprWithTy,
        target_ty: Type,
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

    Range {
        start: Option<Ptr<Expr>>,
        end: Option<Ptr<Expr>>,
        is_inclusive: bool,
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

    /// `for <iter_var> in <source> <body>`
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

    /// `xx input`
    /// source: Jai
    Autocast {
        expr: ExprWithTy,
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
    // Semicolon(Option<Ptr<Expr>>),
}

impl ExprKind {
    pub(crate) fn block_expects_trailing_semicolon(&self) -> bool {
        match self {
            ExprKind::Block { .. } => false,
            /*
            | ExprKind::StructDef(..)
            | ExprKind::UnionDef(..)
            | ExprKind::EnumDef(..) => false,
            ExprKind::VarDecl(var_decl) => var_decl
                .default
                .map(|e| e.kind.block_expects_trailing_semicolon())
                .unwrap_or(true),
            ExprKind::Extern { .. } => todo!(),
            */
            &ExprKind::If { then_body, else_body, .. } => {
                else_body.unwrap_or(then_body).kind.block_expects_trailing_semicolon()
            },
            ExprKind::Match { .. } => todo!(),
            // ExprKind::Fn(Fn { body, .. })
            ExprKind::For { body, .. } | ExprKind::While { body, .. } => {
                body.kind.block_expects_trailing_semicolon()
            },
            // ExprKind::Catch { .. } => todo!(),
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
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
            ExprKind::Fn(Fn { params, ret_type, body: Some(body) }) => {
                self.span.join(body.full_span())
            },
            ExprKind::Fn(Fn { params, ret_type: Type::Unevaluated(t), body: None }) => {
                self.span.join(t.full_span())
            },
            ExprKind::Fn(Fn { params, ret_type, body: None }) => self.span,
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
            // ExprKind::Semicolon(_) => todo!(),
            _ => self.span,
        }
    }

    pub fn as_var_decl(self) -> Option<VarDecl> {
        let Expr { kind, span } = self;
        match kind {
            ExprKind::VarDecl(decl) => Some(decl),
            ExprKind::Ident(text) => Some(VarDecl::new_basic(Ident { text, span }, Type::Unset)),
            _ => None,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// if `body == None` this is a function type
    pub body: Option<Ptr<Expr>>,
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

impl VarDecl {
    pub fn new_basic(ident: Ident, ty: Type) -> Self {
        VarDecl { markers: DeclMarkers::default(), ident, ty, default: None, is_const: false }
    }
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

impl From<&'static str> for Ident {
    fn from(value: &'static str) -> Self {
        Ident { text: Ptr::from(value), span: Span::new(0, 0) }
    }
}

#[allow(unused)]
pub struct Pattern {
    kind: ExprKind, // TODO: own kind enum
    span: Span,
}

pub type VarDeclList = Ptr<[VarDecl]>;

pub trait VarDeclListTrait {
    fn find_field(&self, name: &str) -> Option<(usize, &VarDecl)>;

    fn as_type_iter(&self) -> impl DoubleEndedIterator<Item = Type> + '_;
}

impl VarDeclListTrait for [VarDecl] {
    fn find_field(&self, name: &str) -> Option<(usize, &VarDecl)> {
        unsafe { forget_lifetime(&*self) }
            .into_iter()
            .enumerate()
            .find(|(_, f)| &*f.ident.text == name)
    }

    fn as_type_iter(&self) -> impl DoubleEndedIterator<Item = Type> + '_ {
        self.iter().map(|decl| decl.ty)
    }
}
