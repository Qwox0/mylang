use super::lexer::{self, Span};

/// Abstract Syntax Tree for source code.
pub type AST = Expr;
pub enum Expr {
    /// `let <name>: <type> = <rhs>`
    Let {
        name: Ident,
        type_: Option<Span>,
        rhs: Box<Expr>,
    },

    /// `<expr>;`
    Semicolon(Box<Expr>),

    /// `{ ... }`
    Block {
        nodes: Vec<Expr>,
    },
    /// `( ... )`
    Parenthesis(Box<Expr>),

    Literal(Span),
    Ident(Ident),
    Type(Type),

    Fn {
        signature: (),
        body: Box<Expr>,
    },
    /// `struct { ... }`
    Struct {
        fields: Vec<StructFields>,
    },
    /// `struct(...)`
    TupleStruct(Vec<Type>),
    Union {},
    Enum {},

    PreOp(PreOpKind, Box<Expr>),
    PostOp(PostOpKind, Box<Expr>),


    Range(Span),
    /// `<lhs> . <rhs>`
    Dot {
        lhs: Box<Expr>,
        rhs: Ident,
    },
    /// `<func> ( <params> )`
    Call {
        func: Box<Expr>,
        params: Vec<Expr>,
    },
    /// `<lhs> [ <idx> ]`
    Index {
        lhs: Box<Expr>,
        idx: Box<Expr>,
    },
}

pub struct Ident {
    name: String,
    span: Span,
}

pub struct StructFields {
    name: Ident,
    type_: Type,
}

pub enum Type {}

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
}

/*
pub struct Expr {
    kind: ExprKind,
}

pub enum ExprKind {
    Array(Vec<Box<Expr>>),

    /// `<A, B>(a: A, b: B) -> C { ... }`
    Function,
    /// `struct { ... }`
    Struct,
    /// `enum { ... }`
    Enum,
    /// `union { ... }`
    Union,
    /// `trait { ... }`
    Trait,
}
*/

fn test() {
    type int = isize;

    enum MyEnum {
        A,
        B(int),
        C { field: int },
    }

    struct MyType1 {
        e: MyEnum,
        name: String,
    }

    enum MyType2 {
        A { name: String },
        B { _0: int, name: String },
        C { field: int, name: String },
    }

    fn test1(x: MyType1) {
        match x.e {
            MyEnum::A => todo!(),
            MyEnum::B(a) => todo!(),
            MyEnum::C { field } => todo!(),
        }
    }

    fn test2(x: MyType2) {
        match x {
            MyType2::A { name } => todo!(),
            MyType2::B { _0, name } => todo!(),
            MyType2::C { field, name } => todo!(),
        }
    }

    let x = 1;
    let t1 = (x);
    let t2 = (x,);
    let t3 = (x, x);
    let t4 = (x, x);
}
