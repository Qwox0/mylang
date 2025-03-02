use crate::{
    codegen::llvm::finalize_ty,
    context::{FilesIndex, primitives},
    diagnostic_reporter::{DiagnosticReporter, cerror},
    parser::lexer::Span,
    ptr::{OPtr, Ptr},
    sema,
    type_::{RangeKind, ty_match},
    util::{UnwrapDebug, then},
};
use core::fmt;

pub mod debug;

/*
macro_rules! ast_variants2 {
    ($($(#[$attr:meta])* $name:ident : $base:ident {
        $($(#[$field_attr:meta])* $field:ident : $ty:ty),* $(,)?
    }),+ $(,)?) => {
        $(ast_variants2! {
            _ $base => $(#[$attr])* $name {
                $($(#[$field_attr])* $field : $ty),*
            }
        })+

        /// [`Ast`] as a rust enum which can be used with pattern matching
        ///
        /// This works because `#[repr]` forces the tag to be the first field <https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting>
        ///
        #[derive(Debug)]
        #[repr(u8)]
        pub enum AstEnum {
            $($name {
                ty: OPtr<Type>,
                span: Span,
                parenthesis_count: u8,
                $($field : $ty),*
            },)+
        }
    };
    (_ Ast => $(#[$attr:meta])* $name:ident {
        $($(#[$field_attr:meta])* $field:ident : $ty:ty),* $(,)?
    }) => {
        #[derive(Debug)]
        $(#[$attr])*
        #[repr(C)]
        pub struct $name {
            pub kind: AstKind,
            pub ty: OPtr<$crate::ast::Type>,
            pub span: Span,
            pub parenthesis_count: u8,
            $(
                $(#[$field_attr])*
                pub $field : $ty
            ),*
        }

        impl HasAstKind for $name {
            #[inline]
            fn get_kind(&self) -> AstKind { self.kind }
        }
    };
    (_ ConstVal => $(#[$attr:meta])* $name:ident {
        $($(#[$field_attr:meta])* $field:ident : $ty:ty),* $(,)?
    }) => {
        ast_variants2! {
            _ Ast => $(#[$attr])* $name {
                // more_fields: u64,
                $($(#[$field_attr])* $field : $ty),*
            }
        }
    };
    (_ Type => $(#[$attr:meta])* $name:ident {
        $($(#[$field_attr:meta])* $field:ident : $ty:ty),* $(,)?
    }) => {
        ast_variants2! {
            _ ConstVal => $(#[$attr])* $name {
                // more_fields2: u64,
                $($(#[$field_attr])* $field : $ty),*
            }
        }
    };
}

mod new {
    use super::{AstKind, HasAstKind, OPtr, Ptr, Span};

    ast_variants2! {
        Ast : Ast {},
        ConstVal : ConstVal {},
        Type : Type {},
    }
}
*/

// --------------

// don't forget to change `AstEnum`, `ConstValEnum`, `TypeEnum`
macro_rules! inherit_ast {
    (
        $(#[$attr:meta])*
        struct $name:ident {
            $(
                $(#[$field_attr:meta])*
                $field:ident : $ty:ty
            ),* $(,)?
        }
    ) => {
        $(#[$attr])*
        #[repr(C)]
        pub struct $name {
            pub kind: AstKind,
            pub ty: OPtr<$crate::ast::Type>,
            /// If the [`Ast`] node is replaced directly, we would lose the correct [`Span`].
            pub replacement: OPtr<Ast>,
            pub span: Span,
            pub parenthesis_count: u8,
            $(
                $(#[$field_attr])*
                pub $field : $ty
            ),*
        }

        impl HasAstKind for $name {
            #[inline]
            fn get_kind(&self) -> AstKind { self.kind }
        }
    };
}
pub(crate) use inherit_ast;

inherit_ast! {
    struct Ast {}
}

/// Constructor for ast nodes
macro_rules! ast_new {
    ($kind:ident {
        $(
            $field:ident
            $( : $val:expr )?
        ),* $(,)?
    }) => {
        crate::ast::$kind {
            kind: crate::ast::AstKind::$kind,
            ty: None,
            replacement: None,
            parenthesis_count: 0,
            $( $field $(: $val)? ),*
        }
    };
}
pub(crate) use ast_new;

macro_rules! type_new {
    ($kind:ident {
        $(
            $field:ident
            $( : $val:expr )?
        ),* $(,)?
    }) => {{
        let kind = crate::ast::AstKind::$kind;
        debug_assert!(crate::ast::Type::KINDS.contains(&kind));
        crate::ast::$kind {
            kind,
            ty: Some(crate::context::primitives().type_ty),
            replacement: None,
            parenthesis_count: 0,
            span: Span::ZERO,
            $( $field $(: $val)? ),*
        }
    }};
}
pub(crate) use type_new;

pub trait HasAstKind {
    fn get_kind(&self) -> AstKind;
}

pub unsafe trait AstVariant: HasAstKind {
    const KIND: AstKind;
}
pub unsafe trait ConstValVariant: AstVariant {}
pub unsafe trait TypeVariant: ConstValVariant {}

macro_rules! ast_variants {
    (
        $(
            $(#[$attr:meta])*
            $name:ident {
                $(
                    $(#[$field_attr:meta])*
                    $field:ident : $ty:ty
                ),* $(,)?
            }
        ),+ $(,)?
        ===== Constant Values =====
        $(
            $(#[$c_attr:meta])*
            $c_name:ident {
                $(
                    $(#[$c_field_attr:meta])*
                    $c_field:ident : $c_ty:ty
                ),* $(,)?
            }
        ),+ $(,)?
        ===== Types =====
        $(
            $(#[$t_attr:meta])*
            $t_name:ident {
                $(
                    $(#[$t_field_attr:meta])*
                    $t_field:ident : $t_ty:ty
                ),* $(,)?
            }
        ),+ $(,)?
    ) => {
        $(
            inherit_ast! {
                #[derive(Debug)]
                $(#[$attr])* struct $name {
                    $(
                        $(#[$field_attr])*
                        $field : $ty
                    ),*
                }
            }

            unsafe impl AstVariant for $name { const KIND: AstKind = AstKind::$name; }
        )+
        $(
            inherit_ast! {
                $(#[$c_attr])* struct $c_name {
                    $(
                        $(#[$c_field_attr])*
                        $c_field : $c_ty
                    ),*
                }
            }

            impl std::fmt::Debug for $c_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    ConstVal::fmt(&Ptr::from_ref(self).cast(), f)
                }
            }

            unsafe impl AstVariant for $c_name { const KIND: AstKind = AstKind::$c_name; }
            unsafe impl ConstValVariant for $c_name {}
        )+
        $(
            inherit_ast! {
                $(#[$t_attr])* struct $t_name {
                    $(
                        $(#[$t_field_attr])*
                        $t_field : $t_ty
                    ),*
                }
            }

            impl std::fmt::Debug for $t_name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    Type::fmt(&Ptr::from_ref(self).upcast_to_type(), f)
                }
            }

            unsafe impl AstVariant for $t_name { const KIND: AstKind = AstKind::$t_name; }
            unsafe impl ConstValVariant for $t_name {}
            unsafe impl TypeVariant for $t_name {}
        )+

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(u8)]
        pub enum AstKind {
            $($name,)+
            $($c_name,)+
            $($t_name,)+
        }

        /// [`Ast`] as a rust enum which can be used with pattern matching
        ///
        /// This works because `#[repr]` forces the tag to be the first field <https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting>
        ///
        #[derive(Debug)]
        #[repr(u8)]
        pub enum AstEnum {
            // don't forget to change `inherit_ast`
            $($name {
                ty: OPtr<$crate::ast::Type>,
                replacement: OPtr<Ast>,
                span: Span,
                parenthesis_count: u8,
                $($field : $ty),*
            },)+
            $($c_name {
                ty: OPtr<$crate::ast::Type>,
                replacement: OPtr<Ast>,
                span: Span,
                parenthesis_count: u8,
                $($c_field : $c_ty),*
            },)+
            $($t_name {
                ty: OPtr<$crate::ast::Type>,
                replacement: OPtr<Ast>,
                span: Span,
                parenthesis_count: u8,
                $($t_field : $t_ty),*
            },)+
        }

        #[derive(Debug)]
        #[repr(u8)]
        pub enum ConstValEnum {
            $(
                $c_name {
                    ty: OPtr<Type>,
                    replacement: OPtr<Ast>,
                    span: Span,
                    parenthesis_count: u8,
                    $($c_field : $c_ty),*
                } = AstKind::$c_name as u8,
            )+
            $(
                $t_name {
                    ty: OPtr<Type>,
                    replacement: OPtr<Ast>,
                    span: Span,
                    parenthesis_count: u8,
                    $($t_field : $t_ty),*
                } = AstKind::$t_name as u8,
            )+
        }

        #[derive(Debug)]
        #[repr(u8)]
        pub enum TypeEnum {
            $(
                $t_name {
                    ty: OPtr<Type>,
                    replacement: OPtr<Ast>,
                    span: Span,
                    parenthesis_count: u8,
                    $(
                        $(#[$t_field_attr])*
                        $t_field : $t_ty
                    ),*
                } = AstKind::$t_name as u8,
            )+
            Unset = u8::MAX,
        }

        impl AstKind {
            pub const fn size_of_variant(self) -> usize {
                match self {
                    $(AstKind::$name => size_of::<$name>(),)+
                    $(AstKind::$c_name => size_of::<$c_name>(),)+
                    $(AstKind::$t_name => size_of::<$t_name>(),)+
                }
            }
        }

        impl ConstVal {
            pub const KINDS: [AstKind; 19] = [$(AstKind::$c_name,)+ $(AstKind::$t_name,)+];
        }

        impl Type {
            pub const KINDS: [AstKind; 12] = [$(AstKind::$t_name,)+];
        }
    };
}

ast_variants! {
    Ident {
        text: Ptr<str>,
    },

    /// `{ <stmt>* }`
    Block {
        has_trailing_semicolon: bool,
        // parent: OPtr<Block>,
        // pos_in_parent: StmtPos,
        /// all statements in this block
        stmts: Ptr<[Ptr<Ast>]>,
        //decls: Vec<Ptr<Decl>>,
    },

    /// `alloc(MyStruct).( a, b, c = <expr>, )`
    /// `               ^^^^^^^^^^^^^^^^^^^^^^` expr.span
    ///
    /// [`Type`] -> value
    /// `*T` -> `*T`
    PositionalInitializer {
        lhs: OPtr<Ast>,
        args: Ptr<[Ptr<Ast>]>,
    },
    /// `alloc(MyStruct).{ a = <expr>, b, }`
    /// `               ^^^^^^^^^^^^^^^^^^^` expr.span
    ///
    /// [`Type`] -> value
    /// `*T` -> `*T`
    NamedInitializer {
        lhs: OPtr<Ast>,
        fields: Ptr<[(Ptr<Ident>, OPtr<Ast>)]>, // TODO: SOA
    },
    /// `alloc(MyArray).[<expr>, <expr>, ..., <expr>,]`
    ArrayInitializer {
        lhs: OPtr<Ast>,
        elements: Ptr<[Ptr<Ast>]>,
    },
    /// `alloc(MyArray).[<expr>; <count>]`
    ArrayInitializerShort {
        lhs: OPtr<Ast>,
        val: Ptr<Ast>,
        count: Ptr<Ast>,
    },

    /// `expr . ident`, `.ident`
    /// `     ^` `       ^` expr.span
    Dot {
        has_lhs: bool,
        lhs: OPtr<Ast>,
        rhs: Ptr<Ident>,
    },
    /// `<lhs> [ <idx> ]`
    /// `              ^` expr.span
    Index {
        lhs: Ptr<Ast>,
        idx: Ptr<Ast>,
    },

    /// `expr.as(ty)`
    /// TODO: remove this [`Ast`] when implementing generic method calls.
    Cast {
        operand: Ptr<Ast>,
        target_ty: Ptr<Ast>,
    },
    /// `xx input`
    /// source: Jai
    Autocast {
        operand: Ptr<Ast>,
    },

    /// `<func> ( <expr>, ..., param=<expr>, ... )`
    /// `                                        ^ expr.span`
    Call {
        func: Ptr<Ast>,
        args: Ptr<[Ptr<Ast>]>,
        /// which argument was piped into this [`Ast::Call`]
        pipe_idx: Option<usize>,
    },

    /// examples: `&<expr>`, `<expr>.*`, `- <expr>`
    /// `          ^` `             ^^` ` ^ expr.span`
    UnaryOp {
        is_postfix: bool,
        op: UnaryOpKind,
        operand: Ptr<Ast>,
    },
    /// `<lhs> op <lhs>`
    /// `      ^^ expr.span`
    BinOp {
        lhs: Ptr<Ast>,
        op: BinOpKind,
        rhs: Ptr<Ast>,
    },
    Range {
        is_inclusive: bool,
        start: OPtr<Ast>,
        end: OPtr<Ast>,
    },

    /// `<lhs> = <lhs>`
    Assign {
        lhs: Ptr<Ast>,
        rhs: Ptr<Ast>,
    },
    /// `<lhs> op= <lhs>`
    BinOpAssign {
        lhs: Ptr<Ast>,
        op: BinOpKind,
        rhs: Ptr<Ast>,
    },

    /// variable declaration (and optional initialization)
    /// `mut rec <name>: <ty>`
    /// `mut rec <name>: <ty> = <init>`
    /// `mut rec <name>: <ty> : <init>`
    /// `mut rec <name> := <init>`
    /// `mut rec <name> :: <init>`
    /// `expr.span` must describe the entire expression if `default.is_none()`,
    /// otherwise only the start is important
    Decl {
        is_const: bool,
        is_extern: bool,
        markers: DeclMarkers,
        ident: Ptr<Ident>,
        var_ty_expr: OPtr<Ast>,
        var_ty: OPtr<Type>,
        /// also used for default value in fn params, struct fields, ...
        init: OPtr<Ast>,
        // init_const_val: OPtr<ConstVal>, // TODO: benchmark this
    },
    /*
    Extern {
        ident: Ptr<Ident>,
        var_ty_expr: Ptr<Ast>,
        var_ty: OPtr<Type>,
    },
    */

    /// `if <cond> <then>` (`else <else>`)
    /// `^^` expr.span
    If {
        was_piped: bool,
        condition: Ptr<Ast>,
        then_body: Ptr<Ast>,
        else_body: OPtr<Ast>,
    },
    /// `match <val> <body>` (`else <else>`)
    Match {
        was_piped: bool,
        val: Ptr<Ast>,
        // TODO
        else_body: OPtr<Ast>,
    },

    /// `for <iter_var> in <source> <body>`
    /// `<source> | for <iter_var> <body>`
    For {
        was_piped: bool,
        source: Ptr<Ast>,
        iter_var: Ptr<Ident>,
        body: Ptr<Ast>,
    },
    /// `while <cond> <body>`
    While {
        was_piped: bool,
        condition: Ptr<Ast>,
        body: Ptr<Ast>,
    },

    /*
    /// `lhs catch ...`
    Catch {
        lhs: Ptr<Ast>,
        // TODO
    },
    */

    Defer { stmt: Ptr<Ast> },

    /// `return <expr>`
    /// `^^^^^^` expr.span
    Return {
        val: OPtr<Ast>,
        parent_fn: OPtr<Fn>,
    },
    Break {
        val: OPtr<Ast>,
    },
    Continue {},

    ===== Constant Values =====

    IntVal { val: i64 },
    FloatVal { val: f64 },
    BoolVal { val: bool },
    CharVal { val: char },
    // BCharLit { val: u8 },
    StrVal { text: Ptr<str> },
    PtrVal { val: u64 },

    ImportDirective {
        path: Ptr<StrVal>,
        files_idx: FilesIndex,
    },

    ===== Types =====

    /// `void`, `never`, `bool`, `type`
    SimpleTy {
        decl: Ptr<Decl>,
    },
    IntTy {
        is_signed: bool,
        bits: u32,
    },
    FloatTy {
        bits: u32,
    },

    /// `*<ty>`
    /// `*mut <ty>`
    PtrTy {
        is_mut: bool,
        pointee: Ptr<Ast>,
    },
    /// `[]T` -> `struct { ptr: *T, len: u64 }`
    /// `[]mut T`
    SliceTy {
        is_mut: bool,
        elem_ty: Ptr<Ast>,
    },
    /// `[<count>]ty`
    ArrayTy {
        len: Ptr<Ast>,
        elem_ty: Ptr<Ast>,
    },

    /// `struct { a: int, b: String, c: (u8, u32) }`
    StructDef { fields: DeclList },
    /// `union { a: int, b: String, c: (u8, u32) }`
    UnionDef { fields: DeclList },
    /// `enum { A, B(i64) }`
    EnumDef { variants: DeclList },

    RangeTy {
        elem_ty: Ptr<Type>,
        rkind: RangeKind,
    },

    /// `?<ty>`
    OptionTy {
        inner_ty: Ptr<Ast>,
    },

    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <type> { <body> }`
    /// `(<ident>, <ident>: <ty>, ..., <ident>,) -> <body>`
    /// `-> <type> { <body> }`
    /// `-> <body>`
    /// `^ expr.span`
    Fn {
        params: DeclList,
        ret_ty_expr: OPtr<Ast>,
        ret_ty: OPtr<Type>,
        /// if `body == None` this is a function type
        body: OPtr<Ast>,
    },
}

inherit_ast! {
    struct ConstVal {}
}

inherit_ast! {
    struct Type {}
}

pub trait UpcastToAst {
    fn upcast(self) -> Ptr<Ast>;
}

macro_rules! impl_UpcastToAst {
    ($($name:ty),*) => { $(
        impl UpcastToAst for Ptr<$name> {
            fn upcast(self) -> Ptr<Ast> {
                self.cast()
            }
        }
    )* };
}

impl_UpcastToAst! { AstEnum, ConstVal, Type }

impl<V: AstVariant> UpcastToAst for Ptr<V> {
    fn upcast(self) -> Ptr<Ast> {
        debug_assert_eq!(self.get_kind(), V::KIND);
        self.cast()
    }
}

impl<V: ConstValVariant> Ptr<V> {
    pub fn upcast_to_const_val(self) -> Ptr<ConstVal> {
        debug_assert_eq!(self.get_kind(), V::KIND);
        self.cast()
    }
}

impl<V: TypeVariant> Ptr<V> {
    pub fn upcast_to_type(self) -> Ptr<Type> {
        debug_assert_eq!(self.get_kind(), V::KIND);
        self.cast()
    }
}

impl Ptr<Ast> {
    pub fn is_const_val(self) -> bool {
        self.kind.is_const_val_kind()
    }

    /// TODO: check if this is cheaper than [`Ptr::has_type_kind`]
    pub fn is_type(self) -> bool {
        if self.ty.u() != primitives().type_ty {
            return false;
        }
        debug_assert!(self.has_type_kind(), "expected type kind; got: {:?}", self.kind);
        debug_assert_ne!(self.ty, None);
        true
    }

    /// only use this for debugging. otherwise use [`Ptr::is_type`] instead
    pub fn has_type_kind(self) -> bool {
        self.kind.is_type_kind()
    }

    /// resolve all replacements
    pub fn rep(self) -> Ptr<Ast> {
        let mut active = self;
        while let Some(replacement) = active.replacement {
            debug_assert!(replacement != active);
            debug_assert!(replacement != self);
            active = replacement;
        }
        active
    }

    /// resolve all replacements
    pub fn rep_mut(&mut self) -> &mut Ptr<Ast> {
        let mut active = self;
        while let Some(replacement) = active.as_mut().replacement.as_mut() {
            debug_assert!(*replacement != *active);
            active = replacement;
        }
        active
    }

    pub fn downcast<V: AstVariant>(self) -> Ptr<V> {
        let p = self.rep();
        debug_assert_eq!(p.kind, V::KIND);
        p.cast()
    }

    pub fn try_downcast<V: AstVariant>(self) -> OPtr<V> {
        let p = self.rep();
        then!(p.kind == V::KIND => p.downcast())
    }

    /// downcast to a [`ConstVal`]
    pub fn downcast_const_val(self) -> Ptr<ConstVal> {
        let p = self.rep();
        debug_assert!(p.is_const_val());
        p.cast()
    }

    pub fn try_get_const_val(self) -> Result<Ptr<ConstVal>, sema::SemaError> {
        let p = self.rep();
        then!(p.is_const_val() => p.downcast_const_val()).ok_or_else(|| {
            cerror!(self.full_span(), "value not known at compile time");
            ().into()
        })
    }

    #[inline]
    pub fn downcast_type(self) -> Ptr<Type> {
        let p = self.rep();
        debug_assert!(p.is_type());
        p.cast()
    }

    pub fn try_downcast_type(self) -> OPtr<Type> {
        let p = self.rep();
        then!(p.is_type() => p.downcast_type())
    }

    pub fn downcast_type_ref(&mut self) -> &mut Ptr<Type> {
        let p = self.rep_mut();
        debug_assert!(p.is_type());
        Ptr::from_ref(p).cast::<Ptr<Type>>().as_mut()
    }

    pub fn int<Int: TryFrom<i64>>(self) -> Int
    where Int::Error: fmt::Debug {
        let int = self.downcast::<IntVal>().val;
        debug_assert!(Int::try_from(int).is_ok());
        Int::try_from(int).u()
    }

    /// similar to [`Ast::full_span`] but returns a better span for [`Block`] nodes.
    pub fn return_val_span(self) -> Span {
        self.try_downcast::<Block>()
            .and_then(|b| b.stmts.last().copied())
            .unwrap_or(self)
            .full_span()
    }
}

impl Ptr<ConstVal> {
    pub fn downcast<V: ConstValVariant>(self) -> Ptr<V> {
        debug_assert!(self.replacement.is_none());
        debug_assert_eq!(self.kind, V::KIND);
        debug_assert!(self.upcast().is_const_val());
        self.cast()
    }

    pub fn try_downcast<V: ConstValVariant>(self) -> OPtr<V> {
        debug_assert!(self.replacement.is_none());
        then!(self.kind == V::KIND => self.downcast())
    }

    pub fn downcast_type(self) -> Ptr<Type> {
        debug_assert!(self.replacement.is_none());
        debug_assert!(self.upcast().is_type());
        self.cast()
    }

    pub fn try_downcast_type(self) -> OPtr<Type> {
        debug_assert!(self.replacement.is_none());
        then!(self.upcast().has_type_kind() => self.downcast_type())
    }

    /// Expects `self` to be an [`IntVal`] or a [`FloatVal`].
    pub fn float_val(self) -> f64 {
        if let Some(int) = self.try_downcast::<IntVal>() {
            int.val as f64
        } else {
            self.downcast::<FloatVal>().val
        }
    }
}

impl Ptr<Type> {
    pub fn downcast<V: TypeVariant>(self) -> Ptr<V> {
        debug_assert!(self.replacement.is_none());
        debug_assert_eq!(self.kind, V::KIND);
        debug_assert!(self.upcast().has_type_kind());
        self.cast()
    }

    pub fn downcast_ref<V: TypeVariant>(&mut self) -> &mut Ptr<V> {
        debug_assert!(self.replacement.is_none());
        debug_assert_eq!(self.kind, V::KIND);
        debug_assert!(self.upcast().has_type_kind());
        Ptr::from_ref(self).cast::<Ptr<V>>().as_mut()
    }

    pub fn try_downcast<V: TypeVariant>(self) -> OPtr<V> {
        debug_assert!(self.replacement.is_none());
        then!(self.kind == V::KIND => self.downcast())
    }

    pub fn try_downcast_ref<V: TypeVariant>(&mut self) -> Option<&mut Ptr<V>> {
        debug_assert!(self.replacement.is_none());
        then!(self.kind == V::KIND => self.downcast_ref())
    }
}

impl Ast {
    /// Convert the ast node into a matchable rust enum
    #[inline]
    pub fn matchable(&self) -> Ptr<AstEnum> {
        Ptr::<Ast>::from_ref(self).cast::<AstEnum>()
    }

    pub(crate) fn block_expects_trailing_semicolon(&self) -> bool {
        match self.matchable().as_ref() {
            AstEnum::Block { .. } => false,
            AstEnum::Decl { init, is_const, .. } => {
                if !is_const {
                    return true;
                }
                let Some(init) = init else { return true };
                !matches!(
                    init.kind,
                    AstKind::StructDef | AstKind::UnionDef | AstKind::EnumDef | AstKind::Fn
                )
            },
            &AstEnum::If { then_body, else_body, .. } => {
                else_body.unwrap_or(then_body).block_expects_trailing_semicolon()
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { body, .. } | AstEnum::While { body, .. } => {
                body.block_expects_trailing_semicolon()
            },
            _ => true,
        }
    }

    /// Returns a [`Span`] representing the entire expression.
    pub fn full_span(&self) -> Span {
        let span = self.span;
        if self.parenthesis_count > 0 {
            return span;
        }
        match self.matchable().as_ref() {
            AstEnum::PositionalInitializer { lhs, .. } | AstEnum::NamedInitializer { lhs, .. } => {
                lhs.map(|e| e.full_span().join(span)).unwrap_or(span)
            },
            AstEnum::Dot { lhs, has_lhs, rhs, .. } => {
                lhs.filter(|_| *has_lhs).map(|l| l.full_span()).unwrap_or(span).join(rhs.span)
            },
            AstEnum::Index { lhs, .. } | AstEnum::Cast { operand: lhs, .. } => {
                lhs.full_span().join(span)
            },
            AstEnum::Autocast { operand, .. } | AstEnum::UnaryOp { operand, .. } => {
                span.join(operand.full_span())
            },
            AstEnum::Call { func, args, pipe_idx, .. } => match *pipe_idx {
                Some(i) => args[i].full_span().join(span),
                None => func.full_span().join(span),
            },
            AstEnum::BinOp { lhs, rhs, .. }
            | AstEnum::Assign { lhs, rhs, .. }
            | AstEnum::BinOpAssign { lhs, rhs, .. } => lhs.full_span().join(rhs.full_span()),
            AstEnum::Range { start, end, .. } => {
                let start = start.map(|s| s.full_span()).unwrap_or(span);
                let end = end.map(|e| e.full_span()).unwrap_or(span);
                start.join(end)
            },
            AstEnum::Decl { is_extern: false, init, .. } => match &init {
                Some(e) => span.join(e.full_span()),
                None => span,
            },
            AstEnum::Decl { is_extern: true, var_ty_expr, .. } => {
                span.join(var_ty_expr.u().full_span())
            },
            AstEnum::If { condition, then_body, else_body, was_piped, .. } => {
                let r_span = else_body.unwrap_or(*then_body).full_span();
                if *was_piped { condition.full_span() } else { span }.join(r_span)
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { source: l, iter_var: _, body, was_piped, .. }
            | AstEnum::While { condition: l, body, was_piped, .. } => {
                if *was_piped { l.full_span() } else { span }.join(body.full_span())
            },
            // AstEnum::Catch { .. } => todo!(),
            AstEnum::Return { val, .. } => match val {
                Some(val) => span.join(val.full_span()),
                None => span,
            },
            AstEnum::ImportDirective { path, .. } => span.join(path.span),

            AstEnum::SimpleTy { .. } => todo!(),
            AstEnum::IntTy { .. } => todo!(),
            AstEnum::FloatTy { .. } => todo!(),
            AstEnum::PtrTy { pointee: i, .. }
            | AstEnum::SliceTy { elem_ty: i, .. }
            | AstEnum::ArrayTy { elem_ty: i, .. }
            | AstEnum::OptionTy { inner_ty: i, .. } => self.span.join(i.full_span()),
            AstEnum::StructDef { .. } | AstEnum::UnionDef { .. } | AstEnum::EnumDef { .. } => span,
            AstEnum::RangeTy { .. } => todo!(),
            AstEnum::Fn { body, ret_ty_expr, .. } => {
                span.join(body.or(*ret_ty_expr).u().full_span())
            },
            _ => span,
        }
    }
}

impl ConstVal {
    #[inline]
    pub fn matchable(&self) -> Ptr<ConstValEnum> {
        Ptr::from(self).cast()
    }
}

impl Type {
    #[inline]
    pub fn matchable(&self) -> Ptr<TypeEnum> {
        Ptr::from(self).cast()
    }
}

impl TypeEnum {
    #[inline]
    pub fn as_type(&self) -> OPtr<Type> {
        match self {
            TypeEnum::Unset => None,
            _ => Some(Ptr::from(self).cast()),
        }
    }
}

pub trait OptionAstExt {
    fn is_type(self) -> bool;

    fn cv(self) -> Ptr<ConstVal>;

    //fn ref_downcast_type(&mut self) -> &mut OPtr<Type>;

    //fn downcast_type_ref(&mut self) -> &mut Ptr<Type>;
}

impl OptionAstExt for Option<Ptr<Ast>> {
    #[inline]
    fn is_type(self) -> bool {
        self.map(Ptr::<Ast>::is_type).unwrap_or(false)
    }

    fn cv(self) -> Ptr<ConstVal> {
        self.u().downcast_const_val()
    }
}

pub trait OptionTypeExt {
    fn matchable(self) -> Ptr<TypeEnum>;
    fn downcast<V: TypeVariant>(self) -> Ptr<V>;
    fn try_downcast<V: TypeVariant>(self) -> OPtr<V>;
    fn upcast(self) -> OPtr<Ast>;
}

impl OptionTypeExt for Option<Ptr<Type>> {
    #[inline]
    fn matchable(self) -> Ptr<TypeEnum> {
        match self {
            Some(t) => t.matchable(),
            None => Ptr::from_ref(&TypeEnum::Unset),
        }
    }

    #[inline]
    fn downcast<V: TypeVariant>(self) -> Ptr<V> {
        self.u().downcast()
    }

    #[inline]
    fn try_downcast<V: TypeVariant>(self) -> OPtr<V> {
        self?.try_downcast()
    }

    #[inline]
    fn upcast(self) -> OPtr<Ast> {
        self.map(Ptr::cast)
    }
}

/// The number of [`Decl`]s before this statement.
///
/// ```text
/// stmt pos=0
/// decl pos=0
/// stmt pos=1
/// stmt pos=1
/// decl pos=1
/// stmt pos=2
/// ```
///
/// A [`Decl`] has the same pos as the statement before it because ...
///
/// ```mylang
/// a := a + 1; // the `a` in the init expr shouldn't be resolved to this decl.
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StmtPos(pub usize);

impl AstEnum {
    #[inline]
    pub fn as_ast(&self) -> Ptr<Ast> {
        Ptr::from(self).cast()
    }
}

impl AstKind {
    #[inline]
    pub fn is_struct_kind(self) -> bool {
        matches!(self, AstKind::StructDef | AstKind::SliceTy)
    }

    #[inline]
    pub fn is_const_val_kind(self) -> bool {
        ConstVal::KINDS.contains(&self)
    }

    #[inline]
    pub fn is_type_kind(self) -> bool {
        Type::KINDS.contains(&self)
    }
}

impl Ident {
    #[inline]
    pub const fn new(text: Ptr<str>, span: Span) -> Ident {
        ast_new!(Ident { span, text /* decl: None */ })
    }
}

impl Dot {
    pub const fn new(lhs: Option<Ptr<Ast>>, rhs: Ptr<Ident>, span: Span) -> Dot {
        ast_new!(Dot { span, lhs, has_lhs: lhs.is_some(), rhs })
    }
}

impl Decl {
    pub const fn new(ident: Ptr<Ident>, span: Span) -> Decl {
        ast_new!(Decl {
            span,
            is_const: false,
            is_extern: false,
            markers: DeclMarkers::default(),
            ident,
            var_ty_expr: None,
            var_ty: None,
            init: None,
        })
    }

    pub fn from_ident(ident: Ptr<Ident>) -> Decl {
        Decl::new(ident, ident.span)
    }
}

impl Ptr<Decl> {
    pub fn const_val(&self) -> Ptr<Ast> {
        let never = primitives().never;
        if self.ty.u() == never {
            return never.upcast();
        }
        debug_assert!(self.is_const);
        let cv = self.init.u().rep();
        debug_assert!(cv.is_const_val());
        cv
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

    /// used during codegen
    pub fn finalize_arg_type(
        &self,
        lhs_ty: &mut Ptr<Type>,
        rhs_ty: &mut Ptr<Type>,
        out_ty: Ptr<Type>,
    ) {
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
            | BinOpKind::Or => {
                finalize_ty(lhs_ty, out_ty);
                finalize_ty(rhs_ty, out_ty);
            },
            BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge => {
                debug_assert_eq!(out_ty, primitives().bool);
                // debug_assert_eq!(lhs_ty, rhs_ty);
            },
        };
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

impl UnaryOpKind {
    /// used during codegen
    pub fn finalize_arg_type(self, arg_ty: &mut Ptr<Type>, out_ty: Ptr<Type>) {
        match self {
            UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf => {
                let pointee = out_ty.downcast::<PtrTy>().pointee.downcast_type();
                debug_assert!(ty_match(*arg_ty, pointee));
                *arg_ty = pointee;
            },
            UnaryOpKind::Deref => {
                let pointee = arg_ty.downcast_ref::<PtrTy>().pointee.downcast_type_ref();
                debug_assert!(ty_match(*pointee, out_ty));
                *pointee = out_ty;
            },
            UnaryOpKind::Not | UnaryOpKind::Neg => {
                debug_assert!(ty_match(*arg_ty, out_ty));
                *arg_ty = out_ty;
            },
            UnaryOpKind::Try => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeclMarkers {
    pub is_pub: bool,
    pub is_mut: bool,
    pub is_rec: bool,
}

impl DeclMarkers {
    pub const fn default() -> Self {
        Self { is_pub: false, is_mut: false, is_rec: false }
    }

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

pub type DeclList = Ptr<[Ptr<Decl>]>;

pub trait DeclListExt {
    fn find_field(&self, name: &str) -> Option<(usize, Ptr<Decl>)>;

    fn iter_types(&self) -> impl DoubleEndedIterator<Item = Ptr<Type>> + '_;
}

impl DeclListExt for [Ptr<Decl>] {
    fn find_field(&self, name: &str) -> Option<(usize, Ptr<Decl>)> {
        self.iter().copied().enumerate().find(|(_, d)| *d.ident.text == *name)
    }

    fn iter_types(&self) -> impl DoubleEndedIterator<Item = Ptr<Type>> + '_ {
        self.iter().map(|decl| decl.var_ty.u())
    }
}
