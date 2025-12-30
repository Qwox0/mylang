use crate::{
    ast::debug::DebugAst,
    codegen::llvm::finalize_ty,
    context::{FilesIndex, ctx, ctx_mut, primitives},
    diagnostics::{HandledErr, cerror},
    intern_pool::Symbol,
    parser::{lexer::Span, unexpected_expr},
    ptr::{OPtr, Ptr},
    scope::Scope,
    type_::ty_match,
    util::{UnwrapDebug, panic_debug, then, unreachable_debug},
};
use core::fmt;

pub mod debug;

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

inherit_ast! {
    struct Ast {}
}

/// Constructor for ast nodes
macro_rules! ast_new {
    (local $kind:ident { $( $(#[$attr:meta])* $field:ident $( : $val:expr )?),* $(,)? }) => {
        crate::ast::$kind {
            kind: crate::ast::AstKind::$kind,
            ty: None,
            replacement: None,
            parenthesis_count: 0,
            $( $(#[$attr])* $field $(: $val)? ),*
        }
    };
    ($kind:ident { $( $(#[$attr:meta])* $field:ident $( : $val:expr )? ),* $(,)? }) => { {
        let expr = ast_new!(local $kind { $( $(#[$attr])* $field $(:$val)?),* });
        crate::context::ctx().alloc.alloc(expr)?
    } };
    ($kind:ident { $( $(#[$attr:meta])* $field:ident $( : $val:expr )? ),* $(,)? }, $span:expr $(,)? ) => {
        ast_new!($kind { span: $span, $( $(#[$attr])* $field $(:$val)?),* })
    };
}
pub(crate) use ast_new;

macro_rules! type_new {
    (local $kind:ident { $( $field:ident $( : $val:expr )?),* $(,)? }) => {{
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
    ($kind:ident { $( $field:ident $( : $val:expr )?),* $(,)? }) => {
        crate::context::ctx().alloc.alloc(crate::ast::type_new!(local $kind { $($field $(:$val)?),* }))?
    };
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
                $(
                    $(#[$field_attr])*
                    $field : $ty
                ),*
            },)+
            $($c_name {
                ty: OPtr<$crate::ast::Type>,
                replacement: OPtr<Ast>,
                span: Span,
                parenthesis_count: u8,
                $(
                    $(#[$c_field_attr])*
                    $c_field : $c_ty
                ),*
            },)+
            $($t_name {
                ty: OPtr<$crate::ast::Type>,
                replacement: OPtr<Ast>,
                span: Span,
                parenthesis_count: u8,
                $(
                    $(#[$t_field_attr])*
                    $t_field : $t_ty
                ),*
            },)+
        }

        pub enum AstMatch {
            $($name(Ptr<$name>),)+
            $($c_name(Ptr<$c_name>),)+
            $($t_name(Ptr<$t_name>),)+
        }

        impl AstMatch {
            fn match_(ast: Ptr<Ast>) -> Self {
                match ast.kind {
                    $(AstKind::$name => AstMatch::$name(ast.flat_downcast::<$name>()),)+
                    $(AstKind::$c_name => AstMatch::$c_name(ast.flat_downcast::<$c_name>()),)+
                    $(AstKind::$t_name => AstMatch::$t_name(ast.flat_downcast::<$t_name>()),)+
                }
            }
        }

        pub enum TypeMatch {
            $($t_name(Ptr<$t_name>),)+
        }

        impl TypeMatch {
            fn match_(ast: Ptr<Type>) -> Self {
                match ast.kind {
                    $(|AstKind::$name)+
                    $(|AstKind::$c_name)+ => unreachable_debug(),
                    $(AstKind::$t_name => TypeMatch::$t_name(ast.downcast::<$t_name>()),)+
                }
            }
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
            pub const KINDS: &[AstKind] = &[$(AstKind::$c_name,)+ $(AstKind::$t_name,)+];
        }

        impl Type {
            pub const KINDS: &[AstKind] = &[$(AstKind::$t_name,)+];
        }
    };
}

ast_variants! {
    Ident {
        sym: Symbol,
        decl: OPtr<Decl>,
    },

    /// `{ <stmt>* }`
    Block {
        has_trailing_semicolon: bool,
        scope: Scope,
        /// all statements (including declarations) in this block
        stmts: Ptr<[Ptr<Ast>]>,
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
        fields: Ptr<[(Ptr<Ident>, OPtr<Ast>)]>, // TODO: SoA
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
    /// `<lhs> [ <idx> ]`, `<lhs> [ <idx> ]mut`
    /// `              ^` expr.span `      ^^^` expr.span
    Index {
        mut_access: bool,
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
        markers: DeclMarkers,
        ident: Ptr<Ident>,
        /// `MyStruct.abc :: /* ... */;`
        /// `^^^^^^^^`
        on_type: OPtr<Ast>,
        var_ty_expr: OPtr<Ast>,
        var_ty: OPtr<Type>,
        /// also used for default value in fn params, struct fields, ...
        init: OPtr<Ast>,
        obj_symbol_name: OPtr<StrVal>,
    },

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
        iter_var: Ptr<Decl>,
        body: Ptr<Ast>,
        scope: Scope,
    },
    /// `while <cond> <body>`
    While {
        was_piped: bool,
        condition: Ptr<Ast>,
        body: Ptr<Ast>,
        // currently no `Scope` needed. This will change when declarations are allowed in `condition`
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

    Empty {},

    ===== Constant Values =====

    IntVal { val: i64 },
    FloatVal { val: f64 },
    BoolVal { val: bool },
    CharVal { val: char },
    // BCharLit { val: u8 },
    StrVal { text: Ptr<str> }, // TODO?: add string interning?
    PtrVal { val: u64 },
    /// used for constant `struct` values, `union` values, `enum` values and `array` values
    AggregateVal {
        /// Always contains all fields in the same order as defined.
        // `.[val; N]`: store `val` only once?
        elements: Ptr<[Ptr<ConstVal>]>,
    },

    ImportDirective {
        path: Ptr<StrVal>,
        files_idx: FilesIndex,
    },
    ExternDirective {
        /* TODO: library */
        decl: OPtr<Decl>,
    },
    IntrinsicDirective {
        intrinsic_name: Ptr<StrVal>,
        decl: OPtr<Decl>,
    },
    ProgramMainDirective {},
    SimpleDirective {
        ret_ty: Ptr<Type>,
    },

    /// TODO: replace with stdlib functions
    SizeOfDirective { type_: Ptr<Ast> },
    SizeOfValDirective { val: Ptr<Ast> },
    AlignOfDirective { type_: Ptr<Ast> },
    OffsetOfDirective { type_: Ptr<Ast>, field: Ptr<Ident> },

    ===== Types =====

    /// `void`, `never`, `bool`, `type`
    SimpleTy {
        is_finalized: bool,
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
    StructDef {
        /// [`Scope::decls`] only contains the constants defined within the struct body.
        scope: Scope,
        // TODO(size): allocate relative to `scope.decls.ptr`; replace with `field_count`
        fields: Ptr<[Ptr<Decl>]>,
        finished_members: usize,
        /// contains the constants which are also in [`Scope::decls`] plus constants which are
        /// defined later.
        // TODO: don't allocate [`Scope::decls`] twice.
        consts: Vec<Ptr<Decl>>,
    },
    /// `union { a: int, b: String, c: (u8, u32) }`
    UnionDef {
        /// [`Scope::decls`] only contains the constants defined within the struct body.
        scope: Scope,
        // TODO(size): allocate relative to `scope.decls.ptr`; replace with `field_count`
        fields: Ptr<[Ptr<Decl>]>,
        finished_members: usize,
        /// contains the constants which are also in [`Scope::decls`] plus constants which are
        /// defined later.
        // TODO: don't allocate [`Scope::decls`] twice.
        consts: Vec<Ptr<Decl>>,
    },
    /// `enum { A, B(i64) }`
    EnumDef {
        /// simple enum == no associated data
        is_simple_enum: bool,
        /// [`Scope::decls`] only contains the constants defined within the struct body.
        scope: Scope,
        // TODO(size): allocate relative to `scope.decls.ptr`; replace with `variant_count`
        variants: Ptr<[Ptr<Decl>]>,
        finished_members: usize,
        /// is present after sema of this ast node. always has the same length as `variants`.
        variant_tags: OPtr<[isize]>,
        /// contains the constants which are also in [`Scope::decls`] plus constants which are
        /// defined later.
        // TODO: don't allocate [`Scope::decls`] twice.
        consts: Vec<Ptr<Decl>>,
        tag_ty: OPtr<IntTy>,
    },

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
    // Note: for normal functions the following might not be true: `fn.ty.ty == primitives.type_ty`
    Fn {
        /// set during sema
        has_known_ret_ty: bool,
        has_varargs: bool,
        params_scope: Scope,
        ret_ty_expr: OPtr<Ast>,
        ret_ty: OPtr<Type>,
        /// if `body == None` this Ast node originated from a function type. Note: normal functions
        /// are also valid [`Type`]s.
        body: OPtr<Ast>,

        #[cfg(debug_assertions)]
        decl: OPtr<Decl>,
    },

    /// only for type hints
    ArrayLikeContainer {
        elem_ty: Ptr<Type>,
    }
}

inherit_ast! {
    struct ConstVal {}
}

inherit_ast! {
    struct Type {}
}

pub trait UpcastToAst: Sized {
    fn upcast(self: Ptr<Self>) -> Ptr<Ast>;

    #[allow(unused)]
    fn upcast_slice(slice: Ptr<[Ptr<Self>]>) -> Ptr<[Ptr<Ast>]>;

    /// resolve possible replacements of this expression
    fn rep(self: Ptr<Self>) -> Ptr<Ast> {
        self.upcast().rep()
    }

    fn full_span(&self) -> Span {
        Ptr::from_ref(self).upcast().full_span()
    }
}

macro_rules! impl_UpcastToAst {
    ($($name:ty),*) => { $(
        impl UpcastToAst for $name {
            fn upcast(self: Ptr<Self>) -> Ptr<Ast> {
                self.cast()
            }

            fn upcast_slice(slice: Ptr<[Ptr<Self>]>) -> Ptr<[Ptr<Ast>]> {
                slice.cast_slice()
            }
        }
    )* };
}

impl_UpcastToAst! { AstEnum, ConstVal, Type }

impl<V: AstVariant> UpcastToAst for V {
    fn upcast(self: Ptr<V>) -> Ptr<Ast> {
        debug_assert_eq!(self.get_kind(), V::KIND);
        self.cast()
    }

    fn upcast_slice(slice: Ptr<[Ptr<Self>]>) -> Ptr<[Ptr<Ast>]> {
        debug_assert!(slice.iter().all(|a| a.get_kind() == V::KIND));
        slice.cast_slice()
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
        if self.ty.u() == primitives().type_ty {
            debug_assert!(self.rep().has_type_kind(), "expected type kind; got: {:?}", self.kind);
            true
        } else if self.ty.u().p_eq(self) {
            debug_assert_eq!(self.kind, AstKind::Fn);
            true
        } else {
            false
        }
    }

    /// only use this for debugging. otherwise use [`Ptr::is_type`] instead
    pub fn has_type_kind(self) -> bool {
        self.kind.is_type_kind()
    }

    /// resolve possible replacements of this expression
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

    pub fn try_rep(self) -> OPtr<Ast> {
        then!(self.replacement.is_some() => self.rep())
    }

    #[inline]
    pub fn set_replacement(self, rep: Ptr<Ast>) {
        debug_assert!(self.replacement.is_none_or(|r| r == rep));
        //debug_assert!(self.replacement.is_none()); // TODO(without `NotFinished`); use this
        if rep.ty.is_none() {
            rep.as_mut().ty = Some(self.ty.u());
        }
        self.as_mut().replacement = Some(rep)
    }

    #[track_caller]
    pub fn downcast<V: AstVariant>(self) -> Ptr<V> {
        self.rep().flat_downcast()
    }

    #[track_caller]
    pub fn flat_downcast<V: AstVariant>(self) -> Ptr<V> {
        debug_assert_eq!(self.kind, V::KIND);
        self.cast()
    }

    pub fn try_downcast<V: AstVariant>(self) -> OPtr<V> {
        self.rep().try_flat_downcast()
    }

    /// [`Self::try_downcast`] but doesn't resolve replacements first.
    pub fn try_flat_downcast<V: AstVariant>(self) -> OPtr<V> {
        then!(self.kind == V::KIND => self.flat_downcast())
    }

    /// downcast to a [`ConstVal`]
    #[track_caller]
    pub fn downcast_const_val(self) -> Ptr<ConstVal> {
        let p = self.rep();
        debug_assert!(p.is_const_val());
        p.cast()
    }

    pub fn try_downcast_const_val(self) -> OPtr<ConstVal> {
        let p = self.rep();
        then!(p.is_const_val() => p.downcast_const_val())
    }

    pub fn flat_downcast_type(self) -> Ptr<Type> {
        //debug_assert!(self.is_type() || self.kind == AstKind::Fn);
        debug_assert!(self.has_type_kind());
        self.cast()
    }

    pub fn try_flat_downcast_type_by_kind(self) -> OPtr<Type> {
        then!(self.has_type_kind() => self.flat_downcast_type())
    }

    #[inline]
    pub fn downcast_type(self) -> Ptr<Type> {
        self.rep().flat_downcast_type()
    }

    pub fn try_downcast_type(self) -> OPtr<Type> {
        then!(self.is_type() => self.downcast_type())
    }

    pub fn try_downcast_type_by_kind(self) -> OPtr<Type> {
        self.rep().try_flat_downcast_type_by_kind()
    }

    pub fn downcast_type_ref(&mut self) -> &mut Ptr<Type> {
        let p = self.rep_mut();
        debug_assert!(p.is_type());
        Ptr::from_ref(p).cast::<Ptr<Type>>().as_mut()
    }

    pub fn int<Int: TryFrom<i64>>(self) -> Int
    where Int::Error: fmt::Debug {
        let int = self.downcast::<IntVal>().val;
        debug_assert!(Int::try_from(int).is_ok(), "{int}");
        Int::try_from(int).u()
    }

    /// similar to [`Ast::full_span`] but returns a better span for [`Block`] nodes.
    pub fn return_val_span(self) -> Span {
        self.try_downcast::<Block>()
            .and_then(|b| b.stmts.last().copied())
            .unwrap_or(self)
            .full_span()
    }

    /// ```mylang
    /// print :: -> i32 { /* ... */ };
    /// if true then print();
    /// ```
    pub fn can_ignore_yielded_value(self) -> bool {
        self.ty.u().matches_void() || self.kind == AstKind::Call
    }

    pub fn try_to_decl(self) -> Result<OPtr<Decl>, HandledErr> {
        match self.matchable2() {
            AstMatch::Decl(decl) => Ok(Some(decl)),
            AstMatch::Ident(i) => ctx().alloc.alloc(Decl::from_ident(i)).map(Some),
            _ => Ok(None),
        }
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
    /// always behaves like a `flat_downcast`.
    #[track_caller]
    pub fn downcast<V: TypeVariant>(self) -> Ptr<V> {
        debug_assert!(self.replacement.is_none());
        debug_assert_eq!(self.kind, V::KIND, "invalid downcast to {:?}", V::KIND);
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

    pub fn downcast_struct_def(self) -> Ptr<StructDef> {
        match self.kind {
            AstKind::StructDef => self.downcast::<StructDef>(),
            AstKind::SliceTy => primitives().untyped_slice_struct_def,
            _ => unreachable_debug(),
        }
    }

    pub fn try_downcast_struct_def(self) -> OPtr<StructDef> {
        debug_assert!(self.replacement.is_none());
        then!(self.kind.is_struct_kind() => self.downcast_struct_def())
    }

    pub fn is_int_lit(self) -> bool {
        let p = primitives();
        self == p.int_lit || self == p.sint_lit
    }

    pub fn is_sint(self) -> bool {
        self.try_downcast::<IntTy>().is_some_and(|i| i.is_signed)
    }

    /// Some types (like pointers) are transparent and allow field/method access on its inner type.
    pub fn flatten_transparent(mut self) -> Ptr<Type> {
        loop {
            match self.matchable().as_ref() {
                TypeEnum::SimpleTy { .. }
                | TypeEnum::IntTy { .. }
                | TypeEnum::FloatTy { .. }
                | TypeEnum::ArrayTy { .. }
                | TypeEnum::StructDef { .. }
                | TypeEnum::UnionDef { .. }
                | TypeEnum::EnumDef { .. }
                | TypeEnum::RangeTy { .. }
                | TypeEnum::OptionTy { .. }
                | TypeEnum::Fn { .. }
                | TypeEnum::SliceTy { .. } => break self,
                TypeEnum::PtrTy { pointee, .. } => self = pointee.downcast_type(),
                TypeEnum::ArrayLikeContainer { .. } | TypeEnum::Unset => {
                    panic_debug!("invalid type")
                },
            }
        }
    }

    /// Returns any kind of sub type.
    /// Useful to guess type inference when a mismatch occurs and reduce unnecessary "cannot infer"
    /// errors.
    pub fn inner_ty(self) -> OPtr<Type> {
        match self.matchable2() {
            TypeMatch::SimpleTy(_) | TypeMatch::IntTy(_) | TypeMatch::FloatTy(_) => None,
            TypeMatch::PtrTy(p) => Some(p.pointee.downcast_type()),
            TypeMatch::SliceTy(s) => Some(s.elem_ty.downcast_type()),
            TypeMatch::ArrayTy(a) => Some(a.elem_ty.downcast_type()),
            TypeMatch::StructDef(_) | TypeMatch::UnionDef(_) | TypeMatch::EnumDef(_) => None,
            TypeMatch::RangeTy(r) => Some(r.elem_ty),
            TypeMatch::OptionTy(o) => Some(o.inner_ty.downcast_type()),
            TypeMatch::Fn(_) => None,
            TypeMatch::ArrayLikeContainer(a) => Some(a.elem_ty),
        }
    }

    pub fn try_downcast_ty_hint<V: TypeVariant>(self) -> OPtr<V> {
        self.try_downcast::<V>().or_else(|| self.inner_ty().try_downcast::<V>())
    }
}

impl Ast {
    /// Convert the ast node into a matchable rust enum
    #[inline]
    pub fn matchable(&self) -> Ptr<AstEnum> {
        Ptr::<Ast>::from_ref(self).cast::<AstEnum>()
    }

    /// doesn't handle `replacements`
    #[inline]
    pub fn matchable2(&self) -> AstMatch {
        AstMatch::match_(Ptr::<Ast>::from_ref(self))
    }

    pub(crate) fn block_expects_trailing_sep(&self) -> bool {
        match self.matchable().as_ref() {
            AstEnum::Block { .. } => false,
            AstEnum::Decl { init, is_const, .. } => {
                if !is_const {
                    return true;
                }
                let Some(init) = init else { return true };
                match init.matchable2() {
                    AstMatch::StructDef(_) | AstMatch::UnionDef(_) | AstMatch::EnumDef(_) => false,
                    AstMatch::Fn(f) => {
                        f.body.unwrap_or_else(|| f.ret_ty_expr.u()).kind != AstKind::Block
                    },
                    _ => true,
                }
            },
            &AstEnum::If { then_body, else_body, .. } => {
                else_body.unwrap_or(then_body).block_expects_trailing_sep()
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { body, .. } | AstEnum::While { body, .. } => {
                body.block_expects_trailing_sep()
            },
            AstEnum::Empty { .. } => false,
            _ => true,
        }
    }

    /// Returns a [`Span`] representing the entire expression.
    pub fn full_span(&self) -> Span {
        let span = self.span;
        let full_span = match self.matchable().as_ref() {
            AstEnum::PositionalInitializer { lhs, .. }
            | AstEnum::NamedInitializer { lhs, .. }
            | AstEnum::ArrayInitializer { lhs, .. }
            | AstEnum::ArrayInitializerShort { lhs, .. } => {
                span.maybe_join(lhs.map(|e| e.full_span()))
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
            AstEnum::Range { start, end, .. } => span
                .maybe_join(start.map(|s| s.full_span()))
                .maybe_join(end.map(|s| s.full_span())),
            AstEnum::Decl { init, .. } => match &init {
                Some(e) => span.join(e.full_span()),
                None => span,
            },
            AstEnum::If { condition, then_body, else_body, was_piped, .. } => {
                let r_span = else_body.unwrap_or(*then_body).full_span();
                if *was_piped { condition.full_span() } else { span }.join(r_span)
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { source: l, body, was_piped, .. }
            | AstEnum::While { condition: l, body, was_piped, .. } => {
                if *was_piped { l.full_span() } else { span }.join(body.full_span())
            },
            // AstEnum::Catch { .. } => todo!(),
            AstEnum::Defer { stmt, .. } => span.join(stmt.full_span()),
            AstEnum::Return { val, .. } => match val {
                Some(val) => span.join(val.full_span()),
                None => span,
            },
            AstEnum::ImportDirective { path, .. } => span.join(path.span),
            AstEnum::ExternDirective { .. } => span,
            AstEnum::IntrinsicDirective { intrinsic_name, .. } => span.join(intrinsic_name.span),

            AstEnum::SizeOfDirective { type_: e, .. }
            | AstEnum::SizeOfValDirective { val: e, .. }
            | AstEnum::AlignOfDirective { type_: e, .. } => span.join(e.full_span()),
            AstEnum::OffsetOfDirective { field, .. } => span.join(field.span),

            AstEnum::SimpleTy { .. } | AstEnum::IntTy { .. } | AstEnum::FloatTy { .. } => span,
            AstEnum::PtrTy { pointee: i, .. }
            | AstEnum::SliceTy { elem_ty: i, .. }
            | AstEnum::ArrayTy { elem_ty: i, .. }
            | AstEnum::OptionTy { inner_ty: i, .. } => span.join(i.full_span()),
            AstEnum::StructDef { .. } | AstEnum::UnionDef { .. } | AstEnum::EnumDef { .. } => span,
            AstEnum::RangeTy { .. } => todo!(),
            AstEnum::Fn { params_scope, body, ret_ty_expr, .. } => span
                .maybe_join(params_scope.decls.get(0).map(|p| {
                    Some(p.ident.span)
                        .filter(|s| *s != Span::ZERO)
                        .unwrap_or_else(|| p.var_ty_expr.u().full_span()) // for special case: `i32 -> i32`
                }))
                .join(body.or(*ret_ty_expr).u().full_span()),
            _ => span,
        };
        if self.parenthesis_count > 0
            && let Some(file) = full_span.file
        {
            let start = file.code.0[..full_span.start].rfind('(').u();
            let end = full_span.end + 1 + file.code.0[full_span.end..].find(')').u();
            return Span::new(start..end, Some(file));
        }
        full_span
    }

    pub fn is_custom_type(&self) -> bool {
        matches!(self.kind, AstKind::StructDef | AstKind::UnionDef | AstKind::EnumDef) // TODO: add `| AstKind::Fn`?
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

    #[inline]
    pub fn matchable2(&self) -> TypeMatch {
        TypeMatch::match_(Ptr::<Type>::from_ref(self))
    }

    pub fn get_arr_elem_ty(&self) -> Ptr<Type> {
        match self.matchable().as_ref() {
            TypeEnum::ArrayTy { elem_ty, .. } | TypeEnum::SliceTy { elem_ty, .. } => {
                elem_ty.downcast_type()
            },
            _ => unreachable_debug(),
        }
    }

    pub fn get_arr_elem_ty_mut(&mut self) -> &mut Ptr<Type> {
        match self.matchable().as_mut() {
            TypeEnum::ArrayTy { elem_ty, .. } | TypeEnum::SliceTy { elem_ty, .. } => {
                elem_ty.downcast_type_ref()
            },
            _ => unreachable_debug(),
        }
    }

    /// For custom types this returns the constants defined inside the scope of the type.
    pub fn get_scope(&self) -> Option<&Scope> {
        match self.matchable().as_ref() {
            TypeEnum::StructDef { scope, .. }
            | TypeEnum::UnionDef { scope, .. }
            | TypeEnum::EnumDef { scope, .. } => Some(scope),
            _ => None,
        }
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

pub trait OptionTypeExt {
    fn matchable(self) -> Ptr<TypeEnum>;
    fn downcast<V: TypeVariant>(self) -> Ptr<V>;
    fn try_downcast<V: TypeVariant>(self) -> OPtr<V>;
}

impl OptionTypeExt for OPtr<Type> {
    #[inline]
    fn matchable(self) -> Ptr<TypeEnum> {
        match self {
            Some(t) => t.matchable(),
            None => Ptr::from_ref(&TypeEnum::Unset),
        }
    }

    #[track_caller]
    #[inline]
    fn downcast<V: TypeVariant>(self) -> Ptr<V> {
        self.u().downcast()
    }

    #[inline]
    fn try_downcast<V: TypeVariant>(self) -> OPtr<V> {
        self?.try_downcast()
    }
}

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

    pub fn is_allowed_top_level(self) -> bool {
        matches!(
            self,
            AstKind::Decl | AstKind::Empty | AstKind::SimpleDirective | AstKind::ImportDirective
        )
    }

    pub fn initializer_kind(self) -> &'static str {
        match self {
            AstKind::PositionalInitializer => "a positional initializer",
            AstKind::NamedInitializer => "a named initializer",
            AstKind::ArrayInitializer | AstKind::ArrayInitializerShort => "an array initializer",
            k => panic_debug!("{k:?} is not an initializer kind"),
        }
    }
}

impl Ident {
    #[inline]
    pub fn new(text: Ptr<str>, span: Span) -> Ident {
        ast_new!(local Ident { span, sym: ctx_mut().symbols.get_or_intern(text), decl: None })
    }
}

impl Dot {
    pub const fn new(lhs: Option<Ptr<Ast>>, rhs: Ptr<Ident>, span: Span) -> Dot {
        ast_new!(local Dot { span, lhs, has_lhs: lhs.is_some(), rhs })
    }
}

impl Decl {
    pub const fn new(ident: Ptr<Ident>, associated_type_expr: OPtr<Ast>, span: Span) -> Decl {
        ast_new!(local Decl {
            span,
            is_const: false,
            markers: DeclMarkers::default(),
            ident,
            on_type: associated_type_expr,
            var_ty_expr: None,
            var_ty: None,
            init: None,
            obj_symbol_name: None,
        })
    }

    pub fn from_ident(ident: Ptr<Ident>) -> Decl {
        Decl::new(ident, None, ident.span)
    }

    /// `MyStruct.ABC : u8 : /* ... */`
    /// `^^^^^^^^^^^^ lhs`
    pub fn from_lhs(lhs: Ptr<Ast>) -> Result<Decl, HandledErr> {
        match lhs.matchable2() {
            AstMatch::Ident(lhs) => Ok(Decl::from_ident(lhs)),
            AstMatch::Dot(dot) => match dot.lhs {
                Some(ty_expr) => Ok(Decl::new(dot.rhs, Some(ty_expr), lhs.full_span())),
                None => {
                    Err(cerror!(dot.span, "A member declaration requires an associated type name"))
                },
            },
            _ => Err(unexpected_expr(lhs, "a variable name")),
        }
    }

    pub fn is_lhs_only(&self) -> bool {
        self.var_ty_expr.is_none() && self.init.is_none()
    }

    pub fn const_val(self: Ptr<Decl>) -> Ptr<Ast> {
        debug_assert!(self.is_const);
        debug_assert!(self.var_ty.is_some() || self.init.u().kind == AstKind::Fn);
        if let Some(t) = self.var_ty
            && t.propagates_out()
        {
            return t.upcast();
        }
        debug_assert!(self.is_const);
        self.init.u().downcast_const_val().upcast()
    }

    pub fn try_const_val(self: Ptr<Decl>) -> OPtr<Ast> {
        then!(self.is_const => self.const_val())
    }

    pub fn lhs_span(&self) -> Span {
        let name_span = self.ident.span;
        name_span.maybe_join(self.on_type.map(|t| t.full_span()))
    }

    pub fn display_lhs(&self) -> impl std::fmt::Display {
        struct DeclLhsDisplay {
            on_type: OPtr<Ast>,
            ident: Ptr<Ident>,
        }

        impl std::fmt::Display for DeclLhsDisplay {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if let Some(ty) = self.on_type {
                    write!(f, "{}.", ty.to_text(false))?;
                }
                write!(f, "{}", self.ident.sym)
            }
        }

        DeclLhsDisplay { on_type: self.on_type, ident: self.ident }
    }

    #[inline]
    pub fn might_need_precompilation(&self) -> bool {
        self.is_const || self.markers.get(DeclMarkers::IS_STATIC_MASK)
    }
}

impl Block {
    pub fn new(
        stmts: Ptr<[Ptr<Ast>]>,
        scope: Scope,
        has_trailing_semicolon: bool,
        span: Span,
    ) -> Self {
        ast_new!(local Block { span, has_trailing_semicolon, stmts, scope })
    }

    pub fn new_anon(stmts: Ptr<[Ptr<Ast>]>, scope: Scope) -> Self {
        Block::new(stmts, scope, false, Span::ZERO)
    }
}

impl EnumDef {
    #[cfg(debug_assertions)]
    pub fn find_variant_ty_for_tag(&self, tag_val: isize) -> Ptr<Type> {
        let idx = self.variant_tags.u().iter().position(|tag| *tag == tag_val).u();
        self.variants[idx].var_ty.u()
    }
}

impl Fn {
    #[inline]
    pub fn params(&self) -> DeclList {
        self.params_scope.decls
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

    /// `&&`, `&&=`, `and`, `and=`
    And,

    /// `||`, `||=`, `or`, `or=`
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
    /// `<expr>.*`
    ///
    /// `* <expr>` is currently not implemented because it is very similar to [`PtrTy`] and makes
    /// parsing annoying
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
                if pointee != primitives().any {
                    *arg_ty = pointee;
                } else {
                    arg_ty.finalize();
                }
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

    pub fn to_text(self) -> &'static str {
        match self {
            UnaryOpKind::AddrOf => "&",
            UnaryOpKind::AddrMutOf => "&mut",
            UnaryOpKind::Deref => ".*",
            UnaryOpKind::Not => "!",
            UnaryOpKind::Neg => "-",
            UnaryOpKind::Try => "?",
        }
    }
}

impl fmt::Display for UnaryOpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeKind {
    /// `..`
    Full,
    /// `start..`
    From,
    /// `..end`
    To,
    /// `..=end`
    ToInclusive,
    /// `start..end`
    Both,
    /// `start..=end`
    BothInclusive,
}

impl RangeKind {
    pub fn get_field_count(self) -> usize {
        match self {
            RangeKind::Full => 0,
            RangeKind::From | RangeKind::To | RangeKind::ToInclusive => 1,
            RangeKind::Both | RangeKind::BothInclusive => 2,
        }
    }

    pub fn has_start(self) -> bool {
        match self {
            RangeKind::Full | RangeKind::To | RangeKind::ToInclusive => false,
            RangeKind::From | RangeKind::Both | RangeKind::BothInclusive => true,
        }
    }

    pub fn has_end(self) -> bool {
        match self {
            RangeKind::Full | RangeKind::From => false,
            RangeKind::To | RangeKind::ToInclusive | RangeKind::Both | RangeKind::BothInclusive => {
                true
            },
        }
    }

    pub fn is_inclusive(self) -> bool {
        match self {
            RangeKind::Full | RangeKind::From | RangeKind::To | RangeKind::Both => false,
            RangeKind::ToInclusive | RangeKind::BothInclusive => true,
        }
    }

    pub fn type_name(self) -> &'static str {
        match self {
            RangeKind::Full => "RangeFull",
            RangeKind::From => "RangeFrom",
            RangeKind::To => "RangeTo",
            RangeKind::ToInclusive => "RangeToInclusive",
            RangeKind::Both => "Range",
            RangeKind::BothInclusive => "RangeInclusive",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeclMarkers {
    data: u8,
}

impl DeclMarkers {
    pub const IS_MUT_MASK: u8 = 0x1;
    pub const IS_PUB_MASK: u8 = 0x4;
    pub const IS_REC_MASK: u8 = 0x2;
    pub const IS_STATIC_MASK: u8 = 0x8;

    pub const fn default() -> Self {
        Self { data: 0 }
    }

    pub fn get(&self, mask: u8) -> bool {
        self.data & mask != 0
    }

    pub fn set(&mut self, mask: u8) {
        self.data |= mask;
    }
}

pub type DeclList = Ptr<[Ptr<Decl>]>;

pub trait DeclListExt {
    fn find_field(&self, sym: Symbol) -> Option<(usize, Ptr<Decl>)>;

    fn iter_types(&self) -> impl ExactSizeIterator<Item = Ptr<Type>> + '_;
}

impl DeclListExt for [Ptr<Decl>] {
    fn find_field(&self, sym: Symbol) -> Option<(usize, Ptr<Decl>)> {
        self.iter().copied().enumerate().find(|(_, d)| d.ident.sym == sym)
    }

    fn iter_types(&self) -> impl ExactSizeIterator<Item = Ptr<Type>> + '_ {
        self.iter().map(|decl| decl.var_ty.u())
    }
}

pub fn is_pos_arg(a: &Ptr<Ast>) -> bool {
    a.kind != AstKind::Assign || a.parenthesis_count > 0
}
