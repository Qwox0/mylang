use crate::{
    arena_allocator::{AllocErr, Arena},
    ast::debug::DebugAst,
    context::{FilesIndex, ctx, primitives},
    diagnostics::{HandledErr, cerror2, common::error_cannot_infer_generics},
    intern_pool::Symbol,
    parser::{ParseResult, lexer::Span, unexpected_expr},
    ptr::{OPtr, Ptr},
    scope::{Scope, ScopeAndAggregateInfo, ScopeKind},
    scratch_allocator::TmpPtr,
    sema::SemaUnit,
    type_::{finalize_ty, ty_match},
    util::{
        BitFlags, OptionExt, UnwrapDebug, bitflags, panic_debug, then, to_f64, unreachable_debug,
    },
};
use core::fmt;
use num::BigInt;
use std::iter;

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
pub(crate) use inherit_ast;

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
    ($alloc:expr, $kind:ident { $( $(#[$attr:meta])* $field:ident $( : $val:expr )? ),* $(,)? }) => { {
        $alloc.alloc(ast_new!(local $kind { $( $(#[$attr])* $field $(:$val)?),* }))?
    } };
    ($kind:ident { $( $(#[$attr:meta])* $field:ident $( : $val:expr )? ),* $(,)? }) => { {
        ast_new!(crate::context::ctx().alloc, $kind { $( $(#[$attr])* $field $(:$val)?),* })
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
    ($kind:ident { $( $field:ident $( : $val:expr )?),* $(,)? }, $alloc:expr) => {
        $alloc.alloc(crate::ast::type_new!(local $kind { $($field $(:$val)?),* })).expect("TODO: handle oom")
    };
    ($kind:ident { $( $field:ident $( : $val:expr )?),* $(,)? }) => {
        crate::ast::type_new!($kind { $($field $(:$val)?),* }, crate::context::ctx().alloc)
    };
}
pub(crate) use type_new;

pub trait HasAstKind {
    fn get_kind(&self) -> AstKind;
}

pub unsafe trait AstVariant: HasAstKind + core::fmt::Debug {
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

        pub enum ConstValMatch {
            $($c_name(Ptr<$c_name>),)+
            $($t_name(Ptr<$t_name>),)+
        }

        impl ConstVal {
            pub fn matchable2(self: Ptr<ConstVal>) -> ConstValMatch {
                match self.kind {
                    $(|AstKind::$name)+ => unreachable_debug(),
                    $(AstKind::$c_name => ConstValMatch::$c_name(self.flat_downcast::<$c_name>()),)+
                    $(AstKind::$t_name => ConstValMatch::$t_name(self.flat_downcast::<$t_name>()),)+
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
                    $(AstKind::$t_name => TypeMatch::$t_name(ast.flat_downcast::<$t_name>()),)+
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
        /// all statements (including declarations) in this block
        stmts: Ptr<[Ptr<Ast>]>,
        /// only contains the currently visible declarations. Shadowed decls are replaced.
        decl_scope: Scope,
        finished: usize,
    },

    /// `alloc(MyStruct).( a, b, c = <expr>, )`
    /// `               ^^^^^^^^^^^^^^^^^^^^^^` expr.span
    ///
    /// [`Type`] -> value
    /// `*T` -> `*T`
    PositionalInitializer {
        parsed_with_lhs: bool,
        lhs: OPtr<Ast>,
        args: Ptr<[Ptr<Ast>]>,

        /// can be [`StructDef`] or [`SliceTy`]
        resolved_struct_inst: OPtr<Type>,
    },
    /// `alloc(MyStruct).{ a = <expr>, b, }`
    /// `               ^^^^^^^^^^^^^^^^^^^` expr.span
    ///
    /// [`Type`] -> value
    /// `*T` -> `*T`
    NamedInitializer {
        parsed_with_lhs: bool,
        lhs: OPtr<Ast>,
        fields: Ptr<[(Ptr<Ident>, OPtr<Ast>)]>, // TODO: SoA

        /// can be [`StructDef`] or [`SliceTy`]
        resolved_struct_inst: OPtr<Type>,
    },
    /// `alloc(MyArray).[<expr>, <expr>, ..., <expr>,]`
    ArrayInitializer {
        parsed_with_lhs: bool,
        lhs: OPtr<Ast>,
        elements: Ptr<[Ptr<Ast>]>,
    },
    /// `alloc(MyArray).[<expr>; <count>]`
    ArrayInitializerShort {
        parsed_with_lhs: bool,
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
        resolved_fn_inst: OPtr<Fn>,
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
    /// `<source> orelse <default>`, `<source> ?? <default>`
    OrElse {
        lhs: Ptr<Ast>,
        rhs: Ptr<Ast>,
    },

    /// `<lhs> = <lhs>`
    Assign {
        is_explicit_generic_arg: bool,
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
        // TODO: move this into flags
        is_const: bool,
        flags: DeclFlags,
        has_init_expr: bool,
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
    /// `switch <val> <body>` (`else <else>`)
    Switch {
        was_piped: bool,
        val: Ptr<Ast>,
        cases: Ptr<[SwitchCase]>,
        else_body: OPtr<Ast>,
    },

    /// `for <iter_var> in <source> <body>`
    /// `<source> | for <iter_var> <body>`
    For {
        was_piped: bool,
        source_expr: Ptr<Ast>,
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
    Loop {
        body: Ptr<Ast>,
        break_ty: OPtr<Type>,
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

    IntVal { val: BigInt },
    FloatVal { val: f64 }, // currently the biggest supported float type is `f64` => BigFloat is not needed
    BoolVal { val: bool },
    CharVal { val: char },
    // BCharLit { val: u8 },
    StrVal { text: Ptr<str> }, // TODO?: add string interning?

    RawPtrVal { val: u64 },
    StaticPtrVal { sym: Ptr<Decl> },

    /// used for unfinished `{enum_variant}`s and finished enum values
    EnumVal {
        /// ```mylang
        /// E :: enum { A(i32), B }
        /// E.A;    // is_valid == false
        /// E.A(1); // is_valid == true
        /// E.B;    // is_valid == true
        /// ```
        is_valid: bool,
        enum_ty: Ptr<EnumDef>,
        variant_idx: usize,
        data: OPtr<ConstVal>,
    },
    OptionalVal {
        /// ```mylang
        /// Some;    // is_some == true; val == None
        /// Some(1); // is_some == true; val == Some(...)
        /// null;    // is_some == false; val == None
        /// ```
        is_some: bool,
        val: OPtr<ConstVal>
    },
    /// used for constant `struct` values, `union` values, `enum` values and `array` values
    AggregateVal {
        /// Always contains all fields in the same order as defined.
        // `.[val; N]`: store `val` only once?
        elements: Ptr<[Ptr<ConstVal>]>,
    },

    ImportDirective {
        path: Ptr<StrVal>,
        files_idx: Option<FilesIndex>,
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
        bits: Option<u32>,
    },
    FloatTy {
        bits: Option<u32>,
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
        flags: StructFlags,

        /// [`Scope::decls`] only contains the constants defined within the struct body.
        scope: Scope,
        // TODO(size): allocate relative to `scope.decls.ptr`; replace with `field_count`
        fields: Ptr<[Ptr<Decl>]>,
        generics_scope: OPtr<Scope>,

        external_consts: Vec<Ptr<Decl>>,

        instantiations: Vec<Ptr<StructDef>>,

        /// only valid during sema
        sema_units: Option<TmpPtr<[SemaUnit]>>,
        finished_members: usize,
    },
    /// `union { a: int, b: String, c: (u8, u32) }`
    UnionDef {
        /// [`Scope::decls`] only contains the constants defined within the struct body.
        scope: Scope,
        // TODO(size): allocate relative to `scope.decls.ptr`; replace with `field_count`
        fields: Ptr<[Ptr<Decl>]>,

        external_consts: Vec<Ptr<Decl>>,

        /// only valid during sema
        sema_units: Option<TmpPtr<[SemaUnit]>>,
        finished_members: usize,
    },
    /// `enum { A, B(i64) }`
    EnumDef {
        flags: EnumFlags,

        /// simple enum == no associated data
        is_simple_enum: bool,
        /// [`Scope::decls`] only contains the constants defined within the struct body.
        scope: Scope,
        // TODO(size): allocate relative to `scope.decls.ptr`; replace with `variant_count`
        variants: Ptr<[Ptr<Decl>]>,
        generics_scope: OPtr<Scope>,

        external_consts: Vec<Ptr<Decl>>,
        tag_ty: OPtr<IntTy>,

        instantiations: Vec<Ptr<EnumDef>>,

        /// only valid during sema
        sema_units: Option<TmpPtr<[SemaUnit]>>,
        finished_members: usize,
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
        flags: FnFlags,

        /// params_scope.decls:
        ///     `0..param_count`: parameters, including const parameters
        ///     `param_count..` : template generics
        params_scope: Scope,
        generics_scope: OPtr<Scope>,

        ret_ty_expr: OPtr<Ast>,
        ret_ty: OPtr<Type>,

        /// clones of the function
        instantiations: Vec<Ptr<Fn>>,

        /// if `body == None` this Ast node originated from a function type. Note: normal functions
        /// are also valid [`Type`]s.
        body: OPtr<Ast>,

        #[cfg(debug_assertions)]
        decl: OPtr<Decl>,
    },

    GenericDef {
        name: Ptr<Ident>,

        /// Index into [`Fn::generics`]
        idx: usize,

        cur_inst: OPtr<ConstVal>,

        // /// `[$N]$T`
        // ///      ^^ ty=type
        // ///   ^^ ty=uint
        // // TODO?: move to contraints
        // var_ty: OPtr<Type>,
        // /// expression which must be checked for every generic instantiation.
        // contraints: Vec<Ptr<Ast>>,
    },

    /// only for type hints
    ArrayLikeContainer {
        elem_ty: Ptr<Type>,
    },

}

inherit_ast! {
    struct ConstVal {}
}

inherit_ast! {
    struct Type {}
}

pub unsafe trait UpcastToAst: Sized {
    #[cfg(debug_assertions)]
    fn verify_upcast(self: Ptr<Self>) {}

    fn upcast(self: Ptr<Self>) -> Ptr<Ast> {
        #[cfg(debug_assertions)]
        self.verify_upcast();

        self.cast()
    }

    fn upcast_ref(self: &mut Ptr<Self>) -> &mut Ptr<Ast> {
        Ptr::from_ref(self).cast::<Ptr<Ast>>().as_mut()
    }

    #[allow(unused)]
    fn upcast_slice(slice: Ptr<[Ptr<Self>]>) -> Ptr<[Ptr<Ast>]> {
        #[cfg(debug_assertions)]
        slice.iter().copied().for_each(Self::verify_upcast);

        slice.cast_slice()
    }

    /// resolve possible replacements of this expression
    fn rep(self: Ptr<Self>) -> Ptr<Ast> {
        self.upcast().rep()
    }

    fn full_span(&self) -> Span {
        Ptr::from_ref(self).upcast().full_span()
    }
}

unsafe impl UpcastToAst for AstEnum {}
unsafe impl UpcastToAst for ConstVal {}
unsafe impl UpcastToAst for Type {}
unsafe impl<V: AstVariant> UpcastToAst for V {
    #[cfg(debug_assertions)]
    fn verify_upcast(self: Ptr<Self>) {
        debug_assert_eq!(self.get_kind(), V::KIND);
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
        let p = primitives();
        if self.ty.u() == p.type_ty {
            debug_assert!(self.rep().has_type_kind(), "expected type kind; got: {:?}", self.kind);
            true
        } else if [AstKind::Fn, AstKind::GenericDef].contains(&self.rep().kind) {
            debug_assert!(self.ty.u().p_eq(self));
            true
        } else if self == p.err_ty.upcast() {
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
        self.set_replacement_no_type(rep);
        debug_assert!(self.replacement.is_none_or(|r| r == rep));
        if rep.ty.is_none() {
            rep.as_mut().ty = Some(self.ty.u());
        }
    }

    #[inline]
    pub fn set_replacement_no_type(self, rep: Ptr<Ast>) {
        debug_assert!(self.replacement.is_none_or(|r| r == rep));
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
        debug_assert!(p.kind.is_const_val_kind());
        p.cast()
    }

    pub fn try_downcast_const_val(self) -> OPtr<ConstVal> {
        let p = self.rep();
        then!(p.is_const_val() => p.downcast_const_val())
    }

    pub fn flat_downcast_type(self) -> Ptr<Type> {
        debug_assert!(
            self.ty.is_none_or(|t| t == primitives().type_ty
                || (t.ty == t && matches!(t.kind, AstKind::Fn | AstKind::GenericDef))),
            "`{}`",
            self.ty.display()
        );
        //debug_assert!(self.is_type() || self.kind == AstKind::Fn);
        debug_assert!(self.has_type_kind(), "expected type kind, got {:?}", self.kind);
        self.cast()
    }

    pub fn try_flat_downcast_type_by_kind(self) -> OPtr<Type> {
        then!(self.has_type_kind() => self.flat_downcast_type())
    }

    /// allows generic type definitions
    #[inline]
    pub fn downcast_type2(self) -> Ptr<Type> {
        self.rep().flat_downcast_type()
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn downcast_type(self) -> Ptr<Type> {
        let ty = self.downcast_type2();

        #[cfg(debug_assertions)]
        if let Some(polymorphable) = self.try_downcast_polymorphable()
            && polymorphable.is_generic()
        {
            crate::diagnostics::cwarn!(
                self.span,
                "Called downcast_type on generic type `{ty}`\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }

        ty
    }

    /// returns [`error_cannot_infer_generics2`] for generic type definitions
    pub fn downcast_type_inst(self) -> Result<Ptr<Type>, HandledErr> {
        let ty = self.downcast_type2();

        if let Some(polymorphable) = ty.try_downcast_polymorphable()
            && polymorphable.is_generic()
        {
            return Err(error_cannot_infer_generics(self));
        }

        Ok(ty)
    }

    pub fn try_downcast_type(self) -> OPtr<Type> {
        then!(self.is_type() => self.downcast_type())
    }

    pub fn try_downcast_type2(self) -> OPtr<Type> {
        then!(self.is_type() => self.downcast_type2())
    }

    pub fn try_downcast_type_by_kind(self) -> OPtr<Type> {
        self.rep().try_flat_downcast_type_by_kind()
    }

    pub fn downcast_type_ref(&mut self) -> &mut Ptr<Type> {
        let p = self.rep_mut();
        debug_assert!(p.is_type());
        Ptr::from_ref(p).cast::<Ptr<Type>>().as_mut()
    }

    pub fn try_downcast_struct_def(self) -> OPtr<StructDef> {
        self.try_downcast_type2()?.try_downcast_struct_def()
    }

    #[track_caller]
    pub fn int<Int: TryFrom<&'static BigInt>>(self) -> Int
    where Int::Error: fmt::Debug {
        let int = &self.downcast::<IntVal>().as_ref().val;
        Int::try_from(int).u()
    }

    /// similar to [`Ast::full_span`] but returns a better span for [`Block`] nodes.
    pub fn return_val_span(self) -> Span {
        let mut expr = self;
        while let Some(block) = expr.try_downcast::<Block>() {
            match block.stmts.last().copied() {
                Some(s) => expr = s,
                None => break,
            }
        }
        expr.full_span()
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

    pub fn try_get_symbol_decl(self) -> OPtr<Decl> {
        match self.matchable2() {
            AstMatch::Ident(i) => Some(i.decl.u()),
            AstMatch::Dot(d) => Some(d.rhs.decl.u()),
            _ => None,
        }
    }
}

impl Ptr<ConstVal> {
    #[inline]
    pub fn matchable(&self) -> Ptr<ConstValEnum> {
        self.cast()
    }

    #[track_caller]
    pub fn downcast<V: ConstValVariant>(self) -> Ptr<V> {
        debug_assert!(self.replacement.is_none());
        self.flat_downcast()
    }

    #[track_caller]
    pub fn flat_downcast<V: ConstValVariant>(self) -> Ptr<V> {
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

    pub fn try_downcast_type_ref(&mut self) -> Option<&mut Ptr<Type>> {
        debug_assert!(self.replacement.is_none());
        then!(self.upcast().has_type_kind() => self.upcast_ref().downcast_type_ref())
    }

    /// Expects `self` to be an [`IntVal`] or a [`FloatVal`].
    pub fn float_val(self) -> f64 {
        if let Some(int) = self.try_downcast::<IntVal>() {
            to_f64(&int.val)
        } else {
            self.downcast::<FloatVal>().val
        }
    }

    pub fn finalize(&mut self) -> Self {
        if let Some(ty) = self.try_downcast_type_ref() {
            ty.finalize();
        }
        *self
    }
}

impl Ptr<Type> {
    pub fn upcast_to_const_val(self) -> Ptr<ConstVal> {
        self.upcast().downcast_const_val()
    }

    /// always behaves like a `flat_downcast`.
    #[track_caller]
    pub fn downcast<V: TypeVariant>(self) -> Ptr<V> {
        debug_assert_eq!(self.kind, V::KIND, "invalid downcast to {:?}", V::KIND);
        debug_assert!(self.upcast().has_type_kind());
        self.cast()
    }

    pub fn flat_downcast<V: TypeVariant>(self) -> Ptr<V> {
        //debug_assert!(self.replacement.is_none());
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
        then!(self.kind.is_struct_kind() => self.downcast_struct_def())
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
                TypeEnum::GenericDef { .. } => todo!("Generic"),
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
            TypeMatch::GenericDef(g) => {
                g.cur_inst.and_then(Ptr::<ConstVal>::try_downcast_type).and_then(Ptr::inner_ty)
            },
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
            AstEnum::Switch { else_body, .. } => {
                else_body.as_deref().map(Ast::block_expects_trailing_sep).unwrap_or(false)
            },
            AstEnum::For { body, .. }
            | AstEnum::While { body, .. }
            | AstEnum::Loop { body, .. } => body.block_expects_trailing_sep(),
            AstEnum::Empty { .. } => false,
            _ => true,
        }
    }

    /// Returns a [`Span`] representing the entire expression.
    pub fn full_span(&self) -> Span {
        let span = self.span;
        let full_span = match self.matchable().as_ref() {
            AstEnum::PositionalInitializer { lhs, parsed_with_lhs, .. }
            | AstEnum::NamedInitializer { lhs, parsed_with_lhs, .. }
            | AstEnum::ArrayInitializer { lhs, parsed_with_lhs, .. }
            | AstEnum::ArrayInitializerShort { lhs, parsed_with_lhs, .. } => {
                span.maybe_join(lhs.filter(|_| *parsed_with_lhs).map(|e| e.full_span()))
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
            | AstEnum::BinOpAssign { lhs, rhs, .. }
            | AstEnum::OrElse { lhs, rhs, .. } => lhs.full_span().join(rhs.full_span()),
            AstEnum::Range { start, end, .. } => span
                .maybe_join(start.map(|s| s.full_span()))
                .maybe_join(end.map(|s| s.full_span())),
            AstEnum::Decl { ident, var_ty_expr, has_init_expr, init, .. } => {
                match init.filter(|_| *has_init_expr).or(*var_ty_expr) {
                    Some(e) => span.join(e.full_span()),
                    None => span.join(ident.span),
                }
            },
            AstEnum::If { condition, then_body, else_body, was_piped, .. } => {
                let r_span = else_body.unwrap_or(*then_body).full_span();
                if *was_piped { condition.full_span() } else { span }.join(r_span)
            },
            &AstEnum::Switch { val, else_body, was_piped, .. } => {
                if was_piped { val.full_span() } else { span }
                    .maybe_join(else_body.as_deref().map(Ast::full_span))
            },
            AstEnum::For { source_expr: l, body, was_piped, .. }
            | AstEnum::While { condition: l, body, was_piped, .. } => {
                if *was_piped { l.full_span() } else { span }.join(body.full_span())
            },
            AstEnum::Loop { body, .. } | AstEnum::Defer { stmt: body, .. } => {
                span.join(body.full_span())
            },
            AstEnum::Return { val, .. } => match val {
                Some(val) => span.join(val.full_span()),
                None => span,
            },
            AstEnum::GenericDef { name, .. } => span.join(name.span),

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

impl Type {
    #[inline]
    pub fn matchable(&self) -> Ptr<TypeEnum> {
        Ptr::from_ref(self).cast()
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
        debug_assert!(
            Ptr::from_ref(self).try_downcast_polymorphable().is_none_or(|p| !p.is_generic())
        );
        match self.matchable().as_ref() {
            TypeEnum::StructDef { scope, .. }
            | TypeEnum::UnionDef { scope, .. }
            | TypeEnum::EnumDef { scope, .. } => Some(scope),
            _ => None,
        }
    }

    pub fn get_associated_external_consts(&self) -> Option<&[Ptr<Decl>]> {
        match self.matchable().as_ref() {
            TypeEnum::StructDef { external_consts, .. }
            | TypeEnum::UnionDef { external_consts, .. }
            | TypeEnum::EnumDef { external_consts, .. } => Some(external_consts),
            _ => None,
        }
    }

    // TODO: add SliceTy cases
    pub fn get_fields(&self) -> Option<Ptr<[Ptr<Decl>]>> {
        match self.matchable().as_ref() {
            TypeEnum::StructDef { fields, .. } | TypeEnum::UnionDef { fields, .. } => Some(*fields),
            _ => None,
        }
    }

    /// Counts the number of nested optional layers.
    /// `???int` => 3; `int` => 0
    pub fn count_optional_nesting(self: Ptr<Type>) -> usize {
        self.iter_nested_optionals().count()
    }

    pub fn iter_nested_optionals(
        self: Ptr<Type>,
    ) -> iter::Successors<Ptr<OptionTy>, impl FnMut(&Ptr<OptionTy>) -> Option<Ptr<OptionTy>>> {
        iter::successors(self.try_downcast::<OptionTy>(), |t| t.inner_ty.try_downcast::<OptionTy>())
    }

    pub fn clone_for_finalize(self: Ptr<Self>, alloc: &Arena) -> Result<Ptr<Self>, AllocErr> {
        Ok(match self.matchable2() {
            TypeMatch::SimpleTy(_)
            | TypeMatch::IntTy(_)
            | TypeMatch::FloatTy(_)
            | TypeMatch::StructDef(_)
            | TypeMatch::UnionDef(_)
            | TypeMatch::EnumDef(_)
            | TypeMatch::Fn(_) => self,
            TypeMatch::PtrTy(p) => {
                let pointee = p.pointee.downcast_type().clone_for_finalize(alloc)?.upcast();
                alloc.alloc(PtrTy { pointee, ..*p })?.upcast_to_type()
            },
            TypeMatch::SliceTy(s) => {
                let elem_ty = s.elem_ty.downcast_type().clone_for_finalize(alloc)?.upcast();
                alloc.alloc(SliceTy { elem_ty, ..*s })?.upcast_to_type()
            },
            TypeMatch::ArrayTy(a) => {
                let len = a.len.downcast::<IntVal>();
                let len = alloc.alloc(IntVal { val: len.val.clone(), ..*len })?.upcast();
                let elem_ty = a.elem_ty.downcast_type().clone_for_finalize(alloc)?.upcast();
                alloc.alloc(ArrayTy { len, elem_ty, ..*a })?.upcast_to_type()
            },
            TypeMatch::RangeTy(r) => {
                let elem_ty = r.elem_ty.clone_for_finalize(alloc)?;
                alloc.alloc(RangeTy { elem_ty, ..*r })?.upcast_to_type()
            },
            TypeMatch::OptionTy(o) => {
                let inner_ty = o.inner_ty.downcast_type().clone_for_finalize(alloc)?.upcast();
                alloc.alloc(OptionTy { inner_ty, ..*o })?.upcast_to_type()
            },
            TypeMatch::ArrayLikeContainer(_) => unreachable_debug(),
            TypeMatch::GenericDef(_) => todo!("Generic"),
        })
    }
}

impl TypeEnum {
    #[inline]
    pub fn as_type(&self) -> OPtr<Type> {
        match self {
            TypeEnum::Unset => None,
            _ => Some(Ptr::from_ref(self).cast()),
        }
    }
}

pub trait OPtrExt<T> {
    fn upcast_to_type(self) -> OPtr<Type>
    where T: TypeVariant;
}

impl<T> OPtrExt<T> for OPtr<T> {
    fn upcast_to_type(self) -> OPtr<Type>
    where T: TypeVariant {
        self.map(Ptr::upcast_to_type)
    }
}

pub trait OPtrTypeExt {
    fn matchable(self) -> Ptr<TypeEnum>;
    fn downcast<V: TypeVariant>(self) -> Ptr<V>;
    fn try_downcast<V: TypeVariant>(self) -> OPtr<V>;
    fn try_downcast_ty_hint<V: TypeVariant>(self) -> OPtr<V>;
}

impl OPtrTypeExt for OPtr<Type> {
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

    #[inline]
    fn try_downcast_ty_hint<V: TypeVariant>(self) -> OPtr<V> {
        self.and_then(|t| t.try_downcast_ty_hint::<V>())
    }
}

impl AstEnum {
    #[inline]
    pub fn as_ast(&self) -> Ptr<Ast> {
        Ptr::from_ref(self).cast()
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

impl Dot {
    pub const fn new(lhs: Option<Ptr<Ast>>, rhs: Ptr<Ident>, span: Span) -> Dot {
        ast_new!(local Dot { span, lhs, has_lhs: lhs.is_some(), rhs })
    }
}

impl Decl {
    pub const fn new(ident: Ptr<Ident>, associated_type_expr: OPtr<Ast>, span: Span) -> Decl {
        ast_new!(local Decl {
            span,
            flags: DeclFlags::default(),
            is_const: false,
            has_init_expr: false,
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
    pub fn from_lhs(lhs: Ptr<Ast>) -> ParseResult<Decl> {
        match lhs.matchable2() {
            AstMatch::Ident(lhs) => Ok(Decl::from_ident(lhs)),
            AstMatch::Dot(dot) => match dot.lhs {
                Some(ty_expr) => Ok(Decl::new(dot.rhs, Some(ty_expr), lhs.full_span())),
                None => {
                    Err(cerror2!(dot.span, "A member declaration requires an associated type name"))
                },
            },
            _ => Err(unexpected_expr(lhs, "a variable name")),
        }
    }

    pub fn is_lhs_only(&self) -> bool {
        self.var_ty_expr.is_none() && self.init.is_none()
    }

    pub fn const_val(self: Ptr<Decl>) -> Ptr<ConstVal> {
        debug_assert!(self.is_allowed_in_const());
        debug_assert!(self.var_ty.is_some() || self.init.u().kind == AstKind::Fn);
        if let Some(t) = self.var_ty
            && t.propagates_out()
        {
            return t.upcast().downcast_const_val();
        }
        debug_assert!(self.is_const);
        self.init.u().downcast_const_val()
    }

    pub fn try_const_val(self: Ptr<Decl>) -> OPtr<Ast> {
        then!(self.is_const => self.const_val().upcast())
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
        self.is_allowed_in_const()
    }

    pub fn is_allowed_in_const(&self) -> bool {
        self.is_const || self.flags.get(DeclFlags::IS_STATIC)
    }

    pub fn has_default(&self, is_enum_variant: bool) -> bool {
        if is_enum_variant { false } else { self.init.is_some() }
    }
}

impl Block {
    pub fn new(stmts: Ptr<[Ptr<Ast>]>, has_trailing_semicolon: bool, span: Span) -> Self {
        let mut decl_scope = Scope::new(vec![], ScopeKind::Block);
        decl_scope.verify_no_duplicates().u();
        ast_new!(local Block { span, has_trailing_semicolon, stmts, decl_scope, finished: 0 })
    }
}

impl For {
    pub fn new(
        source_expr: Ptr<Ast>,
        iter_var: Ptr<Ident>,
        body: Ptr<Ast>,
        was_piped: bool,
        span: Span,
        alloc: &Arena,
    ) -> Result<Ptr<For>, HandledErr> {
        let iter_var = alloc.alloc(Decl::from_ident(iter_var))?;
        let scope = Scope::new(vec![iter_var], ScopeKind::ForLoop);
        Ok(ast_new!(alloc, For { source_expr, iter_var, body, scope, was_piped, span }))
    }
}

impl PositionalInitializer {
    pub fn new(lhs: OPtr<Ast>, args: Ptr<[Ptr<Ast>]>, span: Span) -> Self {
        ast_new!(local PositionalInitializer { lhs, args, parsed_with_lhs: lhs.is_some(), resolved_struct_inst: None, span })
    }
}

impl NamedInitializer {
    pub fn new(lhs: OPtr<Ast>, fields: Ptr<[(Ptr<Ident>, OPtr<Ast>)]>, span: Span) -> Self {
        ast_new!(local NamedInitializer { lhs, fields, parsed_with_lhs: lhs.is_some(), resolved_struct_inst: None, span })
    }
}

impl IntVal {
    pub fn new<I>(i: I) -> Result<Ptr<IntVal>, HandledErr>
    where BigInt: From<I> {
        Ok(ast_new!(IntVal { val: BigInt::from(i) }, Span::ZERO))
    }
}

impl EnumVal {
    pub fn variant(&self) -> Ptr<Decl> {
        self.enum_ty.variants.get(self.variant_idx).u()
    }

    pub fn tag_val(&self) -> &BigInt {
        &self.variant().init.u().downcast::<IntVal>().as_ref().val
    }
}

impl EnumDef {
    /// TODO: handle duplicate tags
    #[cfg(debug_assertions)]
    pub fn find_variant_ty_for_tag(&self, tag_val: &BigInt) -> Ptr<Type> {
        let variant = self.variants.into_iter().find(|v| {
            debug_assert!(!v.is_const);
            v.init.u().downcast::<IntVal>().val == *tag_val
        });
        variant.u().var_ty.u()
    }
}

impl Fn {
    pub fn new(
        params: Vec<Ptr<Decl>>,
        ret_ty_expr: OPtr<Ast>,
        body: OPtr<Ast>,
        start_span: Span,
        alloc: &Arena,
    ) -> Result<Ptr<Fn>, AllocErr> {
        debug_assert!(params.iter().all(|p| p.flags.get(DeclFlags::IS_PARAMETER)));
        debug_assert!(params.iter().all(|p| !p.is_const));
        Ok(ast_new!(alloc, Fn {
            flags: FnFlags::default(),
            params_scope: Scope::new(params, ScopeKind::FnParams),
            generics_scope: None,
            ret_ty_expr,
            ret_ty: None,
            instantiations: vec![],
            body,
            #[cfg(debug_assertions)]
            decl: None,
            span: start_span,
        }))
    }

    #[inline]
    pub fn params(&self) -> &[Ptr<Decl>] {
        &self.params_scope.decls
    }
}

impl GenericDef {
    pub fn new(name: Ptr<Ident>, span: Span) -> GenericDef {
        ast_new!(local GenericDef { name, idx: usize::MAX, cur_inst: None, span })
    }

    pub fn generate_decl(self: Ptr<Self>, alloc: &Arena) -> Result<Ptr<Decl>, AllocErr> {
        let mut generic_decl = ast_new!(alloc, Decl {
            span: self.span,
            is_const: true,
            has_init_expr: false,
            flags: DeclFlags::default(),
            ident: self.name,
            on_type: None,
            var_ty_expr: None,

            // @FIXME: This is incorrect, but currently needed because otherwise
            // uses of this generic are never resolved
            // (get_symbol_var_ty would return `NotFinished(VarType)`)
            var_ty: Some(self.upcast_to_type()),

            init: Some(self.upcast()),
            obj_symbol_name: None,
        });
        generic_decl.flags.set(DeclFlags::IS_GENERIC);
        self.as_mut().name.decl.set_once(generic_decl);
        Ok(generic_decl)
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
    pub fn as_binop_text(self) -> &'static str {
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
            BinOpKind::And => "and",
            BinOpKind::Or => "or",
        }
    }

    pub fn as_binop_assign_text(self) -> &'static str {
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
                finalize_ty(lhs_ty, out_ty, false);
                finalize_ty(rhs_ty, out_ty, false);
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
        // type coercion is not possible here => `remove_type_coercion_for_finalize` is not needed
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
                debug_assert!(ty_match(*pointee, out_ty), "{pointee} matches {out_ty}");
                finalize_ty(pointee, out_ty, true);
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

bitflags!(DeclFlags: u8 {
    IS_MUT,
    IS_PUB,
    IS_REC,
    IS_STATIC,
    //IS_CONST,
    IS_DATA_MEMBER,
    IS_CONST_MEMBER,
    IS_PARAMETER,
    IS_GENERIC,
    //IS_EXPLICIT_CONST_PARAM,
});

pub type DeclList = Ptr<[Ptr<Decl>]>;

pub trait DeclListExt {
    fn find_field(&self, sym: Symbol) -> Option<(usize, Ptr<Decl>)>;

    fn iter_types(&self) -> impl ExactSizeIterator<Item = Ptr<Type>> + Clone + '_;
}

impl DeclListExt for [Ptr<Decl>] {
    fn find_field(&self, sym: Symbol) -> Option<(usize, Ptr<Decl>)> {
        self.iter().copied().enumerate().find(|(_, d)| d.ident.sym == sym)
    }

    fn iter_types(&self) -> impl ExactSizeIterator<Item = Ptr<Type>> + Clone + '_ {
        self.iter().map(|decl| {
            //debug_expr!(*decl);
            decl.var_ty.u()
        })
    }
}

bitflags!(FnFlags: u8 {
    /// set during sema
    HAS_KNOWN_RET_TY,
    HAS_VARARGS,

    /// ```mylang
    /// named :: () -> {}
    /// () -> {}; // unnamed
    /// ```
    IS_NAMED,

    IS_TYPE,
    IS_GENERIC,
    IS_INSTANTIATION,
});

bitflags!(StructFlags: u8 {
    IS_GENERIC,
    IS_INSTANTIATION,
});

bitflags!(EnumFlags: u8 {
    IS_TAGGED_UNION,

    IS_GENERIC,
    IS_INSTANTIATION,
});

pub fn is_pos_arg(a: &Ptr<Ast>) -> bool {
    try_downcast_named_arg(*a).is_none()
}

pub fn try_downcast_named_arg(arg: Ptr<Ast>) -> OPtr<Assign> {
    then!(arg.parenthesis_count == 0 => arg.try_downcast::<Assign>()?)
}

#[derive(Debug, Clone, Copy)]
pub struct SwitchCase {
    pub case: Ptr<Ast>,
    pub body: Ptr<Ast>,
    pub scope: Ptr<Scope>,
}

impl SwitchCase {
    pub fn new(case: Ptr<Ast>, body: Ptr<Ast>, alloc: &Arena) -> Result<Self, AllocErr> {
        let scope = alloc.alloc(Scope::new(vec![], ScopeKind::SwitchCase))?;
        Ok(SwitchCase { case, body, scope })
    }
}

pub trait CloneAst<Target = Self>: Sized {
    fn clone_ast(&self, alloc: &Arena) -> Target {
        self._clone_ast(alloc).unwrap_or_else(|_| panic!("oom"))
    }

    fn _clone_ast(&self, alloc: &Arena) -> Result<Target, AllocErr>;
}

impl<T: CloneAst> CloneAst for Ptr<[T]> {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        alloc.alloc_slice_fill_iter(self.iter().map(|t| t.clone_ast(alloc)))
    }
}

impl<T: CloneAst> CloneAst for Vec<T> {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        Ok(self.iter().map(|t| t.clone_ast(alloc)).collect::<Vec<_>>())
    }
}

impl<T: CloneAst> CloneAst for Option<T> {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        Ok(self.as_ref().map(|t| t.clone_ast(alloc)))
    }
}

impl<A: CloneAst, B: CloneAst> CloneAst for (A, B) {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        Ok((self.0.clone_ast(alloc), self.1.clone_ast(alloc)))
    }
}

impl CloneAst for Ptr<Ast> {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        macro_rules! clone {
            ($kind:ident { $( $(#[$attr:meta])* $field:ident $( : $val:expr )?),* $(,)? }) => {
                alloc.alloc(ast_new!(local $kind { span: self.span, $( $(#[$attr])* $field $(:$val)?),* }))?.upcast()
            };
        }

        Ok(match self.matchable().as_ref() {
            &AstEnum::Ident { sym, .. } => clone!(Ident { sym, decl: None }),
            &AstEnum::Block { has_trailing_semicolon, stmts, span, .. } => alloc
                .alloc(Block::new(stmts.clone_ast(alloc), has_trailing_semicolon, span))?
                .upcast(),
            &AstEnum::PositionalInitializer { parsed_with_lhs, lhs, args, .. } => {
                clone!(PositionalInitializer {
                    parsed_with_lhs,
                    lhs: lhs.clone_ast(alloc),
                    args: args.clone_ast(alloc),
                    resolved_struct_inst: None
                })
            },
            &AstEnum::NamedInitializer { parsed_with_lhs, lhs, fields, .. } => {
                clone!(NamedInitializer {
                    parsed_with_lhs,
                    lhs: lhs.clone_ast(alloc),
                    fields: fields.clone_ast(alloc),
                    resolved_struct_inst: None
                })
            },
            &AstEnum::ArrayInitializer { parsed_with_lhs, lhs, elements, .. } => {
                clone!(ArrayInitializer {
                    parsed_with_lhs,
                    lhs: lhs.clone_ast(alloc),
                    elements: elements.clone_ast(alloc),
                })
            },
            &AstEnum::ArrayInitializerShort { parsed_with_lhs, lhs, val, count, .. } => {
                clone!(ArrayInitializerShort {
                    parsed_with_lhs,
                    lhs: lhs.clone_ast(alloc),
                    val: val.clone_ast(alloc),
                    count: count.clone_ast(alloc),
                })
            },
            &AstEnum::Dot { has_lhs, lhs, rhs, .. } => {
                clone!(Dot { has_lhs, lhs: lhs.clone_ast(alloc), rhs: rhs.clone_ast(alloc) })
            },
            &AstEnum::Index { mut_access, lhs, idx, .. } => {
                clone!(Index { mut_access, lhs: lhs.clone_ast(alloc), idx: idx.clone_ast(alloc) })
            },
            &AstEnum::Cast { operand, target_ty, .. } => clone!(Cast {
                operand: operand.clone_ast(alloc),
                target_ty: target_ty.clone_ast(alloc),
            }),
            &AstEnum::Autocast { operand, .. } => {
                clone!(Autocast { operand: operand.clone_ast(alloc) })
            },
            &AstEnum::Call { func, args, pipe_idx, .. } => clone!(Call {
                func: func.clone_ast(alloc),
                args: args.clone_ast(alloc),
                pipe_idx,
                resolved_fn_inst: None,
            }),
            &AstEnum::UnaryOp { is_postfix, op, operand, .. } => {
                clone!(UnaryOp { is_postfix, op, operand: operand.clone_ast(alloc) })
            },
            &AstEnum::BinOp { lhs, op, rhs, .. } => {
                clone!(BinOp { lhs: lhs.clone_ast(alloc), op, rhs: rhs.clone_ast(alloc) })
            },
            &AstEnum::Range { is_inclusive, start, end, .. } => clone!(Range {
                is_inclusive,
                start: start.clone_ast(alloc),
                end: end.clone_ast(alloc),
            }),
            AstEnum::OrElse { lhs, rhs, .. } => {
                clone!(OrElse { lhs: lhs.clone_ast(alloc), rhs: rhs.clone_ast(alloc) })
            },
            &AstEnum::Assign { is_explicit_generic_arg, lhs, rhs, .. } => clone!(Assign {
                is_explicit_generic_arg,
                lhs: lhs.clone_ast(alloc),
                rhs: rhs.clone_ast(alloc),
            }),
            &AstEnum::BinOpAssign { lhs, op, rhs, .. } => {
                clone!(BinOpAssign { lhs: lhs.clone_ast(alloc), op, rhs: rhs.clone_ast(alloc) })
            },
            AstEnum::Decl { .. } => self.downcast::<Decl>()._clone_ast(alloc)?.upcast(),
            &AstEnum::If { was_piped, condition, then_body, else_body, .. } => clone!(If {
                was_piped,
                condition: condition.clone_ast(alloc),
                then_body: then_body.clone_ast(alloc),
                else_body: else_body.clone_ast(alloc),
            }),
            &AstEnum::Switch { was_piped, val, cases, else_body, .. } => clone!(Switch {
                was_piped,
                val: val.clone_ast(alloc),
                cases: cases.clone_ast(alloc),
                else_body: else_body.clone_ast(alloc),
            }),
            &AstEnum::For { was_piped, source_expr, iter_var, body, .. } => For::new(
                source_expr.clone_ast(alloc),
                iter_var.ident.clone_ast(alloc),
                body.clone_ast(alloc),
                was_piped,
                self.span,
                alloc,
            )?
            .upcast(),
            &AstEnum::While { was_piped, condition, body, .. } => clone!(While {
                was_piped,
                condition: condition.clone_ast(alloc),
                body: body.clone_ast(alloc),
            }),
            AstEnum::Loop { body, .. } => {
                clone!(Loop { body: body.clone_ast(alloc), break_ty: None })
            },
            AstEnum::Defer { stmt, .. } => clone!(Defer { stmt: stmt.clone_ast(alloc) }),
            AstEnum::Return { val, .. } => {
                clone!(Return { val: val.clone_ast(alloc), parent_fn: None })
            },
            AstEnum::Break { val, .. } => clone!(Break { val: val.clone_ast(alloc) }),
            AstEnum::Continue { .. } => clone!(Continue {}),
            AstEnum::Empty { .. } => clone!(Empty {}),
            AstEnum::IntVal { val, .. } => clone!(IntVal { val: val.clone() }),
            &AstEnum::FloatVal { val, .. } => clone!(FloatVal { val }),
            &AstEnum::BoolVal { val, .. } => clone!(BoolVal { val }),
            &AstEnum::CharVal { val, .. } => clone!(CharVal { val }),
            &AstEnum::StrVal { text, .. } => clone!(StrVal { text }),
            &AstEnum::RawPtrVal { val, .. } => clone!(RawPtrVal { val }),
            AstEnum::StaticPtrVal { .. }
            | AstEnum::EnumVal { .. }
            | AstEnum::OptionalVal { .. }
            | AstEnum::AggregateVal { .. } => unreachable_debug(),
            &AstEnum::ImportDirective { path, files_idx, .. } => {
                clone!(ImportDirective { path, files_idx })
            },
            AstEnum::ExternDirective { .. } => clone!(ExternDirective { decl: None }),
            &AstEnum::IntrinsicDirective { intrinsic_name, .. } => {
                clone!(IntrinsicDirective { intrinsic_name, decl: None })
            },
            AstEnum::ProgramMainDirective { .. } => clone!(ProgramMainDirective {}),
            &AstEnum::SimpleDirective { ret_ty, .. } => clone!(SimpleDirective { ret_ty }),
            AstEnum::SizeOfDirective { type_, .. } => {
                clone!(SizeOfDirective { type_: type_.clone_ast(alloc) })
            },
            AstEnum::SizeOfValDirective { val, .. } => {
                clone!(SizeOfValDirective { val: val.clone_ast(alloc) })
            },
            AstEnum::AlignOfDirective { type_, .. } => {
                clone!(AlignOfDirective { type_: type_.clone_ast(alloc) })
            },
            AstEnum::OffsetOfDirective { type_, field, .. } => clone!(OffsetOfDirective {
                type_: type_.clone_ast(alloc),
                field: field.clone_ast(alloc)
            }),
            AstEnum::SimpleTy { .. } | AstEnum::IntTy { .. } | AstEnum::FloatTy { .. } => *self,
            &AstEnum::PtrTy { is_mut, pointee, .. } => {
                clone!(PtrTy { is_mut, pointee: pointee.clone_ast(alloc) })
            },
            &AstEnum::SliceTy { is_mut, elem_ty, .. } => {
                clone!(SliceTy { is_mut, elem_ty: elem_ty.clone_ast(alloc) })
            },
            AstEnum::ArrayTy { len, elem_ty, .. } => {
                clone!(ArrayTy { len: len.clone_ast(alloc), elem_ty: elem_ty.clone_ast(alloc) })
            },
            AstEnum::StructDef { flags, scope, generics_scope, .. } => {
                let ScopeAndAggregateInfo { scope, fields } =
                    Scope::for_aggregate(scope.decls.clone_ast(alloc), alloc, ScopeKind::Struct)?;
                clone!(StructDef {
                    flags: *flags,
                    scope,
                    generics_scope: generics_scope
                        .map(|s| alloc.alloc(Scope::for_generics(s.decls.clone_ast(alloc))))
                        .transpose()?,
                    instantiations: vec![],
                    sema_units: None,
                    fields,
                    external_consts: vec![],
                    finished_members: 0
                })
            },
            AstEnum::UnionDef { .. } | AstEnum::EnumDef { .. } => {
                todo!()
            },
            AstEnum::RangeTy { .. } => unreachable_debug(),
            AstEnum::OptionTy { inner_ty, .. } => {
                clone!(OptionTy { inner_ty: inner_ty.clone_ast(alloc) })
            },
            AstEnum::Fn { .. } => self.downcast::<Fn>()._clone_ast(alloc)?.upcast(),
            AstEnum::GenericDef { name, .. } => {
                alloc.alloc(GenericDef::new(name.clone_ast(alloc), self.span))?.upcast()
            },
            AstEnum::ArrayLikeContainer { .. } => unreachable_debug(),
        })
    }
}

impl CloneAst for Ptr<Ident> {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        Ok(ast_new!(alloc, Ident { sym: self.sym, decl: None, span: self.span }))
    }
}

impl CloneAst for SwitchCase {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        SwitchCase::new(self.case.clone_ast(alloc), self.body.clone_ast(alloc), alloc)
    }
}

impl CloneAst for Ptr<Decl> {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        alloc.alloc(ast_new!(local Decl {
            is_const: self.is_const,
            has_init_expr: self.has_init_expr,
            flags: self.flags,
            ident: self.ident.clone_ast(alloc),
            on_type: self.on_type.clone_ast(alloc),
            var_ty_expr: self.var_ty_expr.clone_ast(alloc),
            var_ty: None,
            init: self.init.clone_ast(alloc),
            obj_symbol_name: None,
            span: self.span
        }))
    }
}

impl CloneAst for Ptr<Fn> {
    fn _clone_ast(&self, alloc: &Arena) -> Result<Self, AllocErr> {
        let mut f = Fn::new(
            self.params_scope.decls.clone_ast(alloc),
            self.ret_ty_expr.clone_ast(alloc),
            self.body.clone_ast(alloc),
            self.span,
            alloc,
        )?;
        f.flags = self.flags;
        f.flags.unset(FnFlags::HAS_KNOWN_RET_TY); // sema flag
        // also clones generics_scope, even though it is only generated after parsing, to not differ
        // from StructDef cloning.
        debug_assert_eq!(self.generics_scope.is_some(), self.flags.get(FnFlags::IS_GENERIC));
        if let Some(generics_scope) = self.generics_scope {
            f.generics_scope =
                Some(alloc.alloc(Scope::for_generics(generics_scope.decls.clone_ast(alloc)))?)
        }
        Ok(f)
    }
}

impl<V: AstVariant> CloneAst<Ptr<V>> for V {
    #[inline]
    fn _clone_ast(&self, alloc: &Arena) -> Result<Ptr<V>, AllocErr> {
        Ok(Ptr::from_ref(self).upcast()._clone_ast(alloc)?.downcast::<V>())
    }
}
