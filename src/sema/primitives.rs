use crate::{
    arena_allocator::{AllocErr, Arena},
    ast::{self, UpcastToAst},
    parser::lexer::Span,
    ptr::Ptr,
};

#[derive(Debug)]
pub struct Primitives {
    // Types:
    pub void_ty: Ptr<ast::Type>,
    pub never: Ptr<ast::Type>,
    pub never_ptr_ty: Ptr<ast::Type>,
    pub any: Ptr<ast::Type>,
    pub any_ptr_ty: Ptr<ast::Type>,
    pub u0: Ptr<ast::Type>,
    pub u8: Ptr<ast::Type>,
    pub u16: Ptr<ast::Type>,
    pub u32: Ptr<ast::Type>,
    pub u64: Ptr<ast::Type>,
    pub u128: Ptr<ast::Type>,
    pub i8: Ptr<ast::Type>,
    pub i16: Ptr<ast::Type>,
    pub i32: Ptr<ast::Type>,
    pub i64: Ptr<ast::Type>,
    pub i128: Ptr<ast::Type>,
    pub bool: Ptr<ast::Type>,
    pub char: Ptr<ast::Type>,
    pub f32: Ptr<ast::Type>,
    pub f64: Ptr<ast::Type>,
    pub str_slice_ty: Ptr<ast::Type>,
    pub type_ty: Ptr<ast::Type>,

    // internal Types:
    pub unknown_ty: Ptr<ast::Type>,
    pub int_lit: Ptr<ast::Type>,
    pub sint_lit: Ptr<ast::Type>,
    pub float_lit: Ptr<ast::Type>,
    pub method_stub: Ptr<ast::Type>,
    pub enum_variant: Ptr<ast::Type>,
    pub module: Ptr<ast::Type>,
    pub library: Ptr<ast::Type>,

    // Other:
    /// TODO: remove nil
    pub nil: Ptr<ast::Decl>,
    pub slice_ptr_field_ident: Ptr<ast::Ident>,
    pub slice_len_field: Ptr<ast::Decl>,
    /// struct def for type `[]untyped`. The inner type can be unknown because llvm doesn't care
    /// about pointee types.
    pub untyped_slice_struct_def: Ptr<ast::StructDef>,
    pub empty_array_ty: Ptr<ast::ArrayTy>,
}

impl Primitives {
    pub fn setup(decls: &mut Vec<Ptr<ast::Decl>>, alloc: &Arena) -> Self {
        Self::try_setup(decls, alloc).unwrap_or_else(|e| panic!("allocation failed: {e:?}"))
    }

    pub fn try_setup(decls: &mut Vec<Ptr<ast::Decl>>, alloc: &Arena) -> Result<Self, AllocErr> {
        decls.reserve(30);

        macro_rules! ast_new {
            ($kind:ident {
                $(
                    $field:ident
                    $( : $val:expr )?
                ),* $(,)?
            }) => {
                alloc.alloc(crate::ast::$kind {
                    kind: crate::ast::AstKind::$kind,
                    ty: None,
                    replacement: None,
                    parenthesis_count: 0,
                    span: Span::ZERO,
                    $( $field $(: $val)? ),*
                })?
            };
        }

        let new_primitive_decl = |decl_name: &str| {
            let mut ident = ast_new!(Ident { text: Ptr::from_ref(decl_name), decl: None });
            let mut decl = alloc.alloc(ast::Decl::from_ident(ident))?;
            ident.decl = Some(decl);
            decl.is_const = true;
            Ok::<_, AllocErr>(decl)
        };

        let type_ty_decl = new_primitive_decl("type")?;
        insert_symbol_no_duplicate(decls, type_ty_decl);
        let type_ty =
            ast_new!(SimpleTy { decl: type_ty_decl, is_finalized: true }).upcast_to_type();

        let void_ty_decl = new_primitive_decl("void")?;
        insert_symbol_no_duplicate(decls, void_ty_decl);
        let void_ty =
            ast_new!(SimpleTy { decl: void_ty_decl, is_finalized: true }).upcast_to_type();

        let init_ty = |t: Ptr<ast::Type>| t.as_mut().ty = Some(type_ty);

        let init_decl = |d: Ptr<ast::Decl>, t: Ptr<ast::Type>, init| {
            let d = d.as_mut();
            d.ty = Some(void_ty);
            d.var_ty = Some(t);
            //d.ident.ty = Some(t); // TODO: include this?
            d.init = init;
            if let Some(init) = init {
                init.as_mut().ty = Some(t);
            }
        };

        let init_ty_decl = |d, t: Ptr<ast::Type>| {
            init_ty(t);
            init_decl(d, type_ty, Some(t.upcast()));
        };

        init_ty_decl(type_ty_decl, type_ty);
        init_ty_decl(void_ty_decl, void_ty);

        macro_rules! new_primitive_ty {
            ($decl_name:expr,simple_ty, finalized: $finalized:expr) => {{
                let decl = new_primitive_decl($decl_name)?;
                insert_symbol_no_duplicate(decls, decl);
                let ty = ast_new!(SimpleTy { decl, is_finalized: $finalized }).upcast_to_type();
                init_ty_decl(decl, ty);
                ty
            }};
            ($decl_name:expr, $ty_kind:ident {
                $( $field:ident : $val:expr),* $(,)?
            }) => {{
                let decl = new_primitive_decl($decl_name)?;
                insert_symbol_no_duplicate(decls, decl);
                let ty = ast_new!($ty_kind { $($field: $val),* }).upcast_to_type();
                init_ty_decl(decl, ty);
                ty
            }};
        }

        let never = new_primitive_ty!("never", simple_ty, finalized: true);
        let never_ptr_ty =
            ast_new!(PtrTy { pointee: never.upcast(), is_mut: true }).upcast_to_type();
        init_ty(never_ptr_ty);

        let any = new_primitive_ty!("any", simple_ty, finalized: true);
        let any_ptr_ty = ast_new!(PtrTy { pointee: any.upcast(), is_mut: false }).upcast_to_type();
        init_ty(any_ptr_ty);

        let u8 = new_primitive_ty!("u8", IntTy { bits: 8, is_signed: false });
        let u64 = new_primitive_ty!("u64", IntTy { bits: 64, is_signed: false });

        let mut untyped_slice_ptr_field = new_primitive_decl("ptr")?;
        untyped_slice_ptr_field.is_const = false;
        init_decl(untyped_slice_ptr_field, any_ptr_ty, None);

        let slice_len_field = new_primitive_decl("len")?;
        init_decl(slice_len_field, u64, None);

        Ok(Primitives {
            void_ty,
            never,
            never_ptr_ty,
            any,
            any_ptr_ty,
            u0: new_primitive_ty!("u0", IntTy { bits: 0, is_signed: false }),
            u8,
            u16: new_primitive_ty!("u16", IntTy { bits: 16, is_signed: false }),
            u32: new_primitive_ty!("u32", IntTy { bits: 32, is_signed: false }),
            u64,
            u128: new_primitive_ty!("u128", IntTy { bits: 128, is_signed: false }),
            i8: new_primitive_ty!("i8", IntTy { bits: 8, is_signed: true }),
            i16: new_primitive_ty!("i16", IntTy { bits: 16, is_signed: true }),
            i32: new_primitive_ty!("i32", IntTy { bits: 32, is_signed: true }),
            i64: new_primitive_ty!("i64", IntTy { bits: 64, is_signed: true }),
            i128: new_primitive_ty!("i128", IntTy { bits: 128, is_signed: true }),
            bool: new_primitive_ty!("bool", simple_ty, finalized: true),
            char: new_primitive_ty!("char", simple_ty, finalized: true),
            f32: new_primitive_ty!("f32", FloatTy { bits: 32 }),
            f64: new_primitive_ty!("f64", FloatTy { bits: 64 }),
            str_slice_ty: {
                let str_slice =
                    ast_new!(SliceTy { elem_ty: u8.upcast(), is_mut: false }).upcast_to_type();
                init_ty(str_slice);
                str_slice
            },
            type_ty,

            unknown_ty: new_primitive_ty!("{unknown_ty}", simple_ty, finalized: true),
            int_lit: new_primitive_ty!("{integer literal}", simple_ty, finalized: false),
            sint_lit: new_primitive_ty!("{signed integer literal}", simple_ty, finalized: false),
            float_lit: new_primitive_ty!("{float literal}", simple_ty, finalized: false),
            method_stub: new_primitive_ty!("{method stub}", simple_ty, finalized: false),
            enum_variant: new_primitive_ty!("{enum variant}", simple_ty, finalized: false),
            module: new_primitive_ty!("{module}", simple_ty, finalized: true),
            library: new_primitive_ty!("{library}", simple_ty, finalized: true),

            nil: {
                let decl = new_primitive_decl("nil")?;
                init_decl(decl, never_ptr_ty, Some(ast_new!(PtrVal { val: 0 }).upcast()));
                insert_symbol_no_duplicate(decls, decl);
                decl
            },
            slice_ptr_field_ident: untyped_slice_ptr_field.ident,
            slice_len_field,
            untyped_slice_struct_def: {
                let def = ast_new!(StructDef {
                    fields: alloc.alloc_slice(&[untyped_slice_ptr_field, slice_len_field])?,
                    consts: Vec::new(),
                });
                init_ty(def.upcast_to_type());
                def
            },
            empty_array_ty: {
                let arr = ast_new!(ArrayTy {
                    len: ast_new!(IntVal { val: 0 }).upcast(),
                    elem_ty: never.upcast(),
                });
                init_ty(arr.upcast_to_type());
                arr
            },
        })
    }
}

fn insert_symbol_no_duplicate(decls: &mut Vec<Ptr<ast::Decl>>, decl: Ptr<ast::Decl>) {
    if decls.iter().any(|d| &*d.ident.text == &*decl.ident.text) {
        panic!("duplicate primitive name: {}", decl.ident.text.as_ref())
    }
    decls.push(decl);
}
