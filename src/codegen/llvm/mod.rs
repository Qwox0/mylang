use crate::{
    ast::{BinOpKind, Expr, ExprKind, ExprWithTy, Fn, UnaryOpKind, VarDecl, VarDeclList},
    defer_stack::DeferStack,
    ptr::Ptr,
    symbol_table::SymbolTable,
    type_::Type,
    util::{
        self, UnwrapDebug, forget_lifetime, get_aligned_offset, get_padding, panic_debug,
        unreachable_debug,
    },
};
pub use inkwell::targets::TargetMachine;
use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    attributes::Attribute,
    basic_block::BasicBlock,
    builder::{Builder, BuilderError},
    context::Context,
    execution_engine::FunctionLookupError,
    llvm_sys::{LLVMType, LLVMValue, prelude::LLVMValueRef},
    module::{Linkage, Module},
    passes::PassBuilderOptions,
    support::LLVMString,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetTriple},
    types::{
        AnyTypeEnum, ArrayType, AsTypeRef, BasicMetadataTypeEnum, BasicType, BasicTypeEnum,
        FloatType, IntType, PointerType, StructType,
    },
    values::{
        AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValue, BasicValueEnum,
        FloatValue, FunctionValue, InstructionValue, IntValue, PointerValue, StructValue,
    },
};
use std::{
    assert_matches::debug_assert_matches, collections::HashMap, marker::PhantomData, path::Path,
};

pub mod jit;

#[derive(Debug)]
pub enum CodegenError {
    BuilderError(BuilderError),
    InvalidGeneratedFunction,
    InvalidCallProduced,
    FunctionLookupError(FunctionLookupError),
    CannotOptimizeModule(LLVMString),
    CannotCompileObjFile(LLVMString),
    CannotCreateJit(LLVMString),
    CannotMutateRegister,

    AllocErr(bumpalo::AllocErr),
}

type CodegenResult<T> = Result<T, CodegenError>;

impl From<BuilderError> for CodegenError {
    fn from(e: BuilderError) -> Self {
        CodegenError::BuilderError(e)
    }
}

impl From<FunctionLookupError> for CodegenError {
    fn from(e: FunctionLookupError) -> Self {
        CodegenError::FunctionLookupError(e)
    }
}

macro_rules! try_not_never {
    ($sym:expr) => {{
        let sym = $sym;
        match sym {
            sym @ Symbol::Never => return Ok(sym),
            sym => sym,
        }
    }};
}

/// Returns [`Symbol::Never`] if it occurs
macro_rules! try_compile_expr_as_val {
    ($codegen:expr, $expr:expr, $expr_ty:expr) => {{
        let codegen: &mut Codegen = $codegen;
        let expr: Ptr<Expr> = $expr;
        let expr_ty: Type = $expr_ty;
        match codegen.compile_expr(expr, expr_ty)? {
            sym @ Symbol::Never => return Ok(sym),
            sym => codegen.sym_as_val(sym, expr_ty)?,
        }
    }};
    ($codegen:expr, $expr_with_ty:expr) => {{
        let ExprWithTy { expr, ty } = $expr_with_ty;
        try_compile_expr_as_val!($codegen, expr, ty)
    }};
}

/// Returns [`Symbol::Never`] if it occurs
macro_rules! try_get_symbol_as_val {
    ($codegen:expr, $name:expr, $ty:expr) => {{
        let codegen: &mut Codegen = $codegen;
        match codegen.get_symbol($name) {
            sym @ Symbol::Never => return Ok(sym),
            sym => codegen.sym_as_val(sym, $ty)?,
        }
    }};
}

pub struct Codegen<'ctx> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Module<'ctx>,

    symbols: SymbolTable<Symbol<'ctx>>,
    type_table: HashMap<VarDeclList, StructType<'ctx>>,
    defer_stack: DeferStack,

    cur_fn: Option<FunctionValue<'ctx>>,
    cur_loop: Option<Loop<'ctx>>,
    cur_var_name: Option<Ptr<str>>,
    sret_ptr: Option<PointerValue<'ctx>>,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(
        context: &'ctx Context,
        builder: Builder<'ctx>,
        module: Module<'ctx>,
    ) -> Codegen<'ctx> {
        Codegen {
            context,
            builder,
            module,
            symbols: SymbolTable::with_one_scope(),
            type_table: HashMap::new(),
            defer_stack: DeferStack::default(),
            cur_fn: None,
            cur_loop: None,
            cur_var_name: None,
            sret_ptr: None,
        }
    }

    pub fn new_module(context: &'ctx Context, module_name: &str) -> Self {
        let builder = context.create_builder();
        let module = context.create_module(module_name);
        Codegen::new(context, builder, module)
    }

    pub fn replace_mod_with_new(&mut self, new_mod_name: &str) -> Module<'ctx> {
        let mut module = self.context.create_module(new_mod_name);
        std::mem::swap(&mut self.module, &mut module);
        module
    }

    /// replaces the internal module with a new module which has the same name.
    pub fn take_module(&mut self) -> Module<'ctx> {
        let name = self.module.get_name().to_str().unwrap();
        let mut module = self.context.create_module(name);
        std::mem::swap(&mut self.module, &mut module);
        module
    }

    pub fn compile_top_level(&mut self, stmt: Ptr<Expr>) {
        // if let Err(err) = self.compile_expr_to_float(stmt) {
        //     self.errors.push(err);
        // }
        self.compile_expr(stmt, Type::Void).unwrap();
    }

    fn compile_expr(&mut self, expr: Ptr<Expr>, out_ty: Type) -> CodegenResult<Symbol<'ctx>> {
        self.compile_expr_with_write_target(expr, out_ty, None)
    }

    fn compile_expr_with_write_target(
        &mut self,
        expr: Ptr<Expr>,
        out_ty: Type,
        mut write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let out = match &expr.kind {
            ExprKind::Ident(name) => Ok(self.get_symbol(&**name)),
            ExprKind::Literal { code, .. } => match out_ty {
                Type::Never => Ok(Symbol::Never),
                Type::Int { bits, is_signed } => {
                    reg(self.int_type(bits).const_int(code.parse().unwrap_debug(), is_signed))
                },
                Type::Float { bits } => {
                    reg(unsafe { self.float_type(bits).const_float_from_string(code) })
                },
                t => todo!("{expr:#?} {t:?}"),
            },
            ExprKind::BoolLit(bool) => {
                debug_assert_matches!(out_ty, Type::Bool | Type::Unset);
                let b_ty = self.context.bool_type();
                reg(if *bool { b_ty.const_all_ones() } else { b_ty.const_zero() })
            },
            ExprKind::PtrTy { .. } => todo!(),
            ExprKind::SliceTy { .. } => todo!(),
            ExprKind::ArrayTy { .. } => todo!(),
            ExprKind::Fn(_) => todo!("todo: runtime fn"),
            ExprKind::Parenthesis { expr } => {
                self.compile_expr_with_write_target(*expr, out_ty, write_target.take())
            },
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                self.open_scope();
                let res = try {
                    let mut out = Symbol::Void;
                    for s in stmts.iter() {
                        out = self.compile_typed_expr(*s)?;
                        if out == Symbol::Never {
                            break;
                        }
                    }
                    if !*has_trailing_semicolon || out == Symbol::Never {
                        out
                    } else {
                        debug_assert_matches!(out_ty, Type::Void | Type::Unset);
                        Symbol::Void
                    }
                };
                self.close_scope()?;
                res
            },
            &ExprKind::StructDef(fields) => {
                Ok(Symbol::StructDef { fields, ty: self.struct_type(fields, self.cur_var_name) })
            },
            &ExprKind::UnionDef(fields) => {
                Ok(Symbol::UnionDef { fields, ty: self.union_type(fields, self.cur_var_name) })
            },
            &ExprKind::EnumDef(variants) => {
                Ok(Symbol::EnumDef { variants, ty: self.enum_type(variants, self.cur_var_name) })
            },
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::PositionalInitializer { lhs, lhs_ty, args } => {
                let (fields, struct_ty, struct_ptr) = match (lhs, *lhs_ty) {
                    (_, Type::Never) => return Ok(Symbol::Never),
                    (lhs, Type::Type(t)) => {
                        let Type::Struct { fields } = *t else { todo!() };
                        if let Some(lhs) = lhs {
                            self.compile_expr(*lhs, *t)?;
                        }
                        let struct_ty = self.type_table[&fields];
                        let ptr = if let Some(ptr) = write_target.take() {
                            ptr
                        } else {
                            self.build_alloca(struct_ty, "struct", &t)?
                        };

                        (fields, struct_ty, ptr)
                    },
                    (Some(lhs), Type::Ptr { pointee_ty }) => {
                        let Type::Struct { fields } = *pointee_ty else { unreachable_debug() };
                        let struct_ty = self.type_table[&fields];
                        let ptr = try_compile_expr_as_val!(self, *lhs, out_ty).ptr_val();
                        (fields, struct_ty, ptr)
                    },
                    _ => unreachable_debug(),
                };

                for (f_idx, field_def) in fields.iter().enumerate() {
                    let init = if let Some(init) = args.get(f_idx) {
                        *init
                    } else {
                        field_def.default.unwrap_debug()
                    };

                    let field_ptr = self.builder.build_struct_gep(
                        struct_ty,
                        struct_ptr,
                        f_idx as u32,
                        &field_def.ident.text,
                    )?;
                    let _ =
                        self.compile_expr_with_write_target(init, field_def.ty, Some(field_ptr))?;
                }

                stack_val(struct_ptr)
            },
            ExprKind::NamedInitializer { lhs, lhs_ty, fields: values } => {
                let (fields, struct_ty, struct_ptr) = match (lhs, *lhs_ty) {
                    (_, Type::Never) => return Ok(Symbol::Never),
                    (lhs, Type::Type(t)) => {
                        let Type::Struct { fields } = *t else { todo!() };
                        if let Some(lhs) = lhs {
                            self.compile_expr(*lhs, *t)?;
                        }
                        let struct_ty = self.type_table[&fields];
                        let ptr = if let Some(ptr) = write_target.take() {
                            ptr
                        } else {
                            self.build_alloca(struct_ty, "struct", &t)?
                        };

                        (fields, struct_ty, ptr)
                    },
                    (Some(lhs), Type::Ptr { pointee_ty }) => {
                        let Type::Struct { fields } = *pointee_ty else { unreachable_debug() };
                        let struct_ty = self.type_table[&fields];
                        let ptr = try_compile_expr_as_val!(self, *lhs, out_ty).ptr_val();
                        (fields, struct_ty, ptr)
                    },
                    _ => unreachable_debug(),
                };

                let mut is_initialized_field = vec![false; fields.len()];
                for (f, init) in values.iter() {
                    let (f_idx, field_def) = fields.find_field(&*f.text).unwrap_debug();

                    is_initialized_field[f_idx] = true;

                    let field_ptr = self.builder.build_struct_gep(
                        struct_ty,
                        struct_ptr,
                        f_idx as u32,
                        &*f.text,
                    )?;

                    match init {
                        Some(init) => {
                            let _ = self.compile_expr_with_write_target(
                                *init,
                                field_def.ty,
                                Some(field_ptr),
                            );
                        },
                        None => {
                            let val = try_get_symbol_as_val!(self, &*f.text, field_def.ty);
                            self.build_store(field_ptr, val, &field_def.ty)?;
                        },
                    };
                }
                for (f_idx, _) in
                    is_initialized_field.into_iter().enumerate().filter(|(_, is_init)| !is_init)
                {
                    let field = fields[f_idx];
                    let field_ptr = self.builder.build_struct_gep(
                        struct_ty,
                        struct_ptr,
                        f_idx as u32,
                        &field.ident.text,
                    )?;
                    let _ = self.compile_expr_with_write_target(
                        field.default.unwrap_debug(),
                        field.ty,
                        Some(field_ptr),
                    )?;
                }
                stack_val(struct_ptr)
            },
            ExprKind::ArrayInitializer { lhs, lhs_ty, elements } => {
                let Type::Array { len, elem_ty } = out_ty else { unreachable_debug() };
                let elem_cty = self.llvm_type(*elem_ty);
                let arr_ty = elem_cty.basic_ty().array_type(len as u32);
                let arr_ptr = if let Some(target) = write_target.take() {
                    target
                } else {
                    self.builder.build_array_alloca(
                        arr_ty,
                        self.context.i64_type().const_int(len as u64, false),
                        "arr",
                    )?
                };
                let idx_ty = self.context.i64_type();
                for (idx, elem) in elements.iter().enumerate() {
                    let elem_ptr = unsafe {
                        self.builder.build_in_bounds_gep(
                            arr_ty,
                            arr_ptr,
                            &[idx_ty.const_zero(), idx_ty.const_int(idx as u64, false)],
                            "",
                        )
                    }?;
                    //let elem_val = try_compile_expr_as_val!(self, *elem, *elem_ty);
                    //self.build_store(elem_ptr, elem_val, elem_ty.as_ref())?;
                    let _ = self.compile_expr_with_write_target(*elem, *elem_ty, Some(elem_ptr))?;
                }
                stack_val(arr_ptr)
            },
            ExprKind::ArrayInitializerShort { lhs, lhs_ty, val, count: _ } => {
                let Type::Array { len, elem_ty } = out_ty else { panic!() };
                let elem_cty = self.llvm_type(*elem_ty);
                let arr_ty = elem_cty.basic_ty().array_type(len as u32);
                let arr_ptr = if let Some(target) = write_target.take() {
                    target
                } else {
                    self.builder.build_array_alloca(
                        arr_ty,
                        self.context.i64_type().const_int(len as u64, false),
                        "arr",
                    )?
                };
                let elem_val = try_compile_expr_as_val!(self, *val, *elem_ty);
                let idx_ty = self.context.i64_type();
                for idx in 0..len {
                    let elem_ptr = unsafe {
                        self.builder.build_in_bounds_gep(
                            arr_ty,
                            arr_ptr,
                            &[idx_ty.const_zero(), idx_ty.const_int(idx as u64, false)],
                            "",
                        )
                    }?;
                    self.build_store(elem_ptr, elem_val, elem_ty.as_ref())?;
                }
                stack_val(arr_ptr)
            },
            &ExprKind::Dot { lhs, lhs_ty, rhs } => match lhs_ty {
                Type::Never => return Ok(Symbol::Never),
                Type::Int { .. } => todo!(),
                Type::Bool => todo!(),
                Type::Float { .. } => todo!(),
                Type::Ptr { .. } => todo!(),
                Type::Slice { .. } => todo!(),
                Type::Array { .. } => todo!(),
                Type::Struct { fields } => {
                    let lhs = lhs.unwrap_debug();
                    let struct_ty = self.type_table[&fields];
                    let Symbol::Stack(struct_ptr) = try_not_never!(self.compile_expr(lhs, lhs_ty)?)
                    else {
                        panic!()
                    };
                    let (field_idx, _) = fields.find_field(&rhs.text).unwrap_debug();
                    stack_val(self.builder.build_struct_gep(
                        struct_ty,
                        struct_ptr,
                        field_idx as u32,
                        &rhs.text,
                    )?)
                },
                Type::Union { fields } => {
                    let lhs = lhs.unwrap_debug();
                    let union_ty = self.type_table[&fields];
                    let Symbol::Stack(union_ptr) = try_not_never!(self.compile_expr(lhs, lhs_ty)?)
                    else {
                        panic!()
                    };
                    stack_val(self.builder.build_struct_gep(union_ty, union_ptr, 0, &rhs.text)?)
                },
                Type::Enum { .. } => todo!(),
                Type::Type(ty) => match out_ty {
                    Type::Enum { variants } => {
                        let (tag, _) = variants.find_field(&rhs.text).unwrap_debug();
                        let _ = self.llvm_type(*ty);
                        self.compile_enum_val(variants, tag, None, write_target.take())
                    },
                    Type::EnumVariant { enum_ty, idx } => {
                        let Type::Enum { variants } = *enum_ty else { unreachable_debug() };
                        Ok(Symbol::EnumVariant { variants, idx })
                    },
                    _ => unreachable_debug(),
                },
                _ => unreachable_debug(),
            },
            &ExprKind::Index { lhs, idx } => {
                let idx_val = try_compile_expr_as_val!(self, idx.expr, idx.ty);

                let lhs_sym = self.compile_typed_expr(lhs)?;
                let (ptr, elem_ty) = match lhs.ty {
                    Type::Never => return Ok(Symbol::Never),
                    Type::Slice { elem_ty } => {
                        let slice = self.sym_as_val(lhs_sym, lhs.ty)?.struct_val();
                        let ptr =
                            self.builder.build_extract_value(slice, 0, "")?.into_pointer_value();
                        (ptr, elem_ty)
                    },
                    Type::Array { elem_ty, .. } => {
                        let Symbol::Stack(arr_ptr) = lhs_sym else { unreachable_debug() };
                        (arr_ptr, elem_ty)
                    },
                    _ => unreachable_debug(),
                };

                let llvm_elem_ty = self.llvm_type(*elem_ty).basic_ty();
                match idx.ty {
                    Type::Int { .. } => {
                        stack_val(self.build_gep(llvm_elem_ty, ptr, &[idx_val.int_val()])?)
                    },
                    Type::Range { .. } => {
                        let range_val = idx_val.struct_val();
                        let start =
                            self.builder.build_extract_value(range_val, 0, "")?.into_int_value();
                        let end =
                            self.builder.build_extract_value(range_val, 1, "")?.into_int_value();

                        let ptr = unsafe {
                            self.builder.build_in_bounds_gep(llvm_elem_ty, ptr, &[start], "")?
                        };
                        let len = self.builder.build_int_sub(end, start, "")?;
                        self.build_slice(ptr, len)
                    },
                    _ => unreachable!(),
                }
            },
            ExprKind::Call { func, args, .. } => match self.compile_typed_expr(*func)? {
                Symbol::Function { params, val: func } => {
                    let args = params
                        .as_ref()
                        .iter()
                        .enumerate()
                        .map(|(idx, param)| {
                            let arg = args.get(idx).copied().or(param.default).unwrap_debug();
                            Ok(match self.compile_expr(arg, param.ty)? {
                                Symbol::Stack(ptr) => BasicMetadataValueEnum::from(ptr),
                                Symbol::Register(val) => val.basic_metadata_val(),
                                _ => todo!(),
                            })
                        })
                        .collect::<CodegenResult<Vec<_>>>()?;
                    let ret = self.builder.build_call(func, &args, "call")?;
                    match out_ty {
                        // Type::Never => Ok(Symbol::Never),
                        _ => reg(ret),
                    }
                },
                Symbol::EnumVariant { variants, idx } => {
                    debug_assert!(args.len() <= 1);
                    self.compile_enum_val(variants, idx, args.get(0).copied(), write_target.take())
                },
                _ => unreachable_debug(),
            },
            ExprKind::UnaryOp { kind, expr, .. } => {
                //let v = try_compile_expr_as_val!(self, *expr, expr_ty);
                let v = try_not_never!(self.compile_expr(*expr, out_ty)?);
                match kind {
                    UnaryOpKind::AddrOf => {
                        return reg(match v {
                            Symbol::Stack(ptr_value) => ptr_value,
                            Symbol::Register(val) => self.build_stack_alloc_as_ptr(val, &out_ty)?,
                            _ => todo!(),
                        });
                    },
                    UnaryOpKind::AddrMutOf => todo!(),
                    _ => {},
                }

                let v = self.sym_as_val(v, out_ty)?;
                match kind {
                    UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf => unreachable_debug(),
                    /*
                    UnaryOpKind::Deref => reg(self.build_load(
                        self.llvm_type(expr_ty).basic_ty(),
                        v.ptr_val(),
                        "deref",
                    )?),
                    */
                    UnaryOpKind::Deref => stack_val(v.ptr_val()),
                    UnaryOpKind::Not => match out_ty {
                        Type::Bool => reg(self.builder.build_not(v.bool_val(), "not")?),
                        _ => todo!(),
                    },
                    UnaryOpKind::Neg => match out_ty {
                        Type::Int { is_signed: true, .. } => {
                            reg(self.builder.build_int_neg(v.int_val(), "neg")?)
                        },
                        Type::Float { .. } => {
                            reg(self.builder.build_float_neg(v.float_val(), "neg")?)
                        },
                        _ => todo!(),
                    },
                    UnaryOpKind::Try => todo!(),
                }
            },
            &ExprKind::BinOp { lhs, op, rhs, arg_ty } => {
                let ty = op.finalize_arg_type(arg_ty, out_ty);
                let lhs_val = try_compile_expr_as_val!(self, lhs, ty);
                if arg_ty == Type::Bool && matches!(op, BinOpKind::And | BinOpKind::Or) {
                    return self.build_bool_short_circuit_binop(lhs_val.bool_val(), rhs, op);
                }
                let rhs_val = try_compile_expr_as_val!(self, rhs, ty);
                if let Type::Range { elem_ty } | Type::RangeInclusive { elem_ty } = out_ty {
                    self.build_range(lhs_val, rhs_val, *elem_ty)
                } else {
                    match ty {
                        Type::Int { is_signed, .. } => self
                            .build_int_binop(lhs_val.int_val(), rhs_val.int_val(), is_signed, op)
                            .map(reg_sym),
                        Type::Float { .. } => self
                            .build_float_binop(lhs_val.float_val(), rhs_val.float_val(), op)
                            .map(reg_sym),
                        Type::Bool => {
                            self.build_bool_binop(lhs_val.bool_val(), rhs_val.bool_val(), op)
                        },
                        t => todo!("{:?}", t),
                    }
                }
            },
            &ExprKind::Assign { lhs, rhs } => {
                let Symbol::Stack(stack_var) = self.compile_typed_expr(lhs)? else {
                    unreachable_debug()
                };
                let _ = self.compile_expr_with_write_target(rhs, lhs.ty, Some(stack_var))?;
                //let rhs = try_compile_expr_as_val!(self, rhs, lhs.ty);
                //self.build_store(stack_var, rhs, &lhs.ty)?;
                Ok(Symbol::Void)
            },
            &ExprKind::BinOpAssign { lhs, op, rhs } => {
                let Symbol::Stack(stack_var) = self.compile_typed_expr(lhs)? else {
                    unreachable_debug()
                };
                let arg_ty = lhs.ty;
                let lhs_llvm_ty = self.llvm_type(arg_ty).basic_ty();
                let lhs_val = self.build_load(lhs_llvm_ty, stack_var, "lhs", &lhs.ty)?;

                if arg_ty == Type::Bool && matches!(op, BinOpKind::And | BinOpKind::Or) {
                    return self.build_bool_short_circuit_binop(lhs_val.into_int_value(), rhs, op);
                }

                let rhs_val = try_compile_expr_as_val!(self, rhs, arg_ty);
                let binop_res = match arg_ty {
                    Type::Int { is_signed, .. } => self.build_int_binop(
                        lhs_val.into_int_value(),
                        rhs_val.int_val(),
                        is_signed,
                        op,
                    )?,
                    Type::Float { .. } => {
                        self.build_float_binop(lhs_val.into_float_value(), rhs_val.float_val(), op)?
                    },
                    Type::Bool { .. } => {
                        panic!("{:#?}", expr.kind);
                        //self.build_bool_binop(lhs.into_int_value(),
                        // rhs.bool_val(), op)?
                    },
                    t => todo!("{:?}", t),
                };
                self.build_store(stack_var, binop_res, &lhs.ty)?;
                Ok(Symbol::Void)
            },
            ExprKind::VarDecl(VarDecl { markers, ident, ty, default: init, .. }) => {
                let var_name = &*ident.text;
                let prev_var_name = self.cur_var_name.replace(Ptr::from(var_name));

                let sym = if let Some(ExprKind::Fn(f)) = init.map(|p| &p.as_ref().kind) {
                    self.compile_fn(var_name, f)?
                } else if let Some(init) = init {
                    let init = try_not_never!(self.compile_expr(*init, *ty)?);
                    if markers.is_mut
                        && let Symbol::Register(init_val) = init
                    {
                        let stack_ty = self.llvm_type(*ty).basic_ty();
                        let stack_ptr = self.build_alloca(stack_ty, &var_name, ty)?;
                        self.build_store(stack_ptr, init_val, ty)?;
                        Symbol::Stack(stack_ptr)
                    } else {
                        init
                    }
                } else {
                    let stack_ty = self.llvm_type(*ty).basic_ty();
                    let stack_ptr = self.build_alloca(stack_ty, &var_name, ty)?;
                    Symbol::Stack(stack_ptr)
                };

                debug_assert!(!markers.is_mut || matches!(sym, Symbol::Stack(_)));
                let old = self.symbols.insert(var_name.to_string(), sym);
                if old.is_some() {
                    #[cfg(debug_assertions)]
                    println!("LOG: '{var_name}' was shadowed in the same scope");
                }

                self.cur_var_name = prev_var_name;

                debug_assert_matches!(out_ty, Type::Void | Type::Unset);
                Ok(Symbol::Void)
            },
            ExprKind::Extern { ident, ty } => match ty {
                Type::Function(f) => {
                    let val = self.compile_prototype(&ident.text, f).0;
                    let sym = Symbol::Function { params: f.params, val };
                    let _ = self.symbols.insert(ident.text, sym);
                    Ok(Symbol::Void)
                },
                _ => todo!(),
            },
            ExprKind::If { condition, then_body, else_body, .. } => {
                let func = self.cur_fn.unwrap_debug();
                let condition = try_compile_expr_as_val!(self, *condition, Type::Bool).bool_val();

                let mut then_bb = self.context.append_basic_block(func, "then");
                let mut else_bb = self.context.append_basic_block(func, "else");
                let merge_bb = self.context.append_basic_block(func, "ifmerge");

                self.builder.build_conditional_branch(condition, then_bb, else_bb)?;

                self.builder.position_at_end(then_bb);
                let then_sym = self.compile_expr(*then_body, out_ty)?;
                if then_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_sym = if let Some(else_body) = else_body {
                    self.compile_expr(*else_body, out_ty)?
                } else {
                    Symbol::Void
                };
                if else_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                else_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);

                match out_ty {
                    Type::Void => return Ok(Symbol::Void),
                    Type::Never => {
                        self.builder.build_unreachable()?;
                        return Ok(Symbol::Never);
                    },
                    _ => {},
                }

                let branch_ty = self.llvm_type(out_ty).basic_ty();
                let phi = self.builder.build_phi(branch_ty, "ifexpr")?;
                match (
                    self.sym_as_val_checked(then_sym, out_ty)?,
                    self.sym_as_val_checked(else_sym, out_ty)?,
                ) {
                    (Some(then_val), Some(else_val)) => {
                        phi.add_incoming(&[(&then_val, then_bb), (&else_val, else_bb)])
                    },
                    (Some(then_val), None) => phi.add_incoming(&[(&then_val, then_bb)]),
                    (None, Some(else_val)) => phi.add_incoming(&[(&else_val, else_bb)]),
                    (None, None) => unreachable_debug(),
                }
                reg(phi)
            },
            ExprKind::Match { .. } => todo!(),
            ExprKind::For { source, iter_var, body, .. } => {
                let Type::Array { len, elem_ty } = source.ty else {
                    panic!("for loop over other types")
                };
                let arr_ty = self.llvm_type(source.ty).arr_ty();
                let Symbol::Stack(arr_ptr) = try_not_never!(self.compile_typed_expr(*source)?)
                else {
                    panic!()
                };

                let func = self.cur_fn.unwrap_debug();
                let entry_bb = self.builder.get_insert_block().unwrap_debug();
                let cond_bb = self.context.append_basic_block(func, "for.cond");
                let body_bb = self.context.append_basic_block(func, "for.body");
                let inc_bb = self.context.append_basic_block(func, "for.inc");
                let end_bb = self.context.append_basic_block(func, "for.end");

                // entry
                self.builder.build_unconditional_branch(cond_bb)?;

                // cond
                self.builder.position_at_end(cond_bb);
                let idx_ty = self.context.i64_type();
                let idx = self.builder.build_phi(idx_ty, "for.idx")?;
                idx.add_incoming(&[(&idx_ty.const_zero(), entry_bb)]);
                let idx_int = idx.as_basic_value().into_int_value();
                //self.symbols.insert("idx", reg_sym(idx_int));

                let len = idx_ty.const_int(len as u64, false);
                let idx_cmp = self.builder.build_int_compare(
                    IntPredicate::ULT,
                    idx_int,
                    len,
                    "for.idx_cmp",
                )?;

                self.builder.build_conditional_branch(idx_cmp, body_bb, end_bb)?;

                // body
                self.builder.position_at_end(body_bb);

                let val =
                    stack_val(self.build_gep(arr_ty, arr_ptr, &[idx_ty.const_zero(), idx_int])?)?;
                self.symbols.insert(&*iter_var.text, val);

                let outer_loop = self.cur_loop.replace(Loop { continue_bb: inc_bb, end_bb });
                let out = self.compile_expr(*body, Type::Void)?;
                self.cur_loop = outer_loop;
                if !matches!(out, Symbol::Never) {
                    self.builder.build_unconditional_branch(inc_bb)?;
                }

                // inc
                self.builder.position_at_end(inc_bb);
                let next_idx =
                    self.builder.build_int_add(idx_int, idx_ty.const_int(1, false), "next_idx")?;
                idx.add_incoming(&[(&next_idx, inc_bb)]);
                self.builder.build_unconditional_branch(cond_bb)?;

                // end
                self.builder.position_at_end(end_bb);
                debug_assert_matches!(out_ty, Type::Void | Type::Unset);
                Ok(Symbol::Void)
            },
            ExprKind::While { condition, body, .. } => {
                let func = self.cur_fn.unwrap_debug();
                let cond_bb = self.context.append_basic_block(func, "while.cond");
                let body_bb = self.context.append_basic_block(func, "while.body");
                let end_bb = self.context.append_basic_block(func, "while.end");

                // entry
                self.builder.build_unconditional_branch(cond_bb)?;

                // cond
                self.builder.position_at_end(cond_bb);
                let cond = try_compile_expr_as_val!(self, *condition, Type::Bool).bool_val();
                self.builder.build_conditional_branch(cond, body_bb, end_bb)?;

                // body
                self.builder.position_at_end(body_bb);
                let outer_loop = self.cur_loop.replace(Loop { continue_bb: cond_bb, end_bb });
                // try_not_never!(self.compile_expr(*body, Type::Void)?);
                let out = self.compile_expr(*body, Type::Void)?;
                self.cur_loop = outer_loop;
                if !matches!(out, Symbol::Never) {
                    self.builder.build_unconditional_branch(cond_bb)?;
                }

                // end
                self.builder.position_at_end(end_bb);
                debug_assert_matches!(out_ty, Type::Void | Type::Unset);
                println!("while body: {:?}", out_ty);
                Ok(Symbol::Void)
            },
            ExprKind::Catch { .. } => todo!(),
            ExprKind::Defer(inner) => {
                self.defer_stack.push_expr(*inner);
                debug_assert_matches!(out_ty, Type::Void | Type::Unset);
                Ok(Symbol::Void)
            },
            ExprKind::Return { expr: val } => {
                if let Some(val) = val {
                    let sym = self.compile_typed_expr(*val)?;
                    self.build_return(sym, val.ty)?;
                } else {
                    self.builder.build_return(None)?;
                }
                // debug_assert_matches!(expr_ty, Type::Never | Type::Unset);
                Ok(Symbol::Never)
            },
            ExprKind::Break { expr } => {
                if expr.is_some() {
                    todo!("break with expr")
                }
                let bb = self.cur_loop.unwrap_debug().end_bb;
                self.builder.build_unconditional_branch(bb)?;
                Ok(Symbol::Never)
            },
            ExprKind::Continue => {
                let bb = self.cur_loop.unwrap_debug().continue_bb;
                self.builder.build_unconditional_branch(bb)?;
                Ok(Symbol::Never)
            },
            ExprKind::Semicolon(_) => todo!(),
        };
        if let Some(target) = write_target
            && let Ok(out) = out
        {
            match out {
                Symbol::Stack(ptr) => {
                    //unreachable_debug(); // TODO
                    let alignment = out_ty.alignment() as u32;
                    let size = self.context.i64_type().const_int(out_ty.size() as u64, false);
                    self.builder.build_memcpy(target, alignment, ptr, alignment, size)?;
                },
                Symbol::Register(val) => {
                    self.build_store(target, val, &out_ty)?;
                },
                _ => {},
            }
        }
        out
    }

    fn compile_typed_expr(&mut self, expr: ExprWithTy) -> CodegenResult<Symbol<'ctx>> {
        self.compile_expr(expr.expr, expr.ty)
    }

    fn compile_enum_val(
        &mut self,
        variants: VarDeclList,
        variant_idx: usize,
        init: Option<Ptr<Expr>>,
        write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let enum_ty = self.type_table[&variants];
        let enum_ptr = if let Some(ptr) = write_target {
            ptr
        } else {
            self.build_alloca(enum_ty, "enum", &Type::Enum { variants })?
        };

        // set tag
        let tag_ptr = enum_ptr;
        let tag_val = self.enum_tag_llvm_type(variants).const_int(variant_idx as u64, false);
        self.build_store(tag_ptr, tag_val, &Self::enum_tag_type(variants))?;

        // set data
        if let Some(init) = init {
            let data_ptr = self.builder.build_struct_gep(enum_ty, enum_ptr, 1, "enum_data")?;
            let _ = self.compile_expr_with_write_target(
                init,
                variants[variant_idx].ty,
                Some(data_ptr),
            )?;
        }

        stack_val(enum_ptr)
    }

    fn compile_fn(&mut self, name: &str, f: &Fn) -> CodegenResult<Symbol<'ctx>> {
        let prev_bb = self.builder.get_insert_block();

        let (fn_val, use_sret) = self.compile_prototype(name, f);
        let mut params_iter = fn_val.get_param_iter();
        let prev_sret_ptr = if use_sret {
            let sret_ptr = params_iter.next().unwrap_debug().into_pointer_value();
            self.sret_ptr.replace(sret_ptr)
        } else {
            self.sret_ptr.take()
        };
        for (idx, param) in params_iter.enumerate() {
            param.set_name(&f.params[idx].ident.text)
        }

        let val = self.compile_fn_body(fn_val, f)?;

        self.sret_ptr = prev_sret_ptr;

        if let Some(prev_bb) = prev_bb {
            self.builder.position_at_end(prev_bb);
        }

        Ok(Symbol::Function { params: f.params, val })
    }

    fn compile_prototype(&mut self, name: &str, f: &Fn) -> (FunctionValue<'ctx>, bool) {
        let ptr_type = BasicMetadataTypeEnum::from(self.ptr_type());
        let ret_type = self.ret_type(f.ret_type);
        let use_sret = ret_type == RetType::SRetParam;
        let mut param_types = if use_sret { vec![ptr_type] } else { Vec::new() };
        f.params
            .iter()
            .map(|VarDecl { ty, .. }| {
                if Self::is_register_passable(*ty) {
                    self.llvm_type(*ty).basic_metadata_ty()
                } else {
                    ptr_type
                }
            })
            .collect_into(&mut param_types);
        let fn_type = match ret_type.into_basic() {
            Some(ret_type) => ret_type.fn_type(&param_types, false),
            None => self.context.void_type().fn_type(&param_types, false),
        };
        let fn_val = self.module.add_function(name, fn_type, Some(Linkage::External));
        if use_sret {
            let llvm_ty = self.llvm_type(f.ret_type).any_ty();
            let sret = self
                .context
                .create_type_attribute(Attribute::get_named_enum_kind_id("sret"), llvm_ty);
            fn_val.add_attribute(inkwell::attributes::AttributeLoc::Param(0), sret);
        }

        (fn_val, use_sret)
    }

    pub fn compile_fn_body(
        &mut self,
        func: FunctionValue<'ctx>,
        f: &Fn,
    ) -> CodegenResult<FunctionValue<'ctx>> {
        let entry = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let outer_fn = self.cur_fn.replace(func);

        self.open_scope();
        let res = try {
            self.symbols.reserve(f.params.len());

            for (param, param_def) in func.get_param_iter().zip(f.params.iter()) {
                let pname = &param_def.ident.text;
                let param = CodegenValue::new(param.as_value_ref());
                let s = if Self::is_register_passable(param_def.ty) {
                    if param_def.markers.is_mut {
                        self.position_builder_at_start(func.get_first_basic_block().unwrap_debug());
                        Symbol::Stack(self.build_stack_alloc_as_ptr(param, &param_def.ty)?)
                    } else {
                        Symbol::Register(param)
                    }
                } else {
                    Symbol::Stack(param.ptr_val())
                };
                self.symbols.insert(pname.to_string(), s);
            }

            let body = self.compile_expr_with_write_target(f.body, f.ret_type, None)?;
            self.build_return(body, f.ret_type)?;

            if func.verify(true) {
                func
            } else {
                #[cfg(debug_assertions)]
                self.module.print_to_stderr();
                unsafe { func.delete() };
                panic_debug("invalid generated function");
                Err(CodegenError::InvalidGeneratedFunction)?
            }
        };
        self.close_scope()?;
        self.cur_fn = outer_fn;
        res
    }

    // -----------------------

    fn build_alloca(
        &self,
        llvm_ty: impl BasicType<'ctx>,
        name: &str,
        ty: &Type,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let ptr = self.builder.build_alloca(llvm_ty, name)?;
        set_alignment(ptr, ty.alignment());
        Ok(ptr)
    }

    fn build_store(
        &self,
        ptr: PointerValue<'ctx>,
        value: impl BasicValue<'ctx>,
        ty: &Type,
    ) -> CodegenResult<InstructionValue<'ctx>> {
        let build_instruction = self.builder.build_store(ptr, value)?;
        set_alignment(build_instruction, ty.alignment());
        Ok(build_instruction)
    }

    fn build_load(
        &self,
        pointee_ty: impl BasicType<'ctx>,
        ptr: PointerValue<'ctx>,
        name: &str,
        ty: &Type,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        assert_ne!(ptr.get_name(), std::ffi::CString::new("inner").unwrap().as_c_str());
        let out = self.builder.build_load(pointee_ty, ptr, name)?;
        set_alignment(out, ty.alignment());
        Ok(out)
    }

    fn build_return(&mut self, ret_sym: Symbol<'ctx>, ret_ty: Type) -> CodegenResult<Symbol<'ctx>> {
        let ret = match (ret_sym, self.ret_type(ret_ty)) {
            (Symbol::Never, _) => return Ok(Symbol::Never),
            (_, RetType::Zst) => None,
            (Symbol::Stack(ptr), RetType::SRetParam) => {
                let sret_ptr = self.sret_ptr.unwrap_debug();
                let alignment = ret_ty.alignment() as u32;
                self.builder.build_memcpy(
                    sret_ptr,
                    alignment,
                    ptr,
                    alignment,
                    self.context.i64_type().const_int(ret_ty.size() as u64, false),
                )?;
                None
            },
            (Symbol::Register(val), RetType::SRetParam) => {
                let sret_ptr = self.sret_ptr.unwrap_debug();
                self.build_store(sret_ptr, val, &ret_ty)?;
                None
            },
            (Symbol::Stack(ptr), RetType::Basic(ty)) => {
                Some(self.build_load(ty, ptr, "", &ret_ty)?)
            },
            (Symbol::Register(val), RetType::Basic(_)) => Some(val.basic_val()),
            _ => panic_debug("unexpected symbol"),
        };
        match ret {
            Some(ret) => self.builder.build_return(Some(&ret))?,
            None => self.builder.build_return(None)?,
        };
        Ok(Symbol::Never)
    }

    #[inline]
    fn build_gep(
        &self,
        pointee_ty: impl BasicType<'ctx>,
        ptr: PointerValue<'ctx>,
        ordered_indexes: &[IntValue<'ctx>],
    ) -> CodegenResult<PointerValue<'ctx>> {
        Ok(unsafe { self.builder.build_in_bounds_gep(pointee_ty, ptr, ordered_indexes, "")? })
    }

    pub fn build_int_binop(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
        is_signed: bool,
        op: BinOpKind,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
            Ok(CodegenValue::new(val.as_value_ref()))
        }

        macro_rules! cmp {
            ($pred:ident, $var_name:expr) => {
                ret(self.builder.build_int_compare(IntPredicate::$pred, lhs, rhs, $var_name)?)
            };
        }

        match op {
            BinOpKind::Mul => ret(self.builder.build_int_mul(lhs, rhs, "mul")?),
            BinOpKind::Div => ret(if is_signed {
                self.builder.build_int_signed_div(lhs, rhs, "div")?
            } else {
                self.builder.build_int_unsigned_div(lhs, rhs, "div")?
            }),
            BinOpKind::Mod => ret(if is_signed {
                self.builder.build_int_signed_rem(lhs, rhs, "mod")?
            } else {
                self.builder.build_int_unsigned_rem(lhs, rhs, "mod")?
            }),
            BinOpKind::Add => ret(self.builder.build_int_add(lhs, rhs, "add")?),
            BinOpKind::Sub => ret(self.builder.build_int_sub(lhs, rhs, "sub")?),
            BinOpKind::ShiftL => ret(self.builder.build_left_shift(lhs, rhs, "shl")?),
            BinOpKind::ShiftR => ret(self.builder.build_right_shift(lhs, rhs, is_signed, "shr")?),
            BinOpKind::BitAnd => ret(self.builder.build_and(lhs, rhs, "bitand")?),
            BinOpKind::BitXor => ret(self.builder.build_xor(lhs, rhs, "bitxor")?),
            BinOpKind::BitOr => ret(self.builder.build_or(lhs, rhs, "bitor")?),
            BinOpKind::Eq => cmp!(EQ, "eq"),
            BinOpKind::Ne => cmp!(NE, "ne"),
            BinOpKind::Lt if is_signed => cmp!(SLT, "lt"),
            BinOpKind::Lt => cmp!(ULT, "lt"),
            BinOpKind::Le if is_signed => cmp!(SLE, "le"),
            BinOpKind::Le => cmp!(ULE, "le"),
            BinOpKind::Gt if is_signed => cmp!(SGT, "gt"),
            BinOpKind::Gt => cmp!(UGT, "gt"),
            BinOpKind::Ge if is_signed => cmp!(SGE, "ge"),
            BinOpKind::Ge => cmp!(UGE, "ge"),

            BinOpKind::And | BinOpKind::Or => unreachable!(),

            BinOpKind::Range => todo!(),
            BinOpKind::RangeInclusive => todo!(),
        }
    }

    pub fn build_float_binop(
        &mut self,
        lhs: FloatValue<'ctx>,
        rhs: FloatValue<'ctx>,
        op: BinOpKind,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
            Ok(CodegenValue::new(val.as_value_ref()))
        }

        macro_rules! cmp {
            ($pred:ident, $var_name:expr) => {
                ret(self.builder.build_float_compare(FloatPredicate::$pred, lhs, rhs, $var_name)?)
            };
        }

        match op {
            BinOpKind::Mul => ret(self.builder.build_float_mul(lhs, rhs, "mul")?),
            BinOpKind::Div => ret(self.builder.build_float_div(lhs, rhs, "div")?),
            BinOpKind::Mod => ret(self.builder.build_float_rem(lhs, rhs, "mod")?),
            BinOpKind::Add => ret(self.builder.build_float_add(lhs, rhs, "add")?),
            BinOpKind::Sub => ret(self.builder.build_float_sub(lhs, rhs, "sub")?),
            BinOpKind::Eq => cmp!(OEQ, "eq"),
            BinOpKind::Ne => cmp!(ONE, "ne"),
            BinOpKind::Lt => cmp!(OLT, "lt"),
            BinOpKind::Le => cmp!(OLE, "le"),
            BinOpKind::Gt => cmp!(OGT, "gt"),
            BinOpKind::Ge => cmp!(OGE, "ge"),
            BinOpKind::Range => todo!(),
            BinOpKind::RangeInclusive => todo!(),
            _ => unreachable!(),
        }
    }

    fn build_bool_binop(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
        op: BinOpKind,
    ) -> CodegenResult<Symbol<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
            Ok(Symbol::Register(CodegenValue::new(val.as_value_ref())))
        }

        match op {
            BinOpKind::Eq => {
                ret(self.builder.build_int_compare(IntPredicate::EQ, lhs, rhs, "eq")?)
            },
            BinOpKind::Ne => {
                ret(self.builder.build_int_compare(IntPredicate::NE, lhs, rhs, "ne")?)
            },

            // false = 0, true = 1
            BinOpKind::Lt => {
                ret(self.builder.build_int_compare(IntPredicate::ULT, lhs, rhs, "lt")?)
            },
            BinOpKind::Le => {
                ret(self.builder.build_int_compare(IntPredicate::ULE, lhs, rhs, "le")?)
            },
            BinOpKind::Gt => {
                ret(self.builder.build_int_compare(IntPredicate::UGT, lhs, rhs, "gt")?)
            },
            BinOpKind::Ge => {
                ret(self.builder.build_int_compare(IntPredicate::UGE, lhs, rhs, "ge")?)
            },
            BinOpKind::And | BinOpKind::Or => unreachable_debug(),
            BinOpKind::BitAnd => ret(self.builder.build_and(lhs, rhs, "and")?),
            BinOpKind::BitOr => ret(self.builder.build_or(lhs, rhs, "or")?),
            BinOpKind::BitXor => ret(self.builder.build_xor(lhs, rhs, "xor")?),
            BinOpKind::Range => todo!(),
            BinOpKind::RangeInclusive => todo!(),
            _ => unreachable!(),
        }
    }

    fn build_bool_short_circuit_binop(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: Ptr<Expr>,
        op: BinOpKind,
    ) -> CodegenResult<Symbol<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
            Ok(Symbol::Register(CodegenValue::new(val.as_value_ref())))
        }

        match op {
            BinOpKind::And => ret({
                let func = self.cur_fn.unwrap_debug();
                let entry_bb = self.builder.get_insert_block().unwrap_debug();
                let mut rhs_bb = self.context.append_basic_block(func, "and.rhs");
                let merge_bb = self.context.append_basic_block(func, "and.merge");

                self.builder.build_conditional_branch(lhs, rhs_bb, merge_bb)?;

                self.builder.position_at_end(rhs_bb);
                let rhs = try_compile_expr_as_val!(self, rhs, Type::Bool);
                self.builder.build_unconditional_branch(merge_bb)?;
                rhs_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(lhs.get_type(), "and")?;
                let false_ = self.context.bool_type().const_zero();
                phi.add_incoming(&[(&false_, entry_bb), (&rhs, rhs_bb)]);
                phi
            }),
            BinOpKind::Or => ret({
                let func = self.cur_fn.unwrap_debug();
                let entry_bb = self.builder.get_insert_block().unwrap_debug();
                let mut rhs_bb = self.context.append_basic_block(func, "or.rhs");
                let merge_bb = self.context.append_basic_block(func, "or.merge");

                self.builder.build_conditional_branch(lhs, merge_bb, rhs_bb)?;

                self.builder.position_at_end(rhs_bb);
                let rhs = try_compile_expr_as_val!(self, rhs, Type::Bool);
                self.builder.build_unconditional_branch(merge_bb)?;
                rhs_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(lhs.get_type(), "and")?;
                let true_ = self.context.bool_type().const_all_ones();
                phi.add_incoming(&[(&true_, entry_bb), (&rhs, rhs_bb)]);
                phi
            }),
            _ => unreachable!(),
        }
    }

    fn build_range(
        &mut self,
        lhs_val: CodegenValue<'ctx>,
        rhs_val: CodegenValue<'ctx>,
        elem_ty: Type,
    ) -> CodegenResult<Symbol<'ctx>> {
        let range = self.range_type(elem_ty).get_undef();
        let range = self.builder.build_insert_value(range, lhs_val, 0, "")?;
        reg(self.builder.build_insert_value(range, rhs_val, 1, "range")?)
    }

    fn build_slice(
        &mut self,
        ptr: PointerValue<'ctx>,
        len: IntValue<'ctx>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let slice = self.slice_ty().get_undef();
        let slice = self.builder.build_insert_value(slice, ptr, 0, "")?;
        reg(self.builder.build_insert_value(slice, len, 1, "slice")?)
    }

    /// LLVM does not make a distinction between signed and unsigned integer
    /// type
    fn int_type(&self, bits: u32) -> IntType<'ctx> {
        match bits {
            8 => self.context.i8_type(),
            16 => self.context.i16_type(),
            32 => self.context.i32_type(),
            64 => self.context.i64_type(),
            128 => self.context.i128_type(),
            bits => self.context.custom_width_int_type(bits),
        }
    }

    fn float_type(&self, bits: u32) -> FloatType<'ctx> {
        match bits {
            16 => self.context.f16_type(),
            32 => self.context.f32_type(),
            64 => self.context.f64_type(),
            128 => self.context.f128_type(),
            bits => todo!("{bits}-bit float"),
        }
    }

    fn ptr_type(&self) -> PointerType<'ctx> {
        self.context.ptr_type(AddressSpace::default())
    }

    fn struct_type(&mut self, fields: VarDeclList, name: Option<Ptr<str>>) -> StructType<'ctx> {
        if let Some(t) = self.type_table.get(&fields).copied() {
            return t;
        }

        let field_types =
            fields.iter().map(|f| self.llvm_type(f.ty).basic_ty()).collect::<Vec<_>>();
        let ty = self.struct_type_inner(&field_types, name, false);
        let a = self.type_table.insert(fields, ty);
        assert!(a.is_none());
        ty
    }

    fn struct_type_inner<'a>(
        &mut self,
        fields: &[BasicTypeEnum<'ctx>],
        name: Option<Ptr<str>>,
        packed: bool,
    ) -> StructType<'ctx> {
        match name.as_ref().map(Ptr::as_ref) {
            Some(name) => {
                let ty = self.context.opaque_struct_type(name);
                if !ty.set_body(fields, packed) {
                    panic!("invalid struct type")
                }
                ty
            },
            None => self.context.struct_type(fields, packed),
        }
    }

    fn union_type(&mut self, fields: VarDeclList, name: Option<Ptr<str>>) -> StructType<'ctx> {
        if let Some(t) = self.type_table.get(&fields).copied() {
            return t;
        }

        let ty = self.union_type_inner(fields, name);
        self.type_table.insert(fields, ty);
        ty
    }

    fn union_type_inner(
        &mut self,
        fields: VarDeclList,
        name: Option<Ptr<str>>,
    ) -> StructType<'ctx> {
        for f in fields.iter() {
            // if f.ty is a custom type, this will compile the type.
            let _ = self.llvm_type(f.ty);
        }
        let fields = if let Some(biggest_alignment_field) = fields
            .iter()
            .map(|f| f.ty)
            .filter(|t| t.size() > 0)
            .max_by(|a, b| a.alignment().cmp(&b.alignment()))
        {
            let remaining_size = Type::Union { fields }.size() - biggest_alignment_field.size();
            let biggest_alignment_field = self.llvm_type(biggest_alignment_field).basic_ty();
            let remaining_size_field =
                self.context.i8_type().array_type((remaining_size) as u32).as_basic_type_enum();
            &[biggest_alignment_field, remaining_size_field] as &[_]
        } else {
            &[]
        };
        self.struct_type_inner(fields, name, false)
    }

    fn enum_type(&mut self, variants: VarDeclList, name: Option<Ptr<str>>) -> StructType<'ctx> {
        if let Some(t) = self.type_table.get(&variants).copied() {
            return t;
        }

        let tag_ty = self.enum_tag_llvm_type(variants).as_basic_type_enum();
        let val_ty = self.union_type_inner(variants, None).as_basic_type_enum();
        let ty = self.struct_type_inner(&[tag_ty, val_ty], name, false);
        self.type_table.insert(variants, ty);
        ty
    }

    #[inline]
    fn enum_tag_llvm_type(&mut self, variants: VarDeclList) -> IntType<'ctx> {
        let variant_count = util::variant_count_to_tag_size_bits(variants.len());
        self.context.custom_width_int_type(variant_count)
        //self.context.i32_type()
    }

    #[inline]
    fn enum_tag_type(variants: VarDeclList) -> Type {
        let variant_count = util::variant_count_to_tag_size_bits(variants.len());
        Type::Int { bits: variant_count, is_signed: false }
    }

    #[inline]
    fn range_type(&mut self, elem_ty: Type) -> StructType<'ctx> {
        let e = self.llvm_type(elem_ty).basic_ty();
        self.struct_type_inner(&[e; 2], None, false)
    }

    #[inline]
    fn slice_ty(&mut self) -> StructType<'ctx> {
        self.struct_type_inner(
            &[self.ptr_type().as_basic_type_enum(), self.context.i64_type().as_basic_type_enum()],
            None,
            false,
        )
    }

    fn llvm_type(&mut self, ty: Type) -> CodegenType<'ctx> {
        CodegenType::new(match ty {
            Type::Void => self.context.void_type().as_type_ref(),
            Type::Never => todo!(),
            Type::Int { bits, .. } => self.int_type(bits).as_type_ref(),
            Type::Bool => self.context.bool_type().as_type_ref(),
            Type::Float { bits } => self.float_type(bits).as_type_ref(),
            Type::Function(_) => todo!(),
            Type::Ptr { .. } => self.ptr_type().as_type_ref(),
            Type::Slice { .. } => self.slice_ty().as_type_ref(),
            Type::Array { len: count, elem_ty: ty } => {
                self.llvm_type(*ty).basic_ty().array_type(count as u32).as_type_ref()
            },
            Type::Struct { fields } => self.struct_type(fields, None).as_type_ref(),
            Type::Union { fields } => self.union_type(fields, None).as_type_ref(),
            Type::Enum { variants } => self.enum_type(variants, None).as_type_ref(),
            Type::Range { elem_ty } | Type::RangeInclusive { elem_ty } => {
                self.range_type(*elem_ty).as_type_ref()
            },
            Type::Type(_) => todo!(),

            Type::IntLiteral
            | Type::FloatLiteral
            | Type::EnumVariant { .. }
            | Type::Unset
            | Type::Unevaluated(_) => panic_debug("unfinished type"),
        })
    }

    /// I gave up trying to implement the C calling convention.
    ///
    /// [`None`] means void
    ///
    /// See <https://discourse.llvm.org/t/questions-about-c-calling-conventions/72414>
    /// TODO: See <https://mcyoung.xyz/2024/04/17/calling-convention/>
    fn ret_type(&mut self, ty: Type) -> RetType<'ctx> {
        let size = ty.size();
        match ty {
            _ if size == 0 => RetType::Zst,
            _ if size > 16 => RetType::SRetParam,
            Type::Struct { fields } => self.ret_type_for_struct(fields.into_type_iter(), size),
            Type::Union { .. } => {
                RetType::Basic(self.llvm_type(self.ret_type_for_union(size)).basic_ty())
            },
            Type::Enum { variants } => {
                let fields = [
                    Type::Int {
                        bits: util::variant_count_to_tag_size_bits(variants.len()),
                        is_signed: false,
                    },
                    Type::Union { fields: variants },
                ];
                self.ret_type_for_struct(fields.into_iter(), size)
            },
            ty => RetType::Basic(self.llvm_type(ty).basic_ty()),
        }
    }

    fn ret_type_for_struct(
        &mut self,
        field_types: impl DoubleEndedIterator<Item = Type>,
        size: usize,
    ) -> RetType<'ctx> {
        #[derive(Debug, PartialEq)]
        enum PrevState {
            None,
            Int,
            Float,
            FloatFloat,
        }

        let mut new_fields = Vec::with_capacity(size);
        let mut prev_state = PrevState::None;
        let mut prev_bytes: u32 = 0;

        macro_rules! push_prev_state_to_new_fields {
            () => {
                match prev_state {
                    PrevState::None => {},
                    PrevState::Int => new_fields.push(
                        self.context.custom_width_int_type(prev_bytes << 3).as_basic_type_enum(),
                    ),
                    PrevState::Float => {
                        new_fields.push(self.context.f32_type().as_basic_type_enum())
                    },
                    PrevState::FloatFloat => {
                        new_fields.push(self.context.f32_type().vec_type(2).as_basic_type_enum())
                    },
                }
            };
        }

        let mut traverse_stack = field_types.rev().collect::<Vec<_>>();
        while let Some(next) = traverse_stack.pop() {
            if next.size() == 0 {
                continue;
            }
            //println!("{:?} (size: {}) @ {} ({:?})", next, next.size(), prev_bytes, prev_state);

            if get_aligned_offset!(prev_bytes, next.alignment() as u32) >= 8 {
                // finished the first 8 bytes
                push_prev_state_to_new_fields!();
                prev_state = PrevState::None;
                prev_bytes = 0;
            }

            let padding = get_padding!(prev_bytes, next.alignment() as u32);
            prev_bytes += padding;

            let simple_ty = match next {
                Type::Array { len, elem_ty } => {
                    traverse_stack.extend(std::iter::repeat_n(*elem_ty, len));
                    continue;
                },
                Type::Struct { fields } => {
                    traverse_stack.extend(fields.into_type_iter().rev());
                    continue;
                },
                Type::Union { .. } => self.ret_type_for_union(next.size()),
                Type::Enum { variants } => {
                    traverse_stack.push(Type::Int {
                        bits: util::variant_count_to_tag_size_bits(variants.len()),
                        is_signed: false,
                    });
                    traverse_stack.push(Type::Union { fields: variants });
                    continue;
                },
                t => t,
            };

            match simple_ty {
                Type::Void
                | Type::Never
                | Type::Type(_)
                | Type::Array { .. }
                | Type::Struct { .. }
                | Type::Union { .. }
                | Type::Enum { .. } => panic_debug("unflatted type"),
                Type::Int { bits, .. } => {
                    prev_state = PrevState::Int;
                    prev_bytes += bits.div_ceil(8);
                },
                Type::Bool => {
                    prev_state = PrevState::Int;
                    prev_bytes += 1;
                },
                Type::Float { bits: 64 } => {
                    debug_assert_eq!(prev_state, PrevState::None);
                    debug_assert_eq!(prev_bytes, 0);
                    new_fields.push(self.context.f64_type().as_basic_type_enum());
                },
                Type::Float { bits: 32 } => {
                    prev_bytes += 4;
                    prev_state = match prev_state {
                        PrevState::None => PrevState::Float,
                        PrevState::Int => PrevState::Int,
                        PrevState::Float => PrevState::FloatFloat,
                        PrevState::FloatFloat => unreachable_debug(),
                    }
                },
                Type::Float { .. } => todo!("float with other sizes"),
                _ => new_fields.push(self.llvm_type(simple_ty).basic_ty()),
            }
        }

        // finished the last 8 bytes
        push_prev_state_to_new_fields!();

        RetType::Basic(self.struct_type_inner(&new_fields, None, false).as_basic_type_enum())
    }

    fn ret_type_for_union(&self, size: usize) -> Type {
        Type::Int { bits: size as u32 * 8, is_signed: false }
    }

    fn sym_as_val(&mut self, sym: Symbol<'ctx>, ty: Type) -> CodegenResult<CodegenValue<'ctx>> {
        Ok(match sym {
            Symbol::Stack(ptr) => {
                let llvm_ty = self.llvm_type(ty).basic_ty();
                CodegenValue::new(self.build_load(llvm_ty, ptr, "", &ty)?.as_value_ref())
            },
            Symbol::Register(val) => val,
            _ => panic_debug("unexpected symbol"),
        })
    }

    fn sym_as_val_checked(
        &mut self,
        sym: Symbol<'ctx>,
        ty: Type,
    ) -> CodegenResult<Option<CodegenValue<'ctx>>> {
        Ok(Some(match sym {
            Symbol::Stack(ptr) => {
                let llvm_ty = self.llvm_type(ty).basic_ty();
                CodegenValue::new(self.build_load(llvm_ty, ptr, "", &ty)?.as_value_ref())
            },
            Symbol::Register(val) => val,
            _ => return Ok(None),
        }))
    }

    #[inline]
    fn get_symbol(&self, name: &str) -> Symbol<'ctx> {
        *self.symbols.get(name).unwrap_debug()
    }

    fn is_register_passable(ty: Type) -> bool {
        match ty {
            Type::Int { .. }
            | Type::Bool
            | Type::Float { .. }
            | Type::Ptr { .. }
            | Type::Slice { .. }
            | Type::Array { .. }
            | Type::Range { .. }
            | Type::RangeInclusive { .. } => true,
            Type::Void
            | Type::Never
            | Type::Function(_)
            | Type::Struct { .. }
            | Type::Union { .. }
            | Type::Enum { .. }
            | Type::EnumVariant { .. }
            | Type::Type(_) => false,
            Type::IntLiteral | Type::FloatLiteral | Type::Unset | Type::Unevaluated(_) => {
                panic_debug("invalid type")
            },
        }
    }

    pub fn init_target_machine(target_triple: Option<&str>) -> TargetMachine {
        Target::initialize_all(&InitializationConfig::default());

        let target_triple = target_triple
            .map(TargetTriple::create)
            .unwrap_or_else(TargetMachine::get_default_triple);
        let target = Target::from_triple(&target_triple).unwrap();

        let cpu = "generic";
        let features = "";
        target
            .create_target_machine(
                &target_triple,
                cpu,
                features,
                OptimizationLevel::Aggressive,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .unwrap()
    }

    fn open_scope(&mut self) {
        self.symbols.open_scope();
        self.defer_stack.open_scope();
    }

    fn close_scope(&mut self) -> CodegenResult<()> {
        let res = self.compile_defer_exprs();
        self.symbols.close_scope();
        self.defer_stack.close_scope();
        res
    }

    #[inline]
    fn compile_defer_exprs(&mut self) -> CodegenResult<()> {
        let exprs = unsafe { forget_lifetime(self.defer_stack.get_cur_scope()) };
        for expr in exprs.iter().rev() {
            self.compile_expr(*expr, Type::Void)?;
        }
        Ok(())
    }

    fn position_builder_at_start(&self, entry: BasicBlock<'ctx>) {
        match entry.get_first_instruction() {
            Some(first_instr) => self.builder.position_before(&first_instr),
            None => self.builder.position_at_end(entry),
        }
    }

    fn build_stack_alloc_as_ptr(
        &self,
        v: impl BasicValue<'ctx>,
        ty: &Type,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let v = v.as_basic_value_enum();
        let alloca = self.build_alloca(v.get_type(), "", ty).unwrap_debug();
        self.build_store(alloca, v, ty)?;
        Ok(alloca)
    }
}

// optimizations
impl<'ctx> Codegen<'ctx> {
    pub fn optimize_module(
        &self,
        target_machine: &TargetMachine,
        level: u8,
    ) -> Result<(), CodegenError> {
        assert!((0..=3).contains(&level));
        let passes = format!("default<O{}>", level);

        // TODO: custom passes:
        //let passes = format!(
        //   "module(cgscc(inline),function({}))",
        //   ["instcombine", "reassociate", "gvn", "simplifycfg",
        //"mem2reg",].join(","), );

        self.module
            .run_passes(&passes, target_machine, PassBuilderOptions::create())
            .map_err(|err| CodegenError::CannotOptimizeModule(err))
    }

    pub fn compile_to_obj_file(
        &self,
        target_machine: &TargetMachine,
        obj_file_path: &Path,
    ) -> Result<(), CodegenError> {
        std::fs::create_dir_all(obj_file_path.parent().unwrap()).unwrap();
        target_machine
            .write_to_file(&self.module, inkwell::targets::FileType::Object, obj_file_path)
            .map_err(|err| CodegenError::CannotCompileObjFile(err))
    }

    pub fn jit_run_fn<Ret>(&self, fn_name: &str, opt: OptimizationLevel) -> CodegenResult<Ret> {
        match self.module.create_jit_execution_engine(opt) {
            Ok(jit) => {
                Ok(unsafe { jit.get_function::<unsafe extern "C" fn() -> Ret>(fn_name)?.call() })
            },
            Err(err) => Err(CodegenError::CannotCreateJit(err)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symbol<'ctx> {
    Void,
    Never,
    Stack(PointerValue<'ctx>),
    //Register(AnyValueEnum<'ctx>),
    Register(CodegenValue<'ctx>),
    Function {
        /// TODO: think of a better way to store the fn definition
        params: VarDeclList,
        val: FunctionValue<'ctx>,
    },
    StructDef {
        fields: VarDeclList,
        ty: StructType<'ctx>,
    },
    UnionDef {
        fields: VarDeclList,
        ty: StructType<'ctx>,
    },
    EnumDef {
        variants: VarDeclList,
        ty: StructType<'ctx>,
    },
    EnumVariant {
        variants: VarDeclList,
        idx: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodegenValue<'ctx> {
    val: *mut LLVMValue,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> CodegenValue<'ctx> {
    pub fn new(val: *mut LLVMValue) -> CodegenValue<'ctx> {
        CodegenValue { val, _marker: PhantomData }
    }

    // /// for zero sized types: `self.val == null`
    // pub fn new_zst(ty: Type) -> CodegenValue<'ctx> {
    //     debug_assert_matches!(ty, Type::Void | Type::Never);
    //     CodegenValue::new(std::ptr::null_mut(), ty)
    // }

    // pub fn try_new_zst(ty: Type) -> Option<CodegenValue<'ctx>> {
    //     match ty {
    //         Type::Void | Type::Never => Some(CodegenValue::new_zst(ty)),
    //         _ => None,
    //     }
    // }

    // pub fn as_type(&self) -> CodegenType {
    //     CodegenType::new(unsafe { LLVMTypeOf(self.val) }, self.ty)
    // }

    pub fn int_val(&self) -> IntValue<'ctx> {
        // debug_assert_matches!(self.ty, Type::Int { .. });
        unsafe { IntValue::new(self.val) }
    }

    pub fn bool_val(&self) -> IntValue<'ctx> {
        // debug_assert_matches!(self.ty, Type::Int { .. });
        unsafe { IntValue::new(self.val) }
    }

    pub fn float_val(&self) -> FloatValue<'ctx> {
        // debug_assert_matches!(self.ty, Type::Float { .. });
        unsafe { FloatValue::new(self.val) }
    }

    pub fn ptr_val(&self) -> PointerValue<'ctx> {
        // debug_assert_matches!(self.ty, Type::Ptr(_));
        unsafe { PointerValue::new(self.val) }
    }

    pub fn struct_val(&self) -> StructValue<'ctx> {
        // debug_assert_matches!(self.ty, Type::Struct { .. });
        unsafe { StructValue::new(self.val) }
    }

    pub fn basic_val(&self) -> BasicValueEnum<'ctx> {
        // TODO: // debug_assert?
        unsafe { BasicValueEnum::new(self.val) }
    }

    pub fn basic_metadata_val(&self) -> BasicMetadataValueEnum<'ctx> {
        // TODO: // debug_assert?
        BasicMetadataValueEnum::try_from(self.any_val()).unwrap_debug()
    }

    pub fn any_val(&self) -> AnyValueEnum<'ctx> {
        // TODO: // debug_assert?
        unsafe { AnyValueEnum::new(self.val) }
    }

    // pub fn is_never(&self) -> bool {
    //     self.ty == Type::Never
    // }
}

unsafe impl<'ctx> AsValueRef for CodegenValue<'ctx> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val
    }
}

unsafe impl<'ctx> AnyValue<'ctx> for CodegenValue<'ctx> {}
unsafe impl<'ctx> BasicValue<'ctx> for CodegenValue<'ctx> {}

pub struct CodegenType<'ctx> {
    inner: *mut LLVMType,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> CodegenType<'ctx> {
    pub fn new(raw: *mut LLVMType) -> CodegenType<'ctx> {
        CodegenType { inner: raw, _marker: PhantomData }
    }

    pub fn int_ty(&self) -> IntType<'ctx> {
        // // debug_assert_matches!(self.ty, Type::Int { .. });
        unsafe { IntType::new(self.inner) }
    }

    pub fn bool_ty(&self) -> IntType<'ctx> {
        // debug_assert_matches!(self.ty, Type::Bool);
        unsafe { IntType::new(self.inner) }
    }

    pub fn float_ty(&self) -> FloatType<'ctx> {
        // debug_assert_matches!(self.ty, Type::Float { .. });
        unsafe { FloatType::new(self.inner) }
    }

    pub fn ptr_ty(&self) -> PointerType<'ctx> {
        // debug_assert_matches!(self.ty, Type::Ptr(..));
        unsafe { PointerType::new(self.inner) }
    }

    pub fn arr_ty(&self) -> ArrayType<'ctx> {
        unsafe { ArrayType::new(self.inner) }
    }

    pub fn struct_ty(&self) -> StructType<'ctx> {
        unsafe { StructType::new(self.inner) }
    }

    pub fn basic_ty(&self) -> BasicTypeEnum<'ctx> {
        unsafe { BasicTypeEnum::new(self.inner) }
    }

    pub fn basic_metadata_ty(&self) -> BasicMetadataTypeEnum<'ctx> {
        BasicMetadataTypeEnum::try_from(self.any_ty()).unwrap_debug()
    }

    pub fn any_ty(&self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(self.inner) }
    }
}

fn reg<'ctx>(v: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
    Ok(Symbol::Register(CodegenValue::new(v.as_value_ref())))
}

fn reg_sym<'ctx>(v: impl AnyValue<'ctx>) -> Symbol<'ctx> {
    Symbol::Register(CodegenValue::new(v.as_value_ref()))
}

fn stack_val<'ctx>(ptr: PointerValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
    Ok(Symbol::Stack(ptr))
}

#[derive(Debug, Clone, Copy)]
pub struct Loop<'ctx> {
    continue_bb: BasicBlock<'ctx>,
    end_bb: BasicBlock<'ctx>,
}

#[allow(unused)]
mod return_strategy {
    use super::Codegen;
    use inkwell::{
        attributes::Attribute,
        context::Context,
        types::{BasicType, BasicTypeEnum},
    };

    /// x86_64 clang: for 8 byte >= size
    /// rust 1.81: for 16 byte >= size
    /// ```llvm
    /// %struct.S = type { [6 x i8] }
    ///
    /// define dso_local i48 @test() {
    /// entry:
    ///   %retval = alloca %struct.S, align 1
    ///   %coerce.dive.coerce = alloca i48, align 8
    ///   %x = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   %arrayidx = getelementptr inbounds [6 x i8], ptr %x, i64 0, i64 5
    ///   store i8 69, ptr %arrayidx, align 1
    ///   %coerce.dive = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %coerce.dive.coerce, ptr align 1 %coerce.dive, i64 6, i1 false)
    ///   %0 = load i48, ptr %coerce.dive.coerce, align 8
    ///   ret i48 %0
    /// }
    ///
    /// declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1
    /// ```
    pub fn one_int<'ctx>(c: &mut Codegen<'ctx>, bytes: usize) -> BasicTypeEnum<'ctx> {
        debug_assert!(bytes <= 8);
        c.context.custom_width_int_type(bytes as u32 * 8).as_basic_type_enum()
    }

    /// x86_64 clang: for 16 byte >= size > 8 byte
    /// ```llvm
    /// %struct.S = type { [14 x i8] }
    ///
    /// define dso_local { i64, i48 } @test() {
    /// entry:
    ///   %retval = alloca %struct.S, align 1
    ///   %coerce.dive.coerce = alloca { i64, i48 }, align 8
    ///   %x = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   %arrayidx = getelementptr inbounds [14 x i8], ptr %x, i64 0, i64 5
    ///   store i8 69, ptr %arrayidx, align 1
    ///   %coerce.dive = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %coerce.dive.coerce, ptr align 1 %coerce.dive, i64 14, i1 false)
    ///   %0 = load { i64, i48 }, ptr %coerce.dive.coerce, align 8
    ///   ret { i64, i48 } %0
    /// }
    ///
    /// declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1
    /// ```
    pub fn two_int<'ctx>(c: &mut Codegen<'ctx>, bytes: usize) -> BasicTypeEnum<'ctx> {
        debug_assert!(bytes > 8 && bytes <= 16);
        let ty_i64 = c.context.i64_type().as_basic_type_enum();
        let remaining_bits = (bytes as u32 - 8) * 8;
        let ty2 = c.context.custom_width_int_type(remaining_bits).as_basic_type_enum();
        c.context.struct_type(&[ty_i64, ty2], false).as_basic_type_enum()
    }

    /// x86_64 clang: for size > 16 byte
    /// rust 1.81: for size > 8 byte
    /// ```llvm
    /// %struct.S = type { [17 x i8] }
    ///
    /// define dso_local void @test(ptr dead_on_unwind noalias writable sret(%struct.S) align 1 %agg.result) {
    /// entry:
    ///   %x = getelementptr inbounds nuw %struct.S, ptr %agg.result, i32 0, i32 0
    ///   %arrayidx = getelementptr inbounds [17 x i8], ptr %x, i64 0, i64 5
    ///   store i8 69, ptr %arrayidx, align 1
    ///   ret void
    /// }
    /// ```
    pub fn ret_param() -> ! {
        let context: Context = todo!();
        let type_ref = todo!();
        context.create_type_attribute(Attribute::get_named_enum_kind_id("sret"), type_ref);
        todo!()
    }
}

fn set_alignment(val: impl AsValueRef, alignment: usize) {
    unsafe { InstructionValue::new(val.as_value_ref()) }
        .set_alignment(alignment as u32)
        .unwrap_debug()
}

#[derive(Debug, PartialEq)]
enum RetType<'ctx> {
    Basic(BasicTypeEnum<'ctx>),
    Zst,
    SRetParam,
}

impl<'ctx> RetType<'ctx> {
    fn into_basic(self) -> Option<BasicTypeEnum<'ctx>> {
        match self {
            RetType::Basic(basic_type_enum) => Some(basic_type_enum),
            RetType::Zst | RetType::SRetParam => None,
        }
    }
}
