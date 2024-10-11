#![allow(unused)]

use self::jit::Jit;
use crate::{
    ast::{
        BinOpKind, DeclMarkers, Expr, ExprKind, ExprWithTy, Fn, Ident, LitKind, UnaryOpKind,
        VarDecl, VarDeclList,
    },
    cli::DebugOptions,
    defer_stack::DeferStack,
    parser::{ParseError, StmtIter, lexer::Code},
    ptr::Ptr,
    symbol_table::SymbolTable,
    type_::Type,
    util::{UnwrapDebug, display_span_in_code, forget_lifetime, panic_debug, unreachable_debug},
};
pub use inkwell::targets::TargetMachine;
use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    basic_block::BasicBlock,
    builder::{Builder, BuilderError},
    context::Context,
    execution_engine::{ExecutionEngine, FunctionLookupError},
    llvm_sys::{
        LLVMPassManager, LLVMType, LLVMValue,
        core::LLVMTypeOf,
        prelude::{LLVMPassManagerRef, LLVMTypeRef, LLVMValueRef},
    },
    module::{Linkage, Module},
    object_file::ObjectFile,
    passes::{PassBuilderOptions, PassManager},
    support::LLVMString,
    targets::{CodeModel, InitializationConfig, RelocMode, Target},
    types::{
        AnyType, AnyTypeEnum, ArrayType, AsTypeRef, BasicMetadataTypeEnum, BasicType,
        BasicTypeEnum, FloatType, IntType, PointerType, StructType,
    },
    values::{
        AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValue, BasicValueEnum,
        FloatValue, FunctionValue, GlobalValue, IntValue, PointerValue, StructValue,
    },
};
use std::{
    assert_matches::debug_assert_matches, collections::HashMap, marker::PhantomData,
    mem::MaybeUninit, path::Path,
};

pub mod jit;

const REPL_EXPR_ANON_FN_NAME: &str = "__repl_expr";

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

pub struct Codegen<'ctx, 'alloc> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Module<'ctx>,

    symbols: SymbolTable<Symbol<'ctx>>,
    type_table: HashMap<VarDeclList, StructType<'ctx>>,
    defer_stack: DeferStack,

    /// the current fn being compiled
    cur_fn: Option<FunctionValue<'ctx>>,

    alloc: &'alloc bumpalo::Bump,
}

impl<'ctx, 'alloc> Codegen<'ctx, 'alloc> {
    pub fn new(
        context: &'ctx Context,
        builder: Builder<'ctx>,
        module: Module<'ctx>,
        alloc: &'alloc bumpalo::Bump,
    ) -> Codegen<'ctx, 'alloc> {
        Codegen {
            context,
            builder,
            module,
            symbols: SymbolTable::with_one_scope(),
            type_table: HashMap::new(),
            defer_stack: DeferStack::default(),
            cur_fn: None,
            alloc,
        }
    }

    pub fn new_module(
        context: &'ctx Context,
        module_name: &str,
        alloc: &'alloc bumpalo::Bump,
    ) -> Self {
        let builder = context.create_builder();
        let module = context.create_module(module_name);
        Codegen::new(context, builder, module, alloc)
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

    pub fn move_module_to(&mut self, jit: &mut Jit<'ctx>) {
        let module = self.take_module();
        jit.add_module(module);
    }

    pub fn compile_top_level(&mut self, stmt: Ptr<Expr>) {
        // if let Err(err) = self.compile_expr_to_float(stmt) {
        //     self.errors.push(err);
        // }
        self.compile_expr(stmt, Type::Void).unwrap();
    }

    fn compile_expr(&mut self, expr: Ptr<Expr>, expr_ty: Type) -> CodegenResult<Symbol<'ctx>> {
        match &expr.kind {
            ExprKind::Ident(name) => Ok(self.get_symbol(&**name)),
            ExprKind::Literal { kind, code } => match expr_ty {
                Type::Never => Ok(Symbol::Never),
                Type::Int { bits, .. } => {
                    reg(self.int_type(bits).const_int(code.parse().unwrap_debug(), false))
                },
                Type::Float { bits } => reg(self.float_type(bits).const_float_from_string(code)),
                t => panic!("{t:?}"),
                _ => unreachable_debug(),
            },
            ExprKind::BoolLit(bool) => {
                debug_assert_matches!(expr_ty, Type::Bool | Type::Unset);
                let b_ty = self.context.bool_type();
                reg(if *bool { b_ty.const_all_ones() } else { b_ty.const_zero() })
            },
            ExprKind::ArrayTy { count, ty } => todo!(),
            ExprKind::ArrayTy2 { ty } => todo!(),
            ExprKind::ArrayLit { elements } => {
                let Type::Array { len, elem_ty } = expr_ty else { unreachable_debug() };
                let elem_cty = self.llvm_type(*elem_ty);
                let arr_ty = elem_cty.basic_ty().array_type(len as u32);
                let arr_ptr = self.builder.build_array_alloca(
                    arr_ty,
                    self.context.i64_type().const_int(len as u64, false),
                    "arr",
                )?;
                let idx_ty = self.context.i64_type();
                for (idx, elem) in elements.iter().enumerate() {
                    let elem_val = try_compile_expr_as_val!(self, *elem, *elem_ty);
                    let elem_ptr = unsafe {
                        self.builder.build_in_bounds_gep(
                            arr_ty,
                            arr_ptr,
                            &[idx_ty.const_zero(), idx_ty.const_int(idx as u64, false)],
                            "",
                        )
                    }?;
                    self.builder.build_store(elem_ptr, elem_val)?;
                }
                stack_val(arr_ptr)
            },
            ExprKind::ArrayLitShort { val, count: _ } => {
                let Type::Array { len, elem_ty } = expr_ty else { panic!() };
                let elem_cty = self.llvm_type(*elem_ty);
                let arr_ty = elem_cty.basic_ty().array_type(len as u32);
                let arr_ptr = self.builder.build_array_alloca(
                    arr_ty,
                    self.context.i64_type().const_int(len as u64, false),
                    "arr",
                )?;
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
                    self.builder.build_store(elem_ptr, elem_val)?;
                }
                stack_val(arr_ptr)
            },
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn(_) => todo!("todo: runtime fn"),
            ExprKind::Parenthesis { expr } => self.compile_expr(*expr, expr_ty),
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
                        debug_assert_matches!(expr_ty, Type::Void | Type::Unset);
                        Symbol::Void
                    }
                };
                self.close_scope();
                res
            },
            &ExprKind::StructDef(fields) => {
                // TODO: Is it possible to set the struct name?
                let field_types =
                    fields.iter().map(|f| self.llvm_type(f.ty).basic_ty()).collect::<Vec<_>>();
                let ty = self.context.struct_type(&field_types, false);
                self.type_table.insert(fields, ty);
                Ok(Symbol::StructDef { fields, ty })
            },
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs, fields: values } => {
                let (fields, struct_ty, struct_ptr) = match (lhs, expr_ty) {
                    (_, Type::Never) => return Ok(Symbol::Never),
                    (lhs, t @ Type::Struct { fields }) => {
                        if let Some(lhs) = lhs {
                            self.compile_expr(*lhs, t)?;
                        }
                        let struct_ty = self.type_table[&fields];
                        let ptr = self.builder.build_alloca(struct_ty, "struct")?;
                        (fields, struct_ty, ptr)
                    },
                    (Some(lhs), Type::Ptr(pointee)) => {
                        let Type::Struct { fields } = *pointee else { unreachable_debug() };
                        let struct_ty = self.type_table[&fields];
                        let ptr = try_compile_expr_as_val!(self, *lhs, expr_ty).ptr_val();
                        (fields, struct_ty, ptr)
                    },
                    (Some(lhs), Type::Struct { fields }) => {
                        let struct_ty = self.type_table[&fields];
                        let ptr = self.builder.build_alloca(struct_ty, "struct")?;
                        (fields, struct_ty, ptr)
                    },
                    _ => unreachable_debug(),
                };

                let mut is_initialized_field = vec![false; fields.len()];
                for (f, init) in values.iter() {
                    let (f_idx, field_def) = fields.find_field(&*f.text).unwrap_debug();

                    is_initialized_field[f_idx] = true;

                    let init_val = match init {
                        Some(init) => try_compile_expr_as_val!(self, *init, field_def.ty),
                        None => try_get_symbol_as_val!(self, &*f.text, field_def.ty),
                    };

                    let field_ptr = self.builder.build_struct_gep(
                        struct_ty,
                        struct_ptr,
                        f_idx as u32,
                        &*f.text,
                    )?;
                    self.builder.build_store(field_ptr, init_val)?;
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
                    let init_sym = self.compile_expr(field.default.unwrap_debug(), field.ty)?;
                    let v = self.sym_as_val(init_sym, field.ty)?;
                    self.builder.build_store(field_ptr, v)?;
                }
                stack_val(struct_ptr)
            },
            &ExprKind::Dot { lhs, rhs } => {
                let Type::Struct { fields } = lhs.ty else { panic!() };
                let struct_ty = self.type_table[&fields];
                let Symbol::Stack(struct_ptr) = try_not_never!(self.compile_typed_expr(lhs)?)
                else {
                    panic!()
                };
                let (field_idx, field) = fields.find_field(&rhs.text).unwrap_debug();
                let field_ptr = self.builder.build_struct_gep(
                    struct_ty,
                    struct_ptr,
                    field_idx as u32,
                    &format!("{}_ptr", &*rhs.text),
                )?;
                /*
                if Self::is_register_passable(expr_ty) {
                    reg(self.builder.build_load(
                        self.llvm_type(field.ty).basic_ty(),
                        field_ptr,
                        &rhs.text,
                    )?)
                } else {
                    stack_val(field_ptr)
                }
                */
                stack_val(field_ptr)
            },
            &ExprKind::Index { lhs, idx } => {
                let Type::Array { len, elem_ty } = lhs.ty else { panic!() };
                let arr_ty = self.llvm_type(lhs.ty).arr_ty();
                let arr = self.compile_typed_expr(lhs)?;
                let Symbol::Stack(arr_ptr) = arr else { panic!() };
                let idx_val = try_compile_expr_as_val!(self, idx.expr, idx.ty);
                debug_assert!(expr_ty == *elem_ty);
                self.build_index(arr_ty, arr_ptr, idx_val.int_val(), *elem_ty)
            },
            ExprKind::Call { func, args, .. } => {
                let func = func.try_to_ident().unwrap_debug();
                let (&params, &func) = self.get_symbol(&func.text).as_fn().unwrap_debug();

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

                reg(self.builder.build_call(func, &args, "call")?)
            },
            ExprKind::UnaryOp { kind, expr, .. } => {
                //let v = try_compile_expr_as_val!(self, *expr, expr_ty);
                let v = try_not_never!(self.compile_expr(*expr, expr_ty)?);
                match kind {
                    UnaryOpKind::AddrOf => {
                        return reg(match v {
                            Symbol::Stack(ptr_value) => ptr_value,
                            Symbol::Register(val) => self.build_stack_alloc_as_ptr(val)?,
                            _ => todo!(),
                        });
                    },
                    UnaryOpKind::AddrMutOf => todo!(),
                    _ => {},
                }

                let v = self.sym_as_val(v, expr_ty)?;
                match kind {
                    UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf => unreachable_debug(),
                    /*
                    UnaryOpKind::Deref => reg(self.builder.build_load(
                        self.llvm_type(expr_ty).basic_ty(),
                        v.ptr_val(),
                        "deref",
                    )?),
                    */
                    UnaryOpKind::Deref => stack_val(v.ptr_val()),
                    UnaryOpKind::Not => match expr_ty {
                        Type::Bool => reg(self.builder.build_not(v.bool_val(), "not")?),
                        _ => todo!(),
                    },
                    UnaryOpKind::Neg => match expr_ty {
                        Type::Float { .. } => {
                            reg(self.builder.build_float_neg(v.float_val(), "neg")?)
                        },
                        _ => todo!(),
                    },
                    UnaryOpKind::Try => todo!(),
                }
            },
            ExprKind::BinOp { lhs, op, rhs, val_ty: ty } => {
                let ty = if op.has_independent_out_ty() { *ty } else { expr_ty };
                if ty == Type::Bool && matches!(op, BinOpKind::And | BinOpKind::Or) {
                    return self.build_bool_short_circuit_binop(*lhs, *rhs, *op);
                }
                let lhs_val = try_compile_expr_as_val!(self, *lhs, ty);
                let rhs_val = try_compile_expr_as_val!(self, *rhs, ty);
                match ty {
                    Type::Int { is_signed, .. } => self
                        .build_int_binop(lhs_val.int_val(), rhs_val.int_val(), is_signed, *op)
                        .map(reg_sym),
                    Type::Float { .. } => self
                        .build_float_binop(lhs_val.float_val(), rhs_val.float_val(), *op)
                        .map(reg_sym),
                    Type::Bool => unreachable_debug(),
                    t => todo!("{:?}", t),
                }
            },
            ExprKind::Assign { lhs, rhs } => {
                let var_name = &*lhs.try_to_ident().expect("todo: non-ident assign lhs").text;
                match self.get_symbol(var_name) {
                    Symbol::Stack(stack_var) => {
                        let rhs = try_compile_expr_as_val!(self, *rhs, lhs.ty);
                        self.builder.build_store(stack_var, rhs)?;
                    },
                    Symbol::Register(_) => return Err(CodegenError::CannotMutateRegister),
                    _ => panic_debug("unexpected symbol"),
                }
                Ok(Symbol::Void)
            },
            ExprKind::BinOpAssign { lhs, op, rhs } => {
                let ExprWithTy { expr: lhs_expr, ty: lhs_ty } = lhs;
                let var_name = &*lhs_expr.try_to_ident().unwrap_debug().text;
                match self.get_symbol(var_name) {
                    Symbol::Stack(stack_var) => {
                        let lhs_llvm_ty = self.llvm_type(*lhs_ty).basic_ty();
                        let lhs = self.builder.build_load(lhs_llvm_ty, stack_var, "lhs")?;
                        let rhs = try_compile_expr_as_val!(self, *rhs, *lhs_ty);
                        let binop_res = match lhs_ty {
                            Type::Int { is_signed, .. } => self.build_int_binop(
                                lhs.into_int_value(),
                                rhs.int_val(),
                                *is_signed,
                                *op,
                            )?,
                            Type::Float { .. } => self.build_float_binop(
                                lhs.into_float_value(),
                                rhs.float_val(),
                                *op,
                            )?,
                            t => todo!("{:?}", t),
                        };
                        self.builder.build_store(stack_var, binop_res)?;
                    },
                    Symbol::Register(a) => panic!("{a:?}"),
                    Symbol::Register(_) => return Err(CodegenError::CannotMutateRegister),
                    Symbol::Function { .. } => panic!("cannot assign to function"),
                    Symbol::StructDef { .. } => panic!("cannot assign to struct definition"),
                    Symbol::Void => todo!(),
                    Symbol::Never => todo!(),
                }
                Ok(Symbol::Void)
            },
            ExprKind::VarDecl(VarDecl { markers, ident, ty, default: init, is_const }) => {
                let var_name = &*ident.text;
                let is_mut = markers.is_mut;
                let v = match init.map(|p| &p.as_ref().kind) {
                    Some(ExprKind::Fn(f)) => self.compile_fn(var_name, f)?,
                    Some(&ExprKind::StructDef(fields)) => {
                        let ty = self.context.opaque_struct_type(var_name);
                        let field_types = fields
                            .iter()
                            .map(|f| self.llvm_type(f.ty).basic_ty())
                            .collect::<Vec<_>>();
                        if !ty.set_body(&field_types, false) {
                            panic!()
                        }
                        //let ty = self.context.struct_type(&field_types, false);
                        self.type_table.insert(fields, ty);
                        Symbol::StructDef { fields, ty }
                    },
                    _ => {
                        if !is_mut && let Some(init) = init {
                            self.compile_expr(*init, *ty)?
                            //Symbol::Register(try_compile_expr_as_val!(self,
                            // *init))
                        } else {
                            let stack_ty = self.llvm_type(*ty).basic_ty();
                            let stack_var = self.builder.build_alloca(stack_ty, &var_name)?;
                            if let Some(init) = init {
                                let init = try_compile_expr_as_val!(self, *init, *ty);
                                self.builder.build_store(stack_var, init)?;
                            }
                            Symbol::Stack(stack_var)
                        }
                    },
                };

                let old = self.symbols.insert(var_name.to_string(), v);
                if old.is_some() {
                    //println!("LOG: '{}' was shadowed in the same scope",
                    // var_name);
                }
                debug_assert_matches!(expr_ty, Type::Void | Type::Unset);
                Ok(Symbol::Void)
            },
            ExprKind::If { condition, then_body, else_body, .. } => {
                let func = self.cur_fn.unwrap_debug();
                let condition = try_compile_expr_as_val!(self, *condition, Type::Bool).bool_val();

                let mut then_bb = self.context.append_basic_block(func, "then");
                let mut else_bb = self.context.append_basic_block(func, "else");
                let merge_bb = self.context.append_basic_block(func, "ifmerge");

                self.builder.build_conditional_branch(condition, then_bb, else_bb)?;

                self.builder.position_at_end(then_bb);
                let then_sym = self.compile_expr(*then_body, expr_ty)?;
                if then_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_sym = if let Some(else_body) = else_body {
                    self.compile_expr(*else_body, expr_ty)?
                } else {
                    Symbol::Void
                };
                if else_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                else_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);

                match expr_ty {
                    Type::Void => return Ok(Symbol::Void),
                    Type::Never => {
                        self.builder.build_unreachable()?;
                        return Ok(Symbol::Never);
                    },
                    _ => {},
                }

                let branch_ty = self.llvm_type(expr_ty).basic_ty();
                let phi = self.builder.build_phi(branch_ty, "ifexpr")?;
                match (
                    self.sym_as_val_checked(then_sym, branch_ty)?,
                    self.sym_as_val_checked(else_sym, branch_ty)?,
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
            ExprKind::Match { val, else_body, .. } => todo!(),
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
                self.builder.build_unconditional_branch(cond_bb);

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

                let val = self.build_index(arr_ty, arr_ptr, idx_int, *elem_ty)?;
                self.symbols.insert(&*iter_var.text, val);

                try_not_never!(self.compile_expr(*body, Type::Void)?);
                self.builder.build_unconditional_branch(inc_bb);

                // inc
                self.builder.position_at_end(inc_bb);
                let next_idx =
                    self.builder.build_int_add(idx_int, idx_ty.const_int(1, false), "next_idx")?;
                idx.add_incoming(&[(&next_idx, inc_bb)]);
                self.builder.build_unconditional_branch(cond_bb);

                // end
                self.builder.position_at_end(end_bb);
                debug_assert_matches!(expr_ty, Type::Void | Type::Unset);
                Ok(Symbol::Void)
            },
            ExprKind::While { condition, body, .. } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Defer(inner) => {
                self.defer_stack.push_expr(*inner);
                debug_assert_matches!(expr_ty, Type::Void | Type::Unset);
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
            ExprKind::Semicolon(_) => todo!(),
        }
    }

    fn compile_typed_expr(&mut self, expr: ExprWithTy) -> CodegenResult<Symbol<'ctx>> {
        self.compile_expr(expr.expr, expr.ty)
    }

    fn compile_fn(&mut self, name: &str, f: &Fn) -> CodegenResult<Symbol<'ctx>> {
        let fn_val = self.compile_prototype(name, f);
        let val = self.compile_fn_body(fn_val, f)?;
        Ok(Symbol::Function { params: f.params, val })
    }

    fn compile_prototype(&mut self, name: &str, f: &Fn) -> FunctionValue<'ctx> {
        let param_types = f
            .params
            .iter()
            .map(|VarDecl { ty, .. }| {
                if Self::is_register_passable(*ty) {
                    self.llvm_type(*ty).basic_metadata_ty()
                } else {
                    BasicMetadataTypeEnum::from(self.context.ptr_type(AddressSpace::default()))
                }
            })
            .collect::<Vec<_>>();
        let fn_type = match f.ret_type {
            Type::Void | Type::Never => self.context.void_type().fn_type(&param_types, false),
            _ => {
                let ret_type = self.llvm_type(f.ret_type).basic_ty();
                ret_type.fn_type(&param_types, false)
            },
        };
        let fn_val = self.module.add_function(name, fn_type, Some(Linkage::External));

        for (idx, param) in fn_val.get_param_iter().enumerate() {
            param.set_name(&f.params[idx].ident.text)
        }

        // self.symbols
        //     .insert(name.to_string(), LLVMSymbol::Function { params, val: fn_val });

        fn_val
    }

    pub fn compile_fn_body(
        &mut self,
        func: FunctionValue<'ctx>,
        f: &Fn,
    ) -> CodegenResult<FunctionValue<'ctx>> {
        let entry = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        self.cur_fn = Some(func);

        self.open_scope();
        let res = try {
            self.symbols.reserve(f.params.len());

            for (param, param_def) in func.get_param_iter().zip(f.params.iter()) {
                let pname = &param_def.ident.text;
                let param = CodegenValue::new(param.as_value_ref());
                let s = if Self::is_register_passable(param_def.ty) {
                    if param_def.markers.is_mut {
                        self.position_builder_at_start(func.get_first_basic_block().unwrap_debug());
                        self.build_stack_alloc(param)?
                    } else {
                        Symbol::Register(param)
                    }
                } else {
                    Symbol::Stack(param.ptr_val())
                };
                self.symbols.insert(pname.to_string(), s);
            }

            let body = self.compile_expr(f.body, f.ret_type)?;
            self.build_return(body, f.ret_type)?;

            if func.verify(true) {
                func
            } else {
                #[cfg(debug_assertions)]
                self.module.print_to_stderr();
                unsafe { func.delete() };
                Err(CodegenError::InvalidGeneratedFunction)?
            }
        };
        self.close_scope();
        res
    }

    pub fn compile_struct_def(&self, fields: Ptr<[VarDecl]>) -> CodegenResult<StructType<'ctx>> {
        let field_types =
            fields.iter().map(|f| self.llvm_type(f.ty).basic_ty()).collect::<Vec<_>>();
        let s_ty = self.context.struct_type(&field_types, false);
        println!("{:?}", s_ty);
        Ok(s_ty)
    }

    // -----------------------

    fn build_return(&self, ret_sym: Symbol<'ctx>, ret_ty: Type) -> CodegenResult<Symbol> {
        match ret_sym {
            Symbol::Never => {},
            Symbol::Void => {
                self.builder.build_return(None)?;
            },
            _ => {
                let ret = self.sym_as_val(ret_sym, ret_ty)?;
                self.builder.build_return(Some(&ret))?;
            },
        }
        Ok(Symbol::Never)
    }

    fn build_index(
        &self,
        arr_ty: ArrayType<'ctx>,
        arr_ptr: PointerValue<'ctx>,
        idx_val: IntValue<'ctx>,
        elem_ty: Type,
    ) -> CodegenResult<Symbol<'ctx>> {
        let elem_ptr = unsafe {
            self.builder.build_in_bounds_gep(
                arr_ty,
                arr_ptr,
                &[self.context.i64_type().const_zero(), idx_val],
                "idx_ptr",
            )?
        };
        if Self::is_register_passable(elem_ty) {
            reg(self
                .builder
                .build_load(self.llvm_type(elem_ty).basic_ty(), elem_ptr, "idx_val")?)
        } else {
            stack_val(elem_ptr)
        }
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

            BinOpKind::Range => todo!(),
            BinOpKind::RangeInclusive => todo!(),
            _ => unreachable!(),
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
        lhs: Ptr<Expr>,
        rhs: Ptr<Expr>,
        op: BinOpKind,
    ) -> CodegenResult<Symbol<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
            Ok(Symbol::Register(CodegenValue::new(val.as_value_ref())))
        }

        let lhs = try_compile_expr_as_val!(self, lhs, Type::Bool).bool_val();

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

    fn llvm_type(&self, ty: Type) -> CodegenType<'ctx> {
        CodegenType::new(match ty {
            Type::Void => self.context.void_type().as_type_ref(),
            Type::Never => todo!(),
            Type::Ptr(_) => self.context.ptr_type(AddressSpace::default()).as_type_ref(),
            Type::Int { bits, .. } => self.int_type(bits).as_type_ref(),
            Type::IntLiteral => panic_debug("int literal type should have been resolved already"),
            Type::Bool => self.context.bool_type().as_type_ref(),
            Type::Float { bits } => self.float_type(bits).as_type_ref(),
            Type::FloatLiteral => {
                panic_debug("float literal type should have been resolved already")
            },
            Type::Function(_) => todo!(),
            Type::Array { len: count, elem_ty: ty } => {
                self.llvm_type(*ty).basic_ty().array_type(count as u32).as_type_ref()
            },
            Type::Struct { fields: ptr } => self.type_table[&ptr].as_type_ref(),
            Type::Union { .. } => todo!(),
            Type::Enum { .. } => todo!(),
            Type::Type(_) => todo!(),
            Type::Unset | Type::Unevaluated(_) => panic_debug("unvalid type"),
            ty => todo!("{:?}", ty),
        })
    }

    fn sym_as_val(&self, sym: Symbol<'ctx>, ty: Type) -> CodegenResult<CodegenValue<'ctx>> {
        Ok(match sym {
            Symbol::Stack(ptr) => {
                let ty = self.llvm_type(ty).basic_ty();
                CodegenValue::new(self.builder.build_load(ty, ptr, "")?.as_value_ref())
            },
            Symbol::Register(val) => val,
            _ => panic_debug("unexpected symbol"),
        })
    }

    fn sym_as_val_checked(
        &self,
        sym: Symbol<'ctx>,
        ty: BasicTypeEnum,
    ) -> CodegenResult<Option<CodegenValue<'ctx>>> {
        Ok(Some(match sym {
            Symbol::Stack(ptr) => {
                CodegenValue::new(self.builder.build_load(ty, ptr, "")?.as_value_ref())
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
            Type::Ptr(_)
            | Type::Int { .. }
            | Type::Bool
            | Type::Float { .. }
            | Type::Array { .. } => true,
            Type::Void
            | Type::Never
            | Type::Function(_)
            | Type::Struct { .. }
            | Type::Union { .. }
            | Type::Enum { .. }
            | Type::Type(_) => false,
            Type::IntLiteral | Type::FloatLiteral | Type::Unset | Type::Unevaluated(_) => {
                panic_debug("invalid type")
            },
        }
    }

    pub fn init_target_machine() -> TargetMachine {
        Target::initialize_all(&InitializationConfig::default());

        let target_triple: Option<&str> = None;
        let target_triple = if let Some(s) = target_triple {
            todo!()
        } else {
            TargetMachine::get_default_triple()
        };
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

    fn load_stack_val(
        &self,
        ptr: PointerValue<'ctx>,
        ty: Type,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        let ty = self.llvm_type(ty).basic_ty();
        Ok(CodegenValue::new(self.builder.build_load(ty, ptr, "")?.as_value_ref()))
    }

    fn build_stack_alloc(&self, v: impl BasicValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
        self.build_stack_alloc_as_ptr(v).and_then(stack_val)
    }

    fn build_stack_alloc_as_ptr(
        &self,
        v: impl BasicValue<'ctx>,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let v = v.as_basic_value_enum();
        let alloca = self.builder.build_alloca(v.get_type(), "").unwrap_debug();
        self.builder.build_store(alloca, v)?;
        Ok(alloca)
    }

    #[inline]
    fn alloc<T: core::fmt::Debug>(&self, val: T) -> CodegenResult<Ptr<T>> {
        match self.alloc.try_alloc(val) {
            Result::Ok(ok) => Ok(Ptr::from(ok)),
            Result::Err(e) => Err(CodegenError::AllocErr(e)),
        }
    }
}

// optimizations
impl<'ctx, 'alloc> Codegen<'ctx, 'alloc> {
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
        filename: &str,
    ) -> Result<(), CodegenError> {
        let p = Path::new(filename);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        target_machine
            .write_to_file(&self.module, inkwell::targets::FileType::Object, p)
            .map_err(|err| CodegenError::CannotCompileObjFile(err))
    }

    pub fn jit_run_fn<Ret>(&mut self, fn_name: &str, opt: OptimizationLevel) -> CodegenResult<Ret> {
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
}

impl<'ctx> Symbol<'ctx> {
    fn as_fn(&self) -> Option<(&VarDeclList, &FunctionValue<'ctx>)> {
        match self {
            Symbol::Function { params, val } => Some((params, val)),
            _ => None,
        }
    }
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
