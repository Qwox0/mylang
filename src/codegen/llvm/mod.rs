#![allow(unused)]

use self::jit::Jit;
use crate::{
    ast::{
        BinOpKind, DeclMarkers, Expr, ExprKind, Fn, Ident, LitKind, PreOpKind, VarDecl, VarDeclList,
    },
    cli::DebugOptions,
    defer_stack::DeferStack,
    parser::{ParseError, StmtIter, lexer::Code},
    ptr::Ptr,
    symbol_table::SymbolTable,
    type_::Type,
    util::{UnwrapDebug, forget_lifetime, unreachable_debug},
};
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
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{
        AnyType, AnyTypeEnum, AsTypeRef, BasicMetadataTypeEnum, BasicType, BasicTypeEnum,
        FloatType, IntType, PointerType, StructType,
    },
    values::{
        AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValue, BasicValueEnum,
        FloatValue, FunctionValue, GlobalValue, IntValue, PointerValue, StructValue,
    },
};
use std::{collections::HashMap, marker::PhantomData, mem::MaybeUninit};

pub mod jit;

const REPL_EXPR_ANON_FN_NAME: &str = "__repl_expr";

#[derive(Debug)]
pub enum CodegenError {
    BuilderError(BuilderError),
    InvalidGeneratedFunction,
    InvalidCallProduced,
    FunctionLookupError(FunctionLookupError),
    CouldntCreateJit(LLVMString),
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

/// Returns [`Symbol::Never`] if it occurs
macro_rules! try_compile_expr_as_val {
    ($codegen:expr, $expr:expr) => {{
        let codegen: &mut Codegen = $codegen;
        let expr: Ptr<Expr> = $expr;
        match codegen.compile_expr(expr)? {
            sym @ Symbol::Never => return Ok(sym),
            sym => codegen.sym_as_val(sym, expr.ty)?,
        }
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

    pub fn into_module(self) -> CodegenModule<'ctx> {
        CodegenModule(self.module)
    }

    pub fn compile_top_level(&mut self, stmt: Ptr<Expr>) {
        // if let Err(err) = self.compile_expr_to_float(stmt) {
        //     self.errors.push(err);
        // }
        self.compile_expr(stmt).unwrap();
    }

    fn compile_expr(&mut self, expr: Ptr<Expr>) -> CodegenResult<Symbol<'ctx>> {
        match &expr.kind {
            ExprKind::Ident(name) => Ok(self.get_symbol(&**name)),
            ExprKind::Literal { kind, code } => {
                match kind {
                    LitKind::Char => todo!(),
                    LitKind::BChar => todo!(),
                    // TODO: int
                    LitKind::Int => reg(self.float_type(expr.ty).const_float_from_string(code)),
                    LitKind::Float => reg(self.float_type(expr.ty).const_float_from_string(code)),
                    LitKind::Str => todo!(),
                }
            },
            ExprKind::BoolLit(bool) => {
                debug_assert!(expr.ty == Type::Bool);
                let b_ty = self.context.bool_type();
                reg(if *bool { b_ty.const_all_ones() } else { b_ty.const_zero() })
            },
            ExprKind::ArrayTy { count, ty } => todo!(),
            ExprKind::ArrayTy2 { ty } => todo!(),
            ExprKind::ArrayLit { elements } => todo!(),
            ExprKind::ArrayLitShort { val, count } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn(_) => todo!("todo: runtime fn"),
            ExprKind::Parenthesis { expr } => self.compile_expr(*expr),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                self.open_scope();
                let res = try {
                    let mut out = Symbol::Void;
                    for s in stmts.iter() {
                        out = self.compile_expr(*s)?;
                        if out == Symbol::Never {
                            break;
                        }
                    }
                    if !*has_trailing_semicolon || out == Symbol::Never {
                        out
                    } else {
                        debug_assert!(expr.ty == Type::Void);
                        Symbol::Void
                    }
                };
                self.close_scope();
                res
            },
            &ExprKind::StructDef(fields) => {
                // TODO: Is it possible to set the struct name
                /*
                // let ty = self.context.opaque_struct_type(var_name);
                let field_types =
                    fields.iter().map(|f| self.llvm_type(f.ty).basic_ty()).collect::<Vec<_>>();
                // if !ty.set_body(&field_types, false) {
                //     panic!()
                // }
                let ty = self.context.struct_type(&field_types, false);
                self.type_table.insert(fields, ty);
                Symbol::StructDef { fields, ty }
                */
                todo!()
            },
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs, fields: values } => {
                let lhs = lhs.expect("todo: infer struct type");
                let t = match lhs.ty {
                    Type::Never => return Ok(Symbol::Never),
                    Type::Type(t) => t,
                    _ => todo!("todo: initializer: more complex lhs"),
                };
                let Type::Struct { fields } = *t else { panic!() };
                let struct_ty = self.type_table[&fields];
                let struct_ptr = self.builder.build_alloca(struct_ty, "struct")?;

                let mut is_initialized_field = vec![false; fields.len()];
                for (f, init) in values.iter() {
                    let (f_idx, field_def) = fields.find_field(&*f.text).unwrap_debug();

                    is_initialized_field[f_idx] = true;

                    let init_val = match init {
                        Some(init) => {
                            debug_assert!(init.ty == field_def.ty || init.ty == Type::Never);
                            try_compile_expr_as_val!(self, *init)
                        },
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
                    let init_sym = self.compile_expr(field.default.unwrap_debug())?;
                    let v = self.sym_as_val(init_sym, field.ty)?;
                    self.builder.build_store(field_ptr, v)?;
                }
                stack_val(struct_ptr)
            },
            ExprKind::Dot { lhs, rhs } => {
                let Type::Struct { fields } = lhs.ty else { panic!() };
                let lhs_ty = self.type_table[&fields];
                let Symbol::Stack(struct_ptr) = self.compile_expr(*lhs)? else { panic!() };
                let (field_idx, field) = fields.find_field(&rhs.text).unwrap_debug();
                let field_ptr = self.builder.build_struct_gep(
                    lhs_ty,
                    struct_ptr,
                    field_idx as u32,
                    &format!("{}_ptr", &*rhs.text),
                )?;
                if matches!(expr.ty, Type::Struct { .. }) {
                    stack_val(field_ptr)
                } else {
                    reg(self.builder.build_load(
                        self.llvm_type(field.ty).basic_ty(),
                        field_ptr,
                        &rhs.text,
                    )?)
                }
            },
            ExprKind::PostOp { kind, expr } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            ExprKind::Call { func, args } => {
                let func = func.try_to_ident().unwrap_debug();
                let (&params, &func) = self.get_symbol(&func.text).as_fn().unwrap_debug();

                let args = params
                    .as_ref()
                    .iter()
                    .enumerate()
                    .map(|(idx, param)| {
                        let arg = args.get(idx).copied().or(param.default).unwrap_debug();
                        Ok(match self.compile_expr(arg)? {
                            Symbol::Stack(ptr) => BasicMetadataValueEnum::from(ptr),
                            Symbol::Register(val) => val.basic_metadata_val(),
                            _ => todo!(),
                        })
                    })
                    .collect::<CodegenResult<Vec<_>>>()?;

                reg(self.builder.build_call(func, &args, "call")?)
            },
            ExprKind::PreOp { kind, expr } => {
                let v = try_compile_expr_as_val!(self, *expr);
                reg(match kind {
                    PreOpKind::AddrOf => todo!(),
                    PreOpKind::AddrMutOf => todo!(),
                    PreOpKind::Deref => todo!(),
                    PreOpKind::Not => todo!(),
                    PreOpKind::Neg => match expr.ty {
                        Type::Float { .. } => self.builder.build_float_neg(v.float_val(), "neg")?,
                        _ => todo!(),
                    },
                }) // TODO: maybe different return types possible in the future
            },
            ExprKind::BinOp { lhs, op, rhs } => {
                let lhs_val = try_compile_expr_as_val!(self, *lhs);
                let rhs_val = try_compile_expr_as_val!(self, *rhs);
                match lhs.ty {
                    Type::Float { .. } => self
                        .build_float_binop(lhs_val.float_val(), rhs_val.float_val(), *op)
                        .map(reg_sym),
                    Type::Bool => {
                        self.build_bool_binop(lhs_val.bool_val(), rhs_val.bool_val(), *op)
                    },
                    _ => todo!(),
                }
            },
            ExprKind::Assign { lhs, rhs } => {
                debug_assert!(lhs.ty == rhs.ty);
                let var_name = &*lhs.try_to_ident().expect("todo: non-ident assign lhs").text;
                match self.get_symbol(var_name) {
                    Symbol::Stack(stack_var) => {
                        let rhs = try_compile_expr_as_val!(self, *rhs);
                        self.builder.build_store(stack_var, rhs)?;
                    },
                    Symbol::Register(_) => return Err(CodegenError::CannotMutateRegister),
                    _ => unreachable_debug(),
                }
                Ok(Symbol::Void)
            },
            ExprKind::BinOpAssign { lhs: lhs_expr, op, rhs: rhs_expr } => {
                debug_assert!(lhs_expr.ty == rhs_expr.ty);
                let var_name = &*lhs_expr.try_to_ident().unwrap_debug().text;
                match self.get_symbol(var_name) {
                    Symbol::Stack(stack_var) => {
                        let lhs_ty = self.llvm_type(lhs_expr.ty).basic_ty();
                        let lhs = self.builder.build_load(lhs_ty, stack_var, "lhs")?;
                        let rhs = try_compile_expr_as_val!(self, *rhs_expr);
                        let binop_res = match lhs_expr.ty {
                            Type::Float { .. } => self.build_float_binop(
                                lhs.into_float_value(),
                                rhs.float_val(),
                                *op,
                            )?,
                            _ => todo!(),
                        };
                        self.builder.build_store(stack_var, binop_res)?;
                    },
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
                            Symbol::Register(try_compile_expr_as_val!(self, *init))
                        } else {
                            let ty = self.llvm_type(*ty).basic_ty();
                            let stack_var = self.builder.build_alloca(ty, &var_name)?;
                            if let Some(init) = init {
                                let init = try_compile_expr_as_val!(self, *init);
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
                debug_assert!(expr.ty == Type::Void);
                Ok(Symbol::Void)
            },
            ExprKind::If { condition, then_body, else_body } => {
                let func = self.cur_fn.unwrap_debug();
                let condition = try_compile_expr_as_val!(self, *condition).bool_val();

                let mut then_bb = self.context.append_basic_block(func, "then");
                let mut else_bb = self.context.append_basic_block(func, "else");
                let merge_bb = self.context.append_basic_block(func, "ifmerge");

                self.builder.build_conditional_branch(condition, then_bb, else_bb)?;

                self.builder.position_at_end(then_bb);
                let then_sym = self.compile_expr(*then_body)?;
                if then_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_sym = if let Some(else_body) = else_body {
                    self.compile_expr(*else_body)?
                } else {
                    Symbol::Void
                };
                if else_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                else_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);

                match expr.ty {
                    Type::Void => return Ok(Symbol::Void),
                    Type::Never => {
                        self.builder.build_unreachable()?;
                        return Ok(Symbol::Never);
                    },
                    _ => {},
                }

                let branch_ty = self.llvm_type(expr.ty).basic_ty();
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
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => todo!(),
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Defer(inner) => {
                self.defer_stack.push_expr(*inner);
                debug_assert!(expr.ty == Type::Void);
                Ok(Symbol::Void)
            },
            ExprKind::Return { expr: val } => {
                if let Some(val) = val {
                    let sym = self.compile_expr(*val)?;
                    self.build_return(sym, val.ty)?;
                } else {
                    self.builder.build_return(None)?;
                }
                debug_assert!(expr.ty == Type::Never);
                Ok(Symbol::Never)
            },
            ExprKind::Semicolon(_) => todo!(),
        }
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

            let body = self.compile_expr(f.body)?;
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

    pub fn build_float_binop(
        &mut self,
        lhs: FloatValue<'ctx>,
        rhs: FloatValue<'ctx>,
        op: BinOpKind,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
            Ok(CodegenValue::new(val.as_value_ref()))
        }

        match op {
            BinOpKind::Mul => ret(self.builder.build_float_mul(lhs, rhs, "mul")?),
            BinOpKind::Div => ret(self.builder.build_float_div(lhs, rhs, "div")?),
            BinOpKind::Mod => ret(self.builder.build_float_rem(lhs, rhs, "mod")?),
            BinOpKind::Add => ret(self.builder.build_float_add(lhs, rhs, "add")?),
            BinOpKind::Sub => ret(self.builder.build_float_sub(lhs, rhs, "sub")?),
            BinOpKind::Eq => {
                ret(self.builder.build_float_compare(FloatPredicate::OEQ, lhs, rhs, "eq")?)
            },
            BinOpKind::Ne => {
                ret(self.builder.build_float_compare(FloatPredicate::ONE, lhs, rhs, "ne")?)
            },
            BinOpKind::Lt => {
                ret(self.builder.build_float_compare(FloatPredicate::OLT, lhs, rhs, "lt")?)
            },
            BinOpKind::Le => {
                ret(self.builder.build_float_compare(FloatPredicate::OLE, lhs, rhs, "le")?)
            },
            BinOpKind::Gt => {
                ret(self.builder.build_float_compare(FloatPredicate::OGT, lhs, rhs, "gt")?)
            },
            BinOpKind::Ge => {
                ret(self.builder.build_float_compare(FloatPredicate::OGE, lhs, rhs, "ge")?)
            },

            BinOpKind::Range => todo!(),
            BinOpKind::RangeInclusive => todo!(),
            _ => unreachable!(),
        }
    }

    // TODO: short-circuiting logical operators
    // for a in [true, false] {
    //     for b in [true, false] {
    //         println!("{:>5} && {:>5}                 = {:>5}", a, b, a && b);
    //         println!("if {:>5} then {:>5} else false = {:>5}", a, b, if a { b }
    // else { false });         assert_eq!(a && b, if a { b } else { false });
    //     }
    // }
    // for a in [true, false] {
    //     for b in [true, false] {
    //         println!("{:>5} || {:>5}                 = {:>5}", a, b, a || b);
    //         println!("if !{:>5} then {:>5} else true = {:>5}", a, b, if a { true
    // } else { b });         assert_eq!(a || b, if a { true } else { b });
    //     }
    // }

    fn build_bool_binop(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
        op: BinOpKind,
    ) -> CodegenResult<Symbol<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
            Ok(Symbol::Register(CodegenValue::new(val.as_value_ref())))
        }

        ret(match op {
            BinOpKind::Eq => self.builder.build_int_compare(IntPredicate::EQ, lhs, rhs, "eq")?,
            BinOpKind::Ne => self.builder.build_int_compare(IntPredicate::NE, lhs, rhs, "ne")?,

            // false = 0, true = 1
            BinOpKind::Lt => self.builder.build_int_compare(IntPredicate::ULT, lhs, rhs, "lt")?,
            BinOpKind::Le => self.builder.build_int_compare(IntPredicate::ULE, lhs, rhs, "le")?,
            BinOpKind::Gt => self.builder.build_int_compare(IntPredicate::UGT, lhs, rhs, "gt")?,
            BinOpKind::Ge => self.builder.build_int_compare(IntPredicate::UGE, lhs, rhs, "ge")?,

            BinOpKind::And | BinOpKind::BitAnd => self.builder.build_and(lhs, rhs, "and")?,
            BinOpKind::Or | BinOpKind::BitOr => self.builder.build_or(lhs, rhs, "or")?,
            BinOpKind::BitXor => self.builder.build_xor(lhs, rhs, "xor")?,
            BinOpKind::Range => todo!(),
            BinOpKind::RangeInclusive => todo!(),
            _ => unreachable!(),
        })
    }

    /// LLVM does not make a distinction between signed and unsigned integer
    /// type
    fn int_type(&self, ty: Type) -> IntType<'ctx> {
        match ty {
            Type::Int { bits: 8, .. } => self.context.i8_type(),
            Type::Int { bits: 16, .. } => self.context.i16_type(),
            Type::Int { bits: 32, .. } => self.context.i32_type(),
            Type::Int { bits: 64, .. } => self.context.i64_type(),
            Type::Int { bits: 128, .. } => self.context.i128_type(),
            Type::Int { .. } => todo!("other int bits"),
            _ => panic!(),
        }
    }

    fn float_type(&self, ty: Type) -> FloatType<'ctx> {
        match ty {
            Type::Float { bits: 16 } => self.context.f16_type(),
            Type::Float { bits: 32 } => self.context.f32_type(),
            Type::Float { bits: 64 } => self.context.f64_type(),
            Type::Float { bits: 128 } => self.context.f128_type(),
            Type::Float { .. } => todo!("other float bits"),
            _ => panic!(),
        }
    }

    fn llvm_type(&self, ty: Type) -> CodegenType<'ctx> {
        CodegenType::new(match ty {
            Type::Void => self.context.void_type().as_type_ref(),
            Type::Never => todo!(),
            Type::Ptr(_) => self.context.ptr_type(AddressSpace::default()).as_type_ref(),
            Type::Int { bits, is_signed } => self.int_type(ty).as_type_ref(),
            // TODO: infer type or set correct type during sema
            Type::IntLiteral => self.context.i64_type().as_type_ref(),
            Type::Bool => self.context.bool_type().as_type_ref(),
            Type::Float { bits } => self.float_type(ty).as_type_ref(),
            // TODO: infer type or set correct type during sema
            Type::FloatLiteral => self.context.f64_type().as_type_ref(),
            Type::Function(_) => todo!(),
            //Type::Struct { .. } =>
            // self.context.get_struct_type("Sub").unwrap().as_type_ref(),
            Type::Struct { fields: ptr } => self.type_table[&ptr].as_type_ref(),
            Type::Union { .. } => todo!(),
            Type::Enum { .. } => todo!(),
            Type::Type(_) => todo!(),
            Type::Unset | Type::Unevaluated(_) => panic!("unvalid type"),
        })
    }

    fn sym_as_val(&self, sym: Symbol<'ctx>, ty: Type) -> CodegenResult<CodegenValue<'ctx>> {
        Ok(match sym {
            Symbol::Stack(ptr) => {
                let ty = self.llvm_type(ty).basic_ty();
                CodegenValue::new(self.builder.build_load(ty, ptr, "")?.as_value_ref())
            },
            Symbol::Register(val) => val,
            _ => unreachable_debug(),
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
            | Type::IntLiteral
            | Type::Bool
            | Type::Float { .. }
            | Type::FloatLiteral => true,
            Type::Void
            | Type::Never
            | Type::Function(_)
            | Type::Struct { .. }
            | Type::Union { .. }
            | Type::Enum { .. }
            | Type::Type(_)
            | Type::Unset
            | Type::Unevaluated(_) => false,
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
            self.compile_expr(*expr)?;
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
        let v = v.as_basic_value_enum();
        let alloca = self.builder.build_alloca(v.get_type(), "").unwrap_debug();
        self.builder.build_store(alloca, v)?;
        stack_val(alloca)
    }

    #[inline]
    fn alloc<T: core::fmt::Debug>(&self, val: T) -> CodegenResult<Ptr<T>> {
        match self.alloc.try_alloc(val) {
            Result::Ok(ok) => Ok(Ptr::from(ok)),
            Result::Err(e) => Err(CodegenError::AllocErr(e)),
        }
    }
}

pub struct CodegenModule<'ctx>(Module<'ctx>);

impl<'ctx> CodegenModule<'ctx> {
    pub fn run_passes(&self, target_machine: &TargetMachine, level: u8) {
        assert!((0..=3).contains(&level));
        let passes = format!("default<O{}>", level);

        // TODO: custom passes:
        //let passes = format!(
        //   "module(cgscc(inline),function({}))",
        //   ["instcombine", "reassociate", "gvn", "simplifycfg",
        //"mem2reg",].join(","), );

        self.0
            .run_passes(&passes, target_machine, PassBuilderOptions::create())
            .unwrap();
    }

    pub fn jit_run_fn(&mut self, fn_name: &str) -> CodegenResult<f64> {
        match self.0.create_jit_execution_engine(OptimizationLevel::None) {
            Ok(jit) => {
                Ok(unsafe { jit.get_function::<unsafe extern "C" fn() -> f64>(fn_name)?.call() })
            },
            Err(err) => Err(CodegenError::CouldntCreateJit(err)),
        }
    }

    pub fn get_functions(&self) -> inkwell::module::FunctionIterator {
        self.0.get_functions()
    }

    pub fn print_to_stderr(&self) {
        self.0.print_to_stderr()
    }

    pub fn get_inner(&self) -> &Module<'ctx> {
        &self.0
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
    //     debug_assert!(matches!(ty, Type::Void | Type::Never));
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
        // debug_assert!(matches!(self.ty, Type::Int { .. }));
        unsafe { IntValue::new(self.val) }
    }

    pub fn bool_val(&self) -> IntValue<'ctx> {
        // debug_assert!(matches!(self.ty, Type::Int { .. }));
        unsafe { IntValue::new(self.val) }
    }

    pub fn float_val(&self) -> FloatValue<'ctx> {
        // debug_assert!(matches!(self.ty, Type::Float { .. }));
        unsafe { FloatValue::new(self.val) }
    }

    pub fn ptr_val(&self) -> PointerValue<'ctx> {
        // debug_assert!(matches!(self.ty, Type::Ptr(_)));
        unsafe { PointerValue::new(self.val) }
    }

    pub fn struct_val(&self) -> StructValue<'ctx> {
        // debug_assert!(matches!(self.ty, Type::Struct { .. }));
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
        // // debug_assert!(matches!(self.ty, Type::Int { .. }));
        unsafe { IntType::new(self.inner) }
    }

    pub fn bool_ty(&self) -> IntType<'ctx> {
        // debug_assert!(matches!(self.ty, Type::Bool));
        unsafe { IntType::new(self.inner) }
    }

    pub fn float_ty(&self) -> FloatType<'ctx> {
        // debug_assert!(matches!(self.ty, Type::Float { .. }));
        unsafe { FloatType::new(self.inner) }
    }

    pub fn ptr_ty(&self) -> PointerType<'ctx> {
        // debug_assert!(matches!(self.ty, Type::Ptr(..)));
        unsafe { PointerType::new(self.inner) }
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
