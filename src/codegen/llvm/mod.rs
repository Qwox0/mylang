#![allow(unused)]

use self::jit::Jit;
use crate::{
    ast::{BinOpKind, DeclMarkers, Expr, ExprKind, Ident, LitKind, PreOpKind, Type, VarDecl},
    cli::DebugOptions,
    parser::{lexer::Code, ParseError, StmtIter},
    ptr::Ptr,
    symbol_table::SymbolTable,
};
use inkwell::{
    basic_block::BasicBlock,
    builder::{Builder, BuilderError},
    context::Context,
    execution_engine::{ExecutionEngine, FunctionLookupError},
    llvm_sys::{
        prelude::{LLVMPassManagerRef, LLVMValueRef},
        LLVMPassManager,
    },
    module::{Linkage, Module},
    object_file::ObjectFile,
    passes::{PassBuilderOptions, PassManager},
    support::LLVMString,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{BasicMetadataTypeEnum, BasicType},
    values::{
        AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValueEnum, FloatValue,
        FunctionValue, PointerValue,
    },
    FloatPredicate, OptimizationLevel,
};
use std::{collections::HashMap, marker::PhantomData, mem::MaybeUninit};

pub mod jit;

const REPL_EXPR_ANON_FN_NAME: &str = "__repl_expr";

#[derive(Debug)]
pub enum CodegenError {
    BuilderError(BuilderError),
    InvalidGeneratedFunction,
    MismatchedArgCount { expected: usize, got: usize },
    InvalidCallProduced,
    FunctionLookupError(FunctionLookupError),

    RedefinedItem(Ptr<str>),
    UnknownIdent(Ptr<str>),
    UnknownFn(Ptr<str>),
    NotAFn(Ptr<str>),

    CouldntCreateJit(LLVMString),
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

pub struct Codegen<'ctx> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Module<'ctx>,

    symbols: SymbolTable<LLVMSymbol<'ctx>>,

    /// the current fn being compiled
    cur_fn: Option<FunctionValue<'ctx>>,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(
        context: &'ctx Context,
        builder: Builder<'ctx>,
        module: Module<'ctx>,
    ) -> Codegen<'ctx> {
        Codegen { context, builder, module, symbols: SymbolTable::with_one_scope(), cur_fn: None }
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

    pub fn move_module_to(&mut self, jit: &mut Jit<'ctx>) {
        let module = self.take_module();
        jit.add_module(module);
    }

    pub fn into_module(self) -> CodegenModule<'ctx> {
        CodegenModule(self.module)
    }

    /// for functions, this adds only the prototype, not the function body
    pub fn add_top_level_symbol(&mut self, expr: Ptr<Expr>) {
        match &expr.kind {
            ExprKind::VarDecl(VarDecl {
                markers,
                ident,
                ty,
                default: Some(init),
                is_const: true,
            }) => match &init.kind {
                ExprKind::Fn { params, ret_type, body } => {
                    self.compile_prototype(&ident.text, *params);
                },
                _ => todo!(),
            },
            ExprKind::Semicolon(Some(expr)) => self.add_top_level_symbol(*expr),
            ExprKind::Semicolon(None) => (),
            _ => todo!(),
        }
    }

    pub fn compile_top_level_fn_body(&mut self, expr: &Expr) -> CodegenResult<()> {
        match &expr.kind {
            ExprKind::VarDecl(VarDecl {
                markers,
                ident,
                ty,
                default: Some(init),
                is_const: true,
            }) => {
                match &init.kind {
                    ExprKind::Fn { params, ret_type, body } => {
                        self.compile_fn_body(&ident.text, *params, *body);
                    },
                    _ => todo!(),
                }
                Ok(())
            },
            ExprKind::Semicolon(Some(expr)) => self.compile_top_level_fn_body(expr),
            ExprKind::Semicolon(None) => Ok(()),
            _ => todo!(),
        }
    }

    pub fn compile_prototype(&mut self, name: &str, params: Ptr<[VarDecl]>) -> FunctionValue<'ctx> {
        let ret_type = self.context.f64_type();
        let args_types = params
            .iter()
            .map(|VarDecl { .. }| BasicMetadataTypeEnum::FloatType(self.context.f64_type()))
            .collect::<Vec<_>>();
        let fn_type = ret_type.fn_type(&args_types, false);
        let fn_val = self.module.add_function(name, fn_type, Some(Linkage::External));

        for (idx, param) in fn_val.get_param_iter().enumerate() {
            param.set_name(&params[idx].ident.text)
        }

        self.symbols
            .insert(name.to_string(), LLVMSymbol::Function { params, val: fn_val });

        fn_val
    }

    pub fn compile_fn_body(
        &mut self,
        name: &str,
        params: Ptr<[VarDecl]>,
        body: Ptr<Expr>,
    ) -> CodegenResult<FunctionValue<'ctx>> {
        let func = self.module.get_function(name).unwrap();
        let entry = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        self.cur_fn = Some(func);

        self.symbols.open_scope();
        self.symbols.reserve(params.len());

        for (param, param_def) in func.get_param_iter().zip(params.iter()) {
            let pname = &param_def.ident.text;
            self.symbols.insert(
                pname.to_string(),
                if param_def.markers.is_mut {
                    let alloca = self.create_entry_block_alloca(func, pname);
                    self.builder.build_store(alloca, param)?;
                    LLVMSymbol::Stack(alloca)
                } else {
                    let reg_val = param.as_any_value_enum();
                    LLVMSymbol::Register(reg_val)
                },
            );
        }

        let body = self.compile_expr_to_float(body)?;
        self.builder.build_return(Some(&body))?;

        self.symbols.close_scope();
        if func.verify(true) {
            Ok(func)
        } else {
            unsafe { func.delete() };
            Err(CodegenError::InvalidGeneratedFunction)
        }
    }

    pub fn compile_expr_to_float(&mut self, expr: Ptr<Expr>) -> CodegenResult<FloatValue<'ctx>> {
        match &expr.kind {
            ExprKind::Ident(name) => {
                // TODO: ident which is not a variable
                let Some(var) = self.symbols.get(name) else {
                    return Err(CodegenError::UnknownIdent(*name));
                };

                let f = match var {
                    LLVMSymbol::Stack(ptr) => self
                        .builder
                        .build_load(self.context.f64_type(), *ptr, name)?
                        .into_float_value(),
                    LLVMSymbol::Register(reg) => reg.into_float_value(),
                    _ => todo!(),
                };

                Ok(f)
            },
            ExprKind::Literal { kind: LitKind::Int | LitKind::Float, code } => {
                Ok(self.context.f64_type().const_float_from_string(code))
            },
            ExprKind::BoolLit(bool) => {
                Ok(self.context.f64_type().const_float(if *bool { 1.0 } else { 0.0 }))
            },
            ExprKind::Literal { .. } => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { params, ret_type, body } => todo!(),
            ExprKind::Parenthesis { expr } => self.compile_expr_to_float(*expr),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                self.symbols.open_scope();
                let stmts = unsafe { stmts.as_ref() };
                let Some((last, stmts)) = stmts.split_last() else {
                    return Ok(self.context.f64_type().const_float(0.0)); // TODO: return void
                };
                for s in stmts {
                    let _ = self.compile_expr_to_float(*s)?;
                }
                let out = self.compile_expr_to_float(*last)?;
                self.symbols.close_scope();
                Ok(if *has_trailing_semicolon {
                    self.context.f64_type().const_float(0.0) // TODO: return void
                } else {
                    out
                })
            },
            ExprKind::StructDef(_) => todo!(),
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs, fields } => panic!(),
            ExprKind::Dot { lhs, rhs } => todo!(),
            //ExprKind::Colon { lhs, rhs } => todo!(),
            ExprKind::PostOp { kind, expr } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            //ExprKind::CompCall { func, args } => todo!(),
            ExprKind::Call { func, args } => {
                let Ok(func) = unsafe { func.as_ref() }.try_to_ident() else {
                    todo!("non ident function call")
                };
                let (params, func) = match self.symbols.get(&func.text) {
                    Some(LLVMSymbol::Function { params, val }) => (params, *val),
                    Some(_) => panic!("cannot call a non function"), // TODO
                    None => {
                        return Err(CodegenError::UnknownFn(func.text));
                    },
                };
                let expected_arg_count = params.len();
                let args = unsafe { args.as_ref() };
                let args = unsafe { params.as_ref() }
                    .iter()
                    .enumerate()
                    .map(|(idx, param)| match args.get(idx).or(param.default.as_ref()) {
                        Some(&arg) => self.compile_expr_to_float(arg).map(Into::into),
                        None => Err(CodegenError::MismatchedArgCount {
                            expected: expected_arg_count,
                            got: args.len(),
                        }),
                    })
                    .collect::<Result<Vec<BasicMetadataValueEnum>, _>>()?;

                match self.builder.build_call(func, &args, "call")?.try_as_basic_value().left() {
                    Some(v) => Ok(v.into_float_value()),
                    None => Err(CodegenError::InvalidCallProduced),
                }
            },
            ExprKind::PreOp { kind, expr, .. } => {
                let expr = self.compile_expr_to_float(*expr)?;
                match kind {
                    PreOpKind::AddrOf => todo!(),
                    PreOpKind::AddrMutOf => todo!(),
                    PreOpKind::Deref => todo!(),
                    PreOpKind::Not => todo!(),
                    PreOpKind::Neg => Ok(self.builder.build_float_neg(expr, "neg")?),
                }
            },
            ExprKind::BinOp { lhs, op, rhs, .. } => {
                let lhs = self.compile_expr_to_float(*lhs)?;
                let rhs = self.compile_expr_to_float(*rhs)?;
                self.build_float_binop(lhs, rhs, *op)
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                let lhs = unsafe { lhs.as_ref() }.try_to_ident().expect("assign lhs is ident");
                let var_name = &*lhs.text;
                match self.symbols.get(var_name) {
                    Some(&LLVMSymbol::Stack(stack_var)) => {
                        let rhs = self.compile_expr_to_float(*rhs)?;
                        self.builder.build_store(stack_var, rhs)?;
                    },
                    Some(LLVMSymbol::Register(_)) => panic!("cannot assign to register"),
                    Some(LLVMSymbol::Function { .. }) => panic!("cannot assign to function"),
                    None => panic!("undefined variable: '{}'", var_name),
                }
                Ok(self.context.f64_type().const_zero()) // TODO: void
            },
            ExprKind::BinOpAssign { lhs, op, rhs, .. } => {
                let lhs = unsafe { lhs.as_ref() }.try_to_ident().expect("assign lhs is ident");
                let var_name = &*lhs.text;
                match self.symbols.get(var_name) {
                    Some(&LLVMSymbol::Stack(stack_var)) => {
                        let lhs = self
                            .builder
                            .build_load(self.context.f64_type(), stack_var, "tmp")?
                            .into_float_value();
                        let rhs = self.compile_expr_to_float(*rhs)?;
                        let binop_res = self.build_float_binop(lhs, rhs, *op)?;
                        self.builder.build_store(stack_var, binop_res)?;
                    },
                    Some(LLVMSymbol::Register(_)) => panic!("cannot assign to register"),
                    Some(LLVMSymbol::Function { .. }) => panic!("cannot assign to function"),
                    None => panic!("undefined variable: '{}'", var_name),
                }
                Ok(self.context.f64_type().const_zero()) // TODO: void
            },
            ExprKind::VarDecl(VarDecl { markers, ident, ty, default, is_const }) => {
                let init = default;
                let var_name = &*ident.text;
                let is_mut = markers.is_mut;
                let v = if !is_mut && let Some(init) = init {
                    let init = self.compile_expr_to_float(*init)?.as_any_value_enum();
                    LLVMSymbol::Register(init)
                } else {
                    let stack_var =
                        self.builder.build_alloca(self.context.f64_type(), &var_name)?;
                    if let Some(init) = init {
                        let init = self.compile_expr_to_float(*init)?;
                        self.builder.build_store(stack_var, init)?;
                    }
                    LLVMSymbol::Stack(stack_var)
                };
                let old = self.symbols.insert(var_name.to_string(), v);
                if old.is_some() {
                    println!("LOG: '{}' was shadowed in the same scope", var_name);
                }
                Ok(self.context.f64_type().const_zero()) // TODO: void
            },
            ExprKind::If { condition, then_body, else_body } => {
                let func = self.cur_fn.unwrap();
                let zero = self.context.f64_type().const_float(0.0);

                let condition = self.compile_expr_to_float(*condition)?;
                let condition = self.builder.build_float_compare(
                    FloatPredicate::ONE,
                    condition,
                    zero,
                    "ifcond",
                )?;

                let mut then_bb = self.context.append_basic_block(func, "then");
                let mut else_bb = self.context.append_basic_block(func, "else");
                let merge_bb = self.context.append_basic_block(func, "ifmerge");

                self.builder.build_conditional_branch(condition, then_bb, else_bb)?;

                self.builder.position_at_end(then_bb);
                let then_val = self.compile_expr_to_float(*then_body)?;
                self.builder.build_unconditional_branch(merge_bb)?;
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_val = if let Some(else_body) = else_body {
                    self.compile_expr_to_float(*else_body)?
                } else {
                    self.context.f64_type().const_zero()
                };
                self.builder.build_unconditional_branch(merge_bb)?;
                else_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(self.context.f64_type(), "ifexpr")?;
                phi.add_incoming(&[(&then_val, then_bb), (&else_val, else_bb)]);

                self.module.print_to_stderr();

                Ok(phi.as_basic_value().into_float_value())
            },
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => todo!(),
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Return { expr } => {
                if let Some(expr) = expr {
                    let val = self.compile_expr_to_float(*expr)?;
                    self.builder.build_return(Some(&val))?;
                } else {
                    self.builder.build_return(None)?;
                }
                Ok(self.context.f64_type().const_zero()) // TODO: type `never`
            },
            ExprKind::Semicolon(_) => todo!(),
        }
    }

    // -----------------------

    pub fn build_float_binop(
        &mut self,
        lhs: FloatValue<'ctx>,
        rhs: FloatValue<'ctx>,
        op: BinOpKind,
    ) -> CodegenResult<FloatValue<'ctx>> {
        match op {
            BinOpKind::Mul => Ok(self.builder.build_float_mul(lhs, rhs, "mul")?),
            BinOpKind::Div => Ok(self.builder.build_float_div(lhs, rhs, "div")?),
            BinOpKind::Mod => Ok(self.builder.build_float_rem(lhs, rhs, "mod")?),
            BinOpKind::Add => Ok(self.builder.build_float_add(lhs, rhs, "add")?),
            BinOpKind::Sub => Ok(self.builder.build_float_sub(lhs, rhs, "sub")?),
            BinOpKind::ShiftL => todo!(),
            BinOpKind::ShiftR => todo!(),
            BinOpKind::BitAnd => todo!(),
            BinOpKind::BitXor => todo!(),
            BinOpKind::BitOr => todo!(),
            BinOpKind::Eq => Ok(self.builder.build_signed_int_to_float(
                self.builder.build_float_compare(FloatPredicate::OEQ, lhs, rhs, "eq")?,
                self.context.f64_type(),
                "floatcast",
            )?),
            BinOpKind::Ne => Ok(self.builder.build_signed_int_to_float(
                self.builder.build_float_compare(FloatPredicate::ONE, lhs, rhs, "ne")?,
                self.context.f64_type(),
                "floatcast",
            )?),
            BinOpKind::Lt => Ok(self.builder.build_signed_int_to_float(
                self.builder.build_float_compare(FloatPredicate::OLT, lhs, rhs, "lt")?,
                self.context.f64_type(),
                "floatcast",
            )?),
            BinOpKind::Le => Ok(self.builder.build_signed_int_to_float(
                self.builder.build_float_compare(FloatPredicate::OLE, lhs, rhs, "le")?,
                self.context.f64_type(),
                "floatcast",
            )?),
            BinOpKind::Gt => Ok(self.builder.build_signed_int_to_float(
                self.builder.build_float_compare(FloatPredicate::OGT, lhs, rhs, "gt")?,
                self.context.f64_type(),
                "floatcast",
            )?),
            BinOpKind::Ge => Ok(self.builder.build_signed_int_to_float(
                self.builder.build_float_compare(FloatPredicate::OGE, lhs, rhs, "ge")?,
                self.context.f64_type(),
                "floatcast",
            )?),
            BinOpKind::And => todo!(),
            BinOpKind::Or => todo!(),
            BinOpKind::Range => todo!(),
            BinOpKind::RangeInclusive => todo!(),
        }
    }

    /// Creates a new stack allocation instruction in the entry block of the
    /// function. <https://github.com/TheDan64/inkwell/blob/5c9f7fcbb0a667f7391b94beb65f1a670ad13221/examples/kaleidoscope/implementation_typed_pointers.rs#L845-L857>
    fn create_entry_block_alloca(
        &self,
        fn_value: FunctionValue<'ctx>,
        name: &str,
    ) -> PointerValue<'ctx> {
        let entry = fn_value.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => self.builder.position_before(&first_instr),
            None => self.builder.position_at_end(entry),
        }

        self.builder.build_alloca(self.context.f64_type(), name).unwrap()
        /*
        let builder = self.context.create_builder();

        let entry = fn_value.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(self.context.f64_type(), name).unwrap()
            */
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

    pub fn debug_symbols(&self) {
        for map in self.symbols.inner() {
            println!("{:#?}", map);
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

#[derive(Debug)]
enum LLVMSymbol<'ctx> {
    Stack(PointerValue<'ctx>),
    Register(AnyValueEnum<'ctx>),
    Function {
        /// TODO: think of a better way to store the fn definition
        params: Ptr<[VarDecl]>,
        val: FunctionValue<'ctx>,
    },
}
