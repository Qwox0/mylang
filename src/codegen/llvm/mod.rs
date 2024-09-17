#![allow(unused)]

use self::jit::Jit;
use crate::{
    ast::{BinOpKind, DeclMarkers, Expr, ExprKind, Fn, Ident, LitKind, PreOpKind, Type, VarDecl},
    cli::DebugOptions,
    parser::{lexer::Code, ParseError, StmtIter},
    ptr::Ptr,
    symbol_table::SymbolTable,
    util::UnwrapDebug,
};
use inkwell::{
    basic_block::BasicBlock,
    builder::{Builder, BuilderError},
    context::Context,
    execution_engine::{ExecutionEngine, FunctionLookupError},
    llvm_sys::{
        core::LLVMTypeOf,
        prelude::{LLVMPassManagerRef, LLVMTypeRef, LLVMValueRef},
        LLVMPassManager, LLVMType, LLVMValue,
    },
    module::{Linkage, Module},
    object_file::ObjectFile,
    passes::{PassBuilderOptions, PassManager},
    support::LLVMString,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{
        AnyType, AnyTypeEnum, AsTypeRef, BasicMetadataTypeEnum, BasicType, BasicTypeEnum,
        FloatType, IntType, PointerType,
    },
    values::{
        AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValue, BasicValueEnum,
        FloatValue, FunctionValue, GlobalValue, IntValue, PointerValue, StructValue,
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
    InvalidCallProduced,
    FunctionLookupError(FunctionLookupError),
    CouldntCreateJit(LLVMString),
    CannotMutateRegister,
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

    symbols: SymbolTable<Symbol<'ctx>>,

    /// the current fn being compiled
    cur_fn: Option<FunctionValue<'ctx>>,
    cur_var_ident: Option<Ident>,
    // errors: Vec<CodegenError>,
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
            cur_fn: None,
            cur_var_ident: None,
            // errors: vec![],
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

    pub fn compile_expr(&mut self, expr: Ptr<Expr>) -> CodegenResult<CodegenValue<'ctx>> {
        macro_rules! v {
            (x $val:expr) => {
                CodegenValue::new($val, expr.ty)
            };
            (x $val:expr, $ty:expr) => {
                CodegenValue::new($val, $ty)
            };
            ($val:expr) => {
                Ok(CodegenValue::new($val, expr.ty))
            };
            ($val:expr, $ty:expr) => {
                Ok(CodegenValue::new($val, $ty))
            };
        }

        match &expr.kind {
            ExprKind::Ident(name) => {
                // TODO: ident which is not a variable
                v!(match self.symbols.get(name).unwrap_debug() {
                    Symbol::Stack(ptr) => {
                        let ty = self.llvm_type(expr.ty).basic_ty();
                        self.builder.build_load(ty, *ptr, name)?.as_value_ref()
                    },
                    Symbol::Register(reg) => reg.as_value_ref(),
                    _ => todo!(),
                })
            },
            ExprKind::Literal { kind, code } => {
                v!(match kind {
                    LitKind::Char => todo!(),
                    LitKind::BChar => todo!(),
                    LitKind::Int => {
                        self.float_type(expr.ty).const_float_from_string(code).as_value_ref() // TODO: int
                    },
                    LitKind::Float =>
                        self.float_type(expr.ty).const_float_from_string(code).as_value_ref(),
                    LitKind::Str => todo!(),
                })
            },
            ExprKind::BoolLit(bool) => {
                debug_assert!(expr.ty == Type::Bool);
                let b_ty = self.context.bool_type();
                v!(if *bool { b_ty.const_all_ones() } else { b_ty.const_zero() }.as_value_ref())
            },
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn(_) => todo!(),
            ExprKind::Parenthesis { expr } => self.compile_expr(*expr),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                self.symbols.open_scope();
                let res = try {
                    let mut out = CodegenValue::new_void(Type::Void);
                    for s in stmts.iter() {
                        out = self.compile_expr(*s)?;
                        if out.ty == Type::Never {
                            break;
                        }
                    }
                    debug_assert!(expr.ty == out.ty);
                    if !*has_trailing_semicolon || out.ty == Type::Never {
                        out
                    } else {
                        CodegenValue::new_void(expr.ty)
                    }
                };
                self.symbols.close_scope();
                res
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
                let func = func.try_to_ident().unwrap_debug();
                let (&params, &func) =
                    self.symbols.get(&func.text).unwrap_debug().as_fn().unwrap_debug();

                let args = params
                    .as_ref()
                    .iter()
                    .enumerate()
                    .map(|(idx, param)| {
                        let arg = args.get(idx).copied().or(param.default).unwrap_debug();
                        self.compile_expr(arg).map(|v| v.basic_metadata_val())
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                v!(self.builder.build_call(func, &args, "call")?.as_value_ref())
            },
            ExprKind::PreOp { kind, expr, .. } => {
                let expr = self.compile_expr(*expr)?;
                v!(match kind {
                    PreOpKind::AddrOf => todo!(),
                    PreOpKind::AddrMutOf => todo!(),
                    PreOpKind::Deref => todo!(),
                    PreOpKind::Not => todo!(),
                    PreOpKind::Neg => match expr.ty {
                        Type::Float { .. } =>
                            self.builder.build_float_neg(expr.float_val(), "neg")?.as_value_ref(),
                        _ => todo!(),
                    },
                }) // TODO: maybe different return types possible in the future
            },
            ExprKind::BinOp { lhs, op, rhs, .. } => {
                let lhs = self.compile_expr(*lhs)?;
                let rhs = self.compile_expr(*rhs)?;
                debug_assert!(lhs.ty == rhs.ty);
                v!(match lhs.ty {
                    Type::Float { .. } => {
                        self.build_float_binop(lhs.float_val(), rhs.float_val(), *op)?
                            .as_value_ref()
                    },
                    _ => todo!(),
                })
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                debug_assert!(lhs.ty == rhs.ty);
                let var_name = &*lhs.try_to_ident().unwrap_debug().text;
                match self.symbols.get(var_name).unwrap_debug() {
                    &Symbol::Stack(stack_var) => {
                        let rhs = self.compile_expr(*rhs)?.basic_val();
                        self.builder.build_store(stack_var, rhs)?;
                    },
                    Symbol::Register(_) => return Err(CodegenError::CannotMutateRegister),
                    Symbol::Function { .. } => panic!("cannot assign to function"),
                }
                Ok(CodegenValue::new_void(expr.ty))
            },
            ExprKind::BinOpAssign { lhs: lhs_expr, op, rhs: rhs_expr, .. } => {
                debug_assert!(lhs_expr.ty == rhs_expr.ty);
                let var_name = &*lhs_expr.try_to_ident().unwrap_debug().text;
                match self.symbols.get(var_name).unwrap_debug() {
                    &Symbol::Stack(stack_var) => {
                        let lhs_ty = self.llvm_type(lhs_expr.ty).basic_ty();
                        let lhs = self.builder.build_load(lhs_ty, stack_var, "lhs")?;
                        let rhs = self.compile_expr(*rhs_expr)?;
                        let binop_res = match lhs_expr.ty {
                            Type::Float { .. } => self
                                .build_float_binop(lhs.into_float_value(), rhs.float_val(), *op)?
                                .as_basic_value_enum(),
                            _ => todo!(),
                        };
                        self.builder.build_store(stack_var, binop_res)?;
                    },
                    Symbol::Register(_) => return Err(CodegenError::CannotMutateRegister),
                    Symbol::Function { .. } => panic!("cannot assign to function"),
                }
                Ok(CodegenValue::new_void(expr.ty))
            },
            ExprKind::VarDecl(VarDecl { markers, ident, ty, default: init, is_const }) => {
                let var_name = &*ident.text;
                let is_mut = markers.is_mut;
                let v = if let Some(Expr { kind: ExprKind::Fn(f), .. }) = init.map(|p| p.as_ref()) {
                    let val = self.compile_fn(var_name, f)?;
                    Symbol::Function { params: f.params, val }
                } else if !is_mut && let Some(init) = init {
                    Symbol::Register(self.compile_expr(*init)?.any_val())
                } else {
                    let ty = self.llvm_type(*ty).basic_ty();
                    let stack_var = self.builder.build_alloca(ty, &var_name)?;
                    if let Some(init) = init {
                        let init = self.compile_expr(*init)?.basic_val();
                        self.builder.build_store(stack_var, init)?;
                    }
                    Symbol::Stack(stack_var)
                };
                let old = self.symbols.insert(var_name.to_string(), v);
                if old.is_some() {
                    //println!("LOG: '{}' was shadowed in the same scope",
                    // var_name);
                }
                Ok(CodegenValue::new_void(expr.ty))
            },
            ExprKind::If { condition, then_body, else_body } => {
                let func = self.cur_fn.unwrap_debug();
                //let zero = self.context.f64_type().const_float(0.0);

                let condition = self.compile_expr(*condition)?;
                debug_assert!(condition.ty == Type::Bool);
                //let condition = self.builder.build_float_compare( FloatPredicate::ONE,
                // condition, zero, "ifcond",)?;
                let condition = condition.bool_val();

                let mut then_bb = self.context.append_basic_block(func, "then");
                let mut else_bb = self.context.append_basic_block(func, "else");
                let merge_bb = self.context.append_basic_block(func, "ifmerge");

                self.builder.build_conditional_branch(condition, then_bb, else_bb)?;

                self.builder.position_at_end(then_bb);
                let then_val = self.compile_expr(*then_body)?;
                self.builder.build_unconditional_branch(merge_bb)?;
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_val = if let Some(else_body) = else_body {
                    self.compile_expr(*else_body)?
                } else {
                    CodegenValue::new_void(Type::Void)
                };
                debug_assert!(then_val.ty == else_val.ty);
                debug_assert!(then_val.ty == expr.ty);
                self.builder.build_unconditional_branch(merge_bb)?;
                else_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);
                if expr.ty == Type::Void {
                    Ok(CodegenValue::new_void(expr.ty))
                } else {
                    let out_ty = self.llvm_type(expr.ty).basic_ty();

                    let phi = self.builder.build_phi(out_ty, "ifexpr")?;
                    phi.add_incoming(&[
                        (&then_val.basic_val(), then_bb),
                        (&else_val.basic_val(), else_bb),
                    ]);

                    // self.module.print_to_stderr();

                    v!(phi.as_value_ref())
                }
            },
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => todo!(),
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Return { expr: val } => {
                if let Some(val) = val {
                    let val = self.compile_expr(*val)?;
                    self.build_return(val)?;
                } else {
                    self.builder.build_return(None)?;
                }
                Ok(self.never(expr.ty))
            },
            ExprKind::Semicolon(_) => todo!(),
        }
    }

    pub fn compile_fn(&mut self, name: &str, f: &Fn) -> CodegenResult<FunctionValue<'ctx>> {
        let fn_val = self.compile_prototype(name, f.params);
        self.compile_fn_body(fn_val, f)
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

        self.symbols.open_scope();
        let res = try {
            self.symbols.reserve(f.params.len());

            for (param, param_def) in func.get_param_iter().zip(f.params.iter()) {
                let pname = &param_def.ident.text;
                self.symbols.insert(
                    pname.to_string(),
                    if param_def.markers.is_mut {
                        let alloca = self.create_entry_block_alloca(func, pname);
                        self.builder.build_store(alloca, param)?;
                        Symbol::Stack(alloca)
                    } else {
                        let reg_val = param.as_any_value_enum();
                        Symbol::Register(reg_val)
                    },
                );
            }

            let body = self.compile_expr(f.body)?;
            debug_assert!(f.ret_type == body.ty || body.ty == Type::Never);
            self.build_return(body)?;

            // self.module.print_to_stderr();

            if func.verify(true) {
                func
            } else {
                unsafe { func.delete() };
                Err(CodegenError::InvalidGeneratedFunction)?
            }
        };
        self.symbols.close_scope();
        res
    }

    // -----------------------

    fn build_return(&self, val: CodegenValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
        match val.ty {
            Type::Never => {},
            Type::Void => {
                self.builder.build_return(None)?;
            },
            _ => {
                debug_assert!(!val.val.is_null());
                self.builder.build_return(Some(&val.basic_val()))?;
            },
        }
        Ok(self.never(Type::Never))
    }

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

    /// LLVM does not make a distinction between signed and unsigned integer
    /// type
    fn int_type(&self, ty: Type) -> IntType<'ctx> {
        match ty {
            Type::Int { bits: 8, .. } => self.context.i8_type(),
            Type::Int { bits: 16, .. } => self.context.i16_type(),
            Type::Int { bits: 32, .. } => self.context.i32_type(),
            Type::Int { bits: 64, .. } => self.context.i64_type(),
            Type::Int { bits: 128, .. } => self.context.i128_type(),
            Type::Int { .. } => todo!("other float bits"),
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
        CodegenType::new(
            match ty {
                Type::Void => self.context.void_type().as_type_ref(),
                Type::Never => todo!(),
                Type::Int { bits, is_signed } => self.int_type(ty).as_type_ref(),
                Type::IntLiteral => todo!(),
                Type::Bool => self.context.bool_type().as_type_ref(),
                Type::Float { bits } => self.float_type(ty).as_type_ref(),
                Type::FloatLiteral => todo!(),
                Type::Function(_) => todo!(),
                Type::Unset => todo!(),
                Type::Unevaluated(_) => todo!(),
            },
            ty,
        )
    }

    fn never(&self, ty: Type) -> CodegenValue<'ctx> {
        debug_assert!(ty == Type::Never);
        CodegenValue::new(std::ptr::null_mut(), ty)
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
enum Symbol<'ctx> {
    Stack(PointerValue<'ctx>),
    Register(AnyValueEnum<'ctx>),
    Function {
        /// TODO: think of a better way to store the fn definition
        params: Ptr<[VarDecl]>,
        val: FunctionValue<'ctx>,
    },
}

impl<'ctx> Symbol<'ctx> {
    fn as_fn(&self) -> Option<(&Ptr<[VarDecl]>, &FunctionValue<'ctx>)> {
        match self {
            Symbol::Function { params, val } => Some((params, val)),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct CodegenValue<'ctx> {
    val: *mut LLVMValue,
    ty: Type,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> CodegenValue<'ctx> {
    pub fn new(val: *mut LLVMValue, ty: Type) -> CodegenValue<'ctx> {
        CodegenValue { val, ty, _marker: PhantomData }
    }

    pub fn new_void(ty: Type) -> CodegenValue<'ctx> {
        debug_assert!(ty == Type::Void);
        CodegenValue::new(std::ptr::null_mut(), ty)
    }

    pub fn as_type(&self) -> CodegenType {
        CodegenType::new(unsafe { LLVMTypeOf(self.val) }, self.ty)
    }

    pub fn int_val(&self) -> IntValue<'ctx> {
        debug_assert!(matches!(self.ty, Type::Int { .. }));
        unsafe { IntValue::new(self.val) }
    }

    pub fn bool_val(&self) -> IntValue<'ctx> {
        debug_assert!(matches!(self.ty, Type::Bool));
        unsafe { IntValue::new(self.val) }
    }

    pub fn float_val(&self) -> FloatValue<'ctx> {
        debug_assert!(matches!(self.ty, Type::Float { .. }));
        unsafe { FloatValue::new(self.val) }
    }

    pub fn ptr_val(&self) -> PointerValue<'ctx> {
        debug_assert!(todo!());
        unsafe { PointerValue::new(self.val) }
    }

    pub fn basic_val(&self) -> BasicValueEnum<'ctx> {
        // TODO: debug_assert?
        unsafe { BasicValueEnum::new(self.val) }
    }

    pub fn basic_metadata_val(&self) -> BasicMetadataValueEnum<'ctx> {
        // TODO: debug_assert?
        BasicMetadataValueEnum::try_from(self.any_val()).unwrap_debug()
    }

    pub fn any_val(&self) -> AnyValueEnum<'ctx> {
        // TODO: debug_assert?
        unsafe { AnyValueEnum::new(self.val) }
    }
}

pub struct CodegenType<'ctx> {
    inner: *mut LLVMType,
    ty: Type,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> CodegenType<'ctx> {
    pub fn new(raw: *mut LLVMType, ty: Type) -> CodegenType<'ctx> {
        CodegenType { inner: raw, ty, _marker: PhantomData }
    }

    pub fn int_ty(&self) -> IntType<'ctx> {
        debug_assert!(matches!(self.ty, Type::Int { .. }));
        unsafe { IntType::new(self.inner) }
    }

    pub fn bool_ty(&self) -> IntType<'ctx> {
        debug_assert!(matches!(self.ty, Type::Bool));
        unsafe { IntType::new(self.inner) }
    }

    pub fn float_ty(&self) -> FloatType<'ctx> {
        debug_assert!(matches!(self.ty, Type::Float { .. }));
        unsafe { FloatType::new(self.inner) }
    }

    pub fn ptr_ty(&self) -> PointerType<'ctx> {
        debug_assert!(todo!());
        unsafe { PointerType::new(self.inner) }
    }

    pub fn basic_ty(&self) -> BasicTypeEnum<'ctx> {
        unsafe { BasicTypeEnum::new(self.inner) }
    }

    pub fn any_ty(&self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(self.inner) }
    }
}
