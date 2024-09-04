#![allow(unused)]

use self::jit::Jit;
use crate::{
    cli::DebugOptions,
    parser::{
        lexer::Code, BinOpKind, DeclMarkers, Expr, ExprKind, Ident, LitKind, PreOpKind, VarDeclKind,
    },
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
    types::BasicMetadataTypeEnum,
    values::{
        AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValueEnum, FloatValue,
        FunctionValue, PointerValue,
    },
    FloatPredicate, OptimizationLevel,
};
use std::{collections::HashMap, marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

pub mod jit;

const REPL_EXPR_ANON_FN_NAME: &str = "__repl_expr";

#[derive(Debug)]
pub enum CError {
    BuilderError(BuilderError),
    InvalidGeneratedFunction,
    MismatchedArgCount { expected: u32, got: usize },
    InvalidCallProduced,
    FunctionLookupError(FunctionLookupError),

    RedefinedItem(Box<str>),
    UnknownIdent(Box<str>),
    UnknownFn(Box<str>),
    NotAFn(Box<str>),

    CouldntCreateJit(LLVMString),
}

impl From<BuilderError> for CError {
    fn from(e: BuilderError) -> Self {
        CError::BuilderError(e)
    }
}

impl From<FunctionLookupError> for CError {
    fn from(e: FunctionLookupError) -> Self {
        CError::FunctionLookupError(e)
    }
}

pub trait Codegen {}

pub struct FileCodegen<'ctx> {
    pub context: &'ctx Context,
}

pub struct REPLCodegen<'ctx> {
    pub context: &'ctx Context,
}

/*
pub struct Compiler<'a, 'ctx, 'c> {
    pub context: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    pub module: &'a Module<'ctx>,

    code: &'c Code,
    variables: HashMap<String, PointerValue<'ctx>>,
}
*/

pub struct Compiler<'ctx, 'c, 'i> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Module<'ctx>,
    code: &'c Code,

    #[cfg(feature = "use_ptr_values")]
    variables: HashMap<String, PointerValue<'ctx>>,
    #[cfg(not(feature = "use_ptr_values"))]
    variables: HashMap<String, AnyValueEnum<'ctx>>,

    //items: HashMap<String, Item<'i>>,
    items: PhantomData<&'i ()>,

    /// the current fn being compiled
    cur_fn: Option<FunctionValue<'ctx>>,
}

impl<'ctx, 'c, 'i> Compiler<'ctx, 'c, 'i> {
    pub fn new(
        context: &'ctx Context,
        builder: Builder<'ctx>,
        module: Module<'ctx>,
        code: &'c Code,
    ) -> Compiler<'ctx, 'c, 'i> {
        Compiler {
            context,
            builder,
            module,
            code,
            variables: HashMap::new(),
            //items: HashMap::new(),
            items: PhantomData,
            cur_fn: None,
        }
    }

    pub fn new_module(context: &'ctx Context, module_name: &str, code: &'c Code) -> Self {
        let builder = context.create_builder();
        let module = context.create_module(module_name);
        Compiler::new(context, builder, module, code)
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

    pub fn compile_top_level(&mut self, expr: &Expr, code: &Code) -> Result<(), CError> {
        match &expr.kind {
            ExprKind::ConstDecl { markers, ident, ty, init } => {
                let ident = unsafe { ident.as_ref() };
                let init = unsafe { init.as_ref() };
                let name = code[ident.span].to_string();
                // if self.items.contains_key(&name) {
                //     return Err(CError::RedefinedItem(name.into_boxed_str()));
                // }
                self.compile_top_level_const_decl(markers, &name, ty, init, code)
            },
            ExprKind::Semicolon(expr) => self.compile_top_level(unsafe { expr.as_ref() }, code),
            _ => todo!(),
        }
    }

    pub fn compile_top_level_const_decl(
        &mut self,
        markers: &DeclMarkers,
        name: &str,
        ty: &Option<NonNull<Expr>>,
        init: &Expr,
        code: &Code,
    ) -> Result<(), CError> {
        match &init.kind {
            ExprKind::Ident => todo!(),
            ExprKind::Literal(_) => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { params, ret_type, body } => {
                self.compile_fn(name, unsafe { params.as_ref() }, *body, code).map(|_| ())
            },
            ExprKind::Parenthesis { expr } => todo!(),
            ExprKind::Block { stmts, .. } => todo!(),
            ExprKind::StructDef(_) => todo!(),
            ExprKind::TupleStructDef(_) => todo!(),
            ExprKind::Union {} => todo!(),
            ExprKind::Enum {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Initializer { lhs, fields } => panic!(),
            ExprKind::Dot { lhs, rhs } => todo!(),
            //ExprKind::Colon { lhs, rhs } => todo!(),
            ExprKind::PostOp { kind, expr } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            //ExprKind::CompCall { func, args } => todo!(),
            ExprKind::Call { func, args } => todo!(),
            ExprKind::PreOp { kind, expr } => todo!(),
            ExprKind::BinOp { lhs, op, rhs } => todo!(),
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => todo!(),
            ExprKind::If { condition, then_body, else_body } => todo!(),
            ExprKind::VarDecl { markers, ident, kind } => todo!(),
            ExprKind::ConstDecl { markers, ident, ty, init } => todo!(),
            ExprKind::Semicolon(_) => todo!(),
            _ => todo!(),
        }
    }

    pub fn compile_prototype(
        &self,
        name: &str,
        params: &[(Ident, Option<NonNull<Expr>>)],
        code: &Code,
        module: &Module<'ctx>,
    ) -> FunctionValue<'ctx> {
        let ret_type = self.context.f64_type();
        let args_types = params
            .iter()
            .map(|(name, ty)| BasicMetadataTypeEnum::FloatType(self.context.f64_type()))
            .collect::<Vec<_>>();
        let fn_type = ret_type.fn_type(&args_types, false);
        let fn_val = module.add_function(name, fn_type, Some(Linkage::External));

        for (idx, param) in fn_val.get_param_iter().enumerate() {
            param.set_name(&code[params[idx].0.span])
        }

        fn_val
    }

    pub fn compile_fn(
        &mut self,
        name: &str,
        params: &[(Ident, Option<NonNull<Expr>>)],
        body: NonNull<Expr>,
        code: &Code,
    ) -> Result<FunctionValue<'ctx>, CError> {
        let func = self.compile_prototype(name, params, code, &self.module);
        let entry = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        self.cur_fn = Some(func);

        self.variables.reserve(params.len());

        for (idx, param) in func.get_param_iter().enumerate() {
            let pname = &code[params[idx].0.span];

            #[cfg(feature = "use_ptr_values")]
            {
                let alloca = self.create_entry_block_alloca(func, pname);
                self.builder.build_store(alloca, param)?;
                self.variables.insert(pname.to_string(), alloca);
            }

            #[cfg(not(feature = "use_ptr_values"))]
            {
                let alloca = param.as_any_value_enum();
                self.variables.insert(pname.to_string(), alloca);
            }
        }

        let body = self.compile_expr_to_float(body, code)?;
        self.builder.build_return(Some(&body))?;

        if func.verify(true) {
            Ok(func)
        } else {
            unsafe { func.delete() };
            Err(CError::InvalidGeneratedFunction)
        }
    }

    pub fn compile_expr_to_float(
        &mut self,
        expr: NonNull<Expr>,
        code: &Code,
    ) -> Result<FloatValue<'ctx>, CError> {
        let expr = unsafe { expr.as_ref() };
        match &expr.kind {
            ExprKind::Ident => {
                let name = &code[expr.span];
                // TODO: ident which is not a variable
                let Some(var) = self.variables.get(name) else {
                    return Err(CError::UnknownIdent(Box::from(name)));
                };

                #[cfg(feature = "use_ptr_values")]
                let f = self.build_load(*var, name).into_float_value();
                #[cfg(not(feature = "use_ptr_values"))]
                let f = var.into_float_value();

                Ok(f)
            },
            ExprKind::Literal(LitKind::Int | LitKind::Float) => {
                let s = &code[expr.span];
                Ok(self.context.f64_type().const_float_from_string(s))
            },
            ExprKind::BoolLit(bool) => {
                Ok(self.context.f64_type().const_float(if *bool { 1.0 } else { 0.0 }))
            },
            ExprKind::Literal(_) => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { params, ret_type, body } => todo!(),
            ExprKind::Parenthesis { expr } => self.compile_expr_to_float(*expr, code),
            ExprKind::Block { stmts, .. } => todo!("Block"),
            ExprKind::StructDef(_) => todo!(),
            ExprKind::TupleStructDef(_) => todo!(),
            ExprKind::Union {} => todo!(),
            ExprKind::Enum {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Ptr { is_mut, ty } => todo!(),
            ExprKind::Initializer { lhs, fields } => panic!(),
            ExprKind::Dot { lhs, rhs } => todo!(),
            //ExprKind::Colon { lhs, rhs } => todo!(),
            ExprKind::PostOp { kind, expr } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            //ExprKind::CompCall { func, args } => todo!(),
            ExprKind::Call { func, args } => {
                let func = unsafe { func.as_ref() };
                if !matches!(func.kind, ExprKind::Ident) {
                    todo!("non ident function call")
                }

                let func = &code[func.span];
                let func = self.get_function(func)?;
                let expected_arg_count = func.count_params();
                if expected_arg_count as usize != args.len() {
                    return Err(CError::MismatchedArgCount {
                        expected: expected_arg_count,
                        got: args.len(),
                    });
                }

                let args = unsafe { args.as_ref() }
                    .into_iter()
                    .map(|arg| self.compile_expr_to_float(*arg, code).map(Into::into))
                    .collect::<Result<Vec<BasicMetadataValueEnum>, _>>()?;

                match self.builder.build_call(func, &args, "call")?.try_as_basic_value().left() {
                    Some(v) => Ok(v.into_float_value()),
                    None => Err(CError::InvalidCallProduced),
                }
            },
            ExprKind::PreOp { kind, expr } => {
                let expr = self.compile_expr_to_float(*expr, code)?;
                match kind {
                    PreOpKind::AddrOf => todo!(),
                    PreOpKind::AddrMutOf => todo!(),
                    PreOpKind::Deref => todo!(),
                    PreOpKind::Not => todo!(),
                    PreOpKind::Neg => Ok(self.builder.build_float_neg(expr, "neg")?),
                }
            },
            ExprKind::BinOp { lhs, op, rhs } => {
                let lhs = self.compile_expr_to_float(*lhs, code)?;
                let rhs = self.compile_expr_to_float(*rhs, code)?;
                let a = 1.0 <= 2.0;
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
            },
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => todo!(),
            ExprKind::VarDecl { markers, ident, kind } => todo!(),
            ExprKind::ConstDecl { markers, ident, ty, init } => todo!(),
            ExprKind::If { condition, then_body, else_body } => {
                let func = self.cur_fn.unwrap();
                let zero = self.context.f64_type().const_float(0.0);

                let condition = self.compile_expr_to_float(*condition, code)?;
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
                let then_val = self.compile_expr_to_float(*then_body, code)?;
                self.builder.build_unconditional_branch(merge_bb)?;
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_val = if let Some(else_body) = else_body {
                    self.compile_expr_to_float(*else_body, code)?
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
                    let val = self.compile_expr_to_float(*expr, code)?;
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

    /*
    pub fn add_item(&mut self, item: Item<'i>) -> Result<(), CError> {
        let name = item.code[item.ident.span].to_string();
        if self.items.contains_key(&name) {
            return Err(CError::RedefinedItem(name.into_boxed_str()));
        }
        self.compile_item(&item);
        self.items.insert(name, item);
        Ok(())
    }
    */

    /// if `stmt` is an [`Expr`] then this function returns the evaluated
    /// [`FloatValue`]
    pub fn compile_stmt(&mut self, stmt: &Expr, code: &Code) -> Result<FloatValue<'_>, CError> {
        todo!()
        /*
        match &stmt.kind {
            StmtKind::Decl { markers, ident, kind } => todo!(),
            StmtKind::Semicolon(expr) => todo!(),
            StmtKind::Expr(expr) => self.compile_expr_to_float(&expr, code),
        }
        */
    }

    pub fn get_function(&mut self, name: &str) -> Result<FunctionValue<'ctx>, CError> {
        if let Some(func) = self.module.get_function(name) {
            return Ok(func);
        };

        let (value, code): (Expr, &Code) = todo!("undefined function");
        /*
        let Some(Item { markers, ident, ty, value, code }) = self.items.get(name) else {
            return Err(CError::UnknownFn(Box::from(name)));
        };
        */

        let ExprKind::Fn { ref params, ref ret_type, ref body } = value.kind else {
            return Err(CError::NotAFn(Box::from(name)));
        };

        /*
        // SAFETY: borrowing self as mut and the fn item at the same time is safe
        // because `compile_fn` does not access `self.items`.
        let params =
            unsafe { std::ptr::from_ref::<[_]>(params.as_ref()).as_ref().unwrap_unchecked() };
        let body = unsafe { std::ptr::from_ref(body.as_ref()).as_ref().unwrap_unchecked() };
        */
        let params = params.clone();
        let body = body.clone();

        //self.compile_fn(name, &params, &body, code)
        Ok(self.compile_prototype(name, unsafe { params.as_ref() }, code, &self.module))
    }

    /*
    pub fn compile_item(&mut self, item: &Item<'i>) -> Result<FunctionValue<'ctx>, CError> {
        let code = item.code;
        match &item.value.kind {
            ExprKind::Ident => todo!(),
            ExprKind::Literal(_) => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { params, ret_type, body } => {
                let n = code[item.ident.span].to_string();
                self.compile_fn(&n, &params, &body, code)
            },
            ExprKind::Parenthesis { expr } => todo!(),
            ExprKind::Block { stmts } => todo!(),
            ExprKind::StructDef(_) => todo!(),
            ExprKind::StructInit { name, fields } => todo!(),
            ExprKind::TupleStructDef(_) => todo!(),
            ExprKind::Union {} => todo!(),
            ExprKind::Enum {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Dot { lhs, rhs } => todo!(),
            ExprKind::Colon { lhs, rhs } => todo!(),
            ExprKind::PostOp { kind, expr } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            ExprKind::CompCall { func, args } => todo!(),
            ExprKind::Call { func, args } => todo!(),
            ExprKind::PreOp { kind, expr } => todo!(),
            ExprKind::BinOp { lhs, op, rhs } => todo!(),
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => todo!(),
            ExprKind::Decl { markers, ident, kind } => todo!(),
            ExprKind::Semicolon(_) => todo!(),
        }
    }
    */

    pub fn compile_var_decl(
        &mut self,
        ident: Ident,
        kind: VarDeclKind,
        code: &Code,
    ) -> Result<FunctionValue<'ctx>, CError> {
        let init = match kind {
            VarDeclKind::WithTy { ty, init } => init.unwrap_or_else(|| todo!("decl")),
            VarDeclKind::InferTy { init } => init,
        };
        let init = unsafe { init.as_ref() };
        match &init.kind {
            ExprKind::Ident => todo!(),
            ExprKind::Literal(_) => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { params, ret_type, body } => {
                let n = code[ident.span].to_string();
                self.compile_fn(&n, unsafe { params.as_ref() }, *body, code)
            },
            ExprKind::Parenthesis { expr } => todo!(),
            ExprKind::Block { stmts, .. } => todo!(),
            ExprKind::StructDef(_) => todo!(),
            ExprKind::TupleStructDef(_) => todo!(),
            ExprKind::Union {} => todo!(),
            ExprKind::Enum {} => todo!(),
            ExprKind::OptionShort(_) => todo!(),
            ExprKind::Initializer { lhs, fields } => panic!(),
            ExprKind::Dot { lhs, rhs } => todo!(),
            //ExprKind::Colon { lhs, rhs } => todo!(),
            ExprKind::PostOp { kind, expr } => todo!(),
            ExprKind::Index { lhs, idx } => todo!(),
            //ExprKind::CompCall { func, args } => todo!(),
            ExprKind::Call { func, args } => todo!(),
            ExprKind::PreOp { kind, expr } => todo!(),
            ExprKind::BinOp { lhs, op, rhs } => todo!(),
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => todo!(),
            ExprKind::If { condition, then_body, else_body } => todo!(),
            ExprKind::VarDecl { markers, ident, kind } => todo!(),
            ExprKind::ConstDecl { markers, ident, ty, init } => todo!(),
            ExprKind::Semicolon(_) => todo!(),
            _ => todo!(),
        }
    }

    pub fn jit_run_fn(&mut self, fn_name: &str) -> Result<f64, CError> {
        match self.module.create_jit_execution_engine(OptimizationLevel::None) {
            Ok(jit) => {
                Ok(unsafe { jit.get_function::<unsafe extern "C" fn() -> f64>(fn_name)?.call() })
            },
            Err(err) => Err(CError::CouldntCreateJit(err)),
        }
    }

    /*
    pub fn jit_run_expr(
        &mut self,
        expr: &Expr,
        code: &Code,
        debug: Option<DebugOptions>,
    ) -> Result<f64, CError> {
        const REPL_EXPR_ANON_FN_NAME: &str = "__repl_expr";
        let func = self.compile_fn(REPL_EXPR_ANON_FN_NAME, &[], expr, code)?;

        self.run_passes_debug(debug);

        if debug == Some(DebugOptions::ReplExpr) {
            func.print_to_stderr();
        }

        let fn_name = func.get_name().to_str().unwrap();

        println!("1",);
        let out = self.jit_run_fn(REPL_EXPR_ANON_FN_NAME);
        println!("2",);
        self.replace_mod_same_name();
        println!("3",);
        out
    }
    */

    pub fn compile_repl_expr(
        &mut self,
        expr: NonNull<Expr>,
        code: &Code,
        debug: Option<DebugOptions>,
    ) -> Result<(), CError> {
        let func = self.compile_fn(REPL_EXPR_ANON_FN_NAME, &[], expr, code)?;

        // self.run_passes_debug(debug);

        if debug == Some(DebugOptions::ReplExpr) {
            func.print_to_stderr();
        }
        Ok(())
    }

    pub fn build_load(&self, ptr: PointerValue<'ctx>, name: &str) -> BasicValueEnum<'ctx> {
        self.builder.build_load(self.context.f64_type(), ptr, name).unwrap()
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
                OptimizationLevel::Default,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .unwrap()
    }

    /// <https://github.com/TheDan64/inkwell/blob/5c9f7fcbb0a667f7391b94beb65f1a670ad13221/examples/kaleidoscope/main.rs#L82-L109>
    pub fn run_passes(&self, target_machine: &TargetMachine) {
        let passes: &[&str] = &[
            "instcombine",
            "reassociate",
            "gvn",
            "simplifycfg",
            // "basic-aa",
            "mem2reg",
        ];

        self.module
            .run_passes(passes.join(",").as_str(), target_machine, PassBuilderOptions::create())
            .unwrap();
    }

    pub fn run_passes_debug(&self, target_machine: &TargetMachine, debug: Option<DebugOptions>) {
        if debug == Some(DebugOptions::LlvmIrUnoptimized) {
            self.module.print_to_stderr();
        }

        self.run_passes(target_machine);

        if debug == Some(DebugOptions::LlvmIrOptimized) {
            self.module.print_to_stderr();
        }
    }
}

fn run_passes_on(module: &Module) {}
