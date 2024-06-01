#![allow(unused)]

use crate::parser::{
    lexer::Code, BinOpKind, Expr, ExprKind, Ident, LetKind, LitKind, Stmt, StmtKind, Type,
};
use inkwell::{
    builder::{Builder, BuilderError},
    context::Context,
    llvm_sys::prelude::LLVMValueRef,
    module::Module,
    passes::PassBuilderOptions,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
    types::BasicMetadataTypeEnum,
    values::{
        AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValueEnum, FloatValue,
        FunctionValue, PointerValue,
    },
    OptimizationLevel,
};
use std::{
    collections::HashMap,
    mem::{uninitialized, MaybeUninit},
};

#[derive(Debug)]
pub enum CError {
    BuilderError(BuilderError),
    InvalidGeneratedFunction,
    UnknownFn,
    MismatchedArgCount { expected: u32, got: usize },
    InvalidCallProduced,
}

impl From<BuilderError> for CError {
    fn from(e: BuilderError) -> Self {
        CError::BuilderError(e)
    }
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

pub struct Compiler<'ctx, 'c> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Module<'ctx>,

    code: &'c Code,
    //variables: HashMap<String, PointerValue<'ctx>>,
    //variables: HashMap<String, LLVMValueRef>,
    variables: HashMap<String, AnyValueEnum<'ctx>>,
}

impl<'ctx, 'c> Compiler<'ctx, 'c> {
    pub fn new(
        context: &'ctx Context,
        builder: Builder<'ctx>,
        module: Module<'ctx>,
        code: &'c Code,
    ) -> Compiler<'ctx, 'c> {
        Compiler { context, builder, module, code, variables: HashMap::new() }
    }

    pub fn new_module(context: &'ctx Context, code: &'c Code, module_name: &str) -> Self {
        let builder = context.create_builder();
        let module = context.create_module(module_name);
        Compiler::new(context, builder, module, code)
    }

    pub fn compile_file(file_name: &str, code: &Code) {
        let context = Context::create();
        let builder = context.create_builder();
        let module = context.create_module(file_name);
    }

    pub fn compile_repl_line(
        context: &'ctx Context,
        builder: &Builder<'ctx>,
        prev_stmts: &mut Vec<()>,
        line: &Code,
    ) {

        // TODO: recompile every previously parsed function into the new module
    }

    /// if `stmt` is an [`Expr`] then this function returns the evaluated
    /// [`FloatValue`]
    pub fn compile_stmt(&self, stmt: &Stmt) -> Result<FloatValue<'_>, CError> {
        match &stmt.kind {
            StmtKind::Let { markers, ident, ty, kind } => todo!(),
            StmtKind::Semicolon(expr) => todo!(),
            StmtKind::Expr(expr) => self.compile_expr_to_float(&expr),
        }
    }

    /*
    pub fn debug_compile_expr(
        context: &'ctx Context,
        builder: &Builder<'ctx>,
        expr: &Expr,
        code: &Code,
    ) {
        let module = context.create_module("repl");
        let mut c = Compiler { context, builder, module: &module, code, variables: HashMap::new() };

        let args_types: &[BasicMetadataTypeEnum] = &[];

        let fn_type = c.context.f64_type().fn_type(args_types, false);
        let function = c.module.add_function("anonymous", fn_type, None);

        let entry = c.context.append_basic_block(function, "entry");
        c.builder.position_at_end(entry);

        let body = c.compile_expr_to_float(expr).unwrap();
        c.builder.build_return(Some(&body)).unwrap();
        let function = if function.verify(true) {
            Ok(function)
        } else {
            unsafe { function.delete() };
            Err("Invalid generated function.")
        }
        .unwrap();

        run_passes_on(&module);

        println!("-> Expression compiled to IR:");
        function.print_to_stderr();

        let ee = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();

        let fn_name = function.get_name().to_str().unwrap();
        let maybe_fn = unsafe { ee.get_function::<unsafe extern "C" fn() -> f64>(fn_name) };
        let compiled_fn = match maybe_fn {
            Ok(f) => f,
            Err(err) => {
                println!("!> Error during execution: {:?}", err);
                return;
            },
        };

        unsafe {
            println!("=> {}", compiled_fn.call());
        }
    }
    */
    pub fn compile_prototype(
        &self,
        name: &str,
        params: &[(Ident, Option<Expr>)],
    ) -> FunctionValue<'ctx> {
        let ret_type = self.context.f64_type();
        let args_types = params
            .iter()
            .map(|(name, ty)| BasicMetadataTypeEnum::FloatType(self.context.f64_type()))
            .collect::<Vec<_>>();
        let fn_type = ret_type.fn_type(&args_types, false);
        let fn_val = self.module.add_function(name, fn_type, None);

        for (idx, param) in fn_val.get_param_iter().enumerate() {
            param.set_name(&self.code[params[idx].0.span])
        }

        fn_val
    }

    pub fn compile_fn(
        &mut self,
        name: &str,
        params: &[(Ident, Option<Expr>)],
        body: &Expr,
    ) -> Result<FunctionValue<'ctx>, CError> {
        let func = self.compile_prototype(name, params);
        let entry = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        self.variables.reserve(params.len());

        for (idx, param) in func.get_param_iter().enumerate() {
            let pname = &self.code[params[idx].0.span];
            //let alloca = self.create_entry_block_alloca(func, pname);
            //self.builder.build_store(alloca, param)?;
            //self.variables.insert(pname.to_string(), alloca);

            let alloca = param.as_any_value_enum();
            //let alloca = param.as_value_ref();
            self.variables.insert(pname.to_string(), alloca);
        }

        let body = self.compile_expr_to_float(body)?;
        self.builder.build_return(Some(&body))?;

        if func.verify(true) {
            Ok(func)
        } else {
            unsafe { func.delete() };
            Err(CError::InvalidGeneratedFunction)
        }
    }

    pub fn compile_let(
        &mut self,
        ident: Ident,
        let_kind: LetKind,
    ) -> Result<FunctionValue<'ctx>, CError> {
        let LetKind::Init(init) = let_kind else { todo!("decl") };
        match init.kind {
            ExprKind::Ident => todo!(),
            ExprKind::Literal(_) => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { params, ret_type, body } => {
                let n = self.code[ident.span].to_string();
                self.compile_fn(&n, &params, &body)
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
        }
    }

    pub fn compile_repl_expr(&mut self, expr: &Expr) -> Result<FunctionValue<'ctx>, CError> {
        /*
        let fn_type = self.context.f64_type().fn_type(&[], false);
        let function = self.module.add_function("anonymous", fn_type, None);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let body = self.compile_expr_to_float(expr).unwrap();
        self.builder.build_return(Some(&body)).unwrap();
        if function.verify(true) {
            Ok(function)
        } else {
            unsafe { function.delete() };
            Err(CError::InvalidGeneratedFunction)
        }
        */

        self.compile_fn("anonymous", &[], expr)

        /*
        run_passes_on(&module);

        println!("-> Expression compiled to IR:");
        function.print_to_stderr();

        let ee = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();

        let fn_name = function.get_name().to_str().unwrap();
        let maybe_fn = unsafe { ee.get_function::<unsafe extern "C" fn() -> f64>(fn_name) };
        let compiled_fn = match maybe_fn {
            Ok(f) => f,
            Err(err) => {
                println!("!> Error during execution: {:?}", err);
                return;
            },
        };

        unsafe {
            println!("=> {}", compiled_fn.call());
        }
        */
    }

    /*
    pub fn new(module_name: &str, code: Code<'c>) -> Compiler<'ctx, 'c> {
        let context = Context::create();
        let builder = context.create_builder();
        let module = context.create_module(module_name);
        Compiler { context: &context, builder, module, code }
    }
    */
    pub fn build_load(&self, ptr: PointerValue<'ctx>, name: &str) -> BasicValueEnum<'ctx> {
        self.builder.build_load(self.context.f64_type(), ptr, name).unwrap()
    }

    pub fn compile_expr_to_float(&self, expr: &Expr) -> Result<FloatValue<'ctx>, CError> {
        match &expr.kind {
            ExprKind::Ident => {
                let name = &self.code[expr.span];
                // TODO: ident which is not a variable
                let var = self.variables.get(name).expect(&format!("has var definition: {}", name));
                //Ok(self.build_load(*var, name).into_float_value())
                //Ok(unsafe { FloatValue::new(*var) })
                Ok(var.into_float_value())
            },
            ExprKind::Literal(LitKind::Int | LitKind::Float) => {
                let s = &self.code[expr.span];
                Ok(self.context.f64_type().const_float_from_string(s))
            },
            ExprKind::Literal(_) => todo!(),
            ExprKind::ArraySemi { val, count } => todo!(),
            ExprKind::ArrayComma { elements } => todo!(),
            ExprKind::Tuple { elements } => todo!(),
            ExprKind::Fn { params, ret_type, body } => todo!(),
            ExprKind::Parenthesis { expr } => self.compile_expr_to_float(expr),
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
            ExprKind::Call { func, args } => {
                if !matches!(func.kind, ExprKind::Ident) {
                    todo!("non ident function call")
                }

                let func = &self.code[func.span];

                let Some(func) = self.module.get_function(func) else {
                    return Err(CError::UnknownFn);
                };
                let expected_arg_count = func.count_params();
                if expected_arg_count as usize != args.len() {
                    return Err(CError::MismatchedArgCount {
                        expected: expected_arg_count,
                        got: args.len(),
                    });
                }

                let args = args
                    .into_iter()
                    .map(|arg| self.compile_expr_to_float(arg).map(Into::into))
                    .collect::<Result<Vec<BasicMetadataValueEnum>, _>>()?;

                match self.builder.build_call(func, &args, "tmp")?.try_as_basic_value().left() {
                    Some(v) => Ok(v.into_float_value()),
                    None => Err(CError::InvalidCallProduced),
                }
            },
            ExprKind::PreOp { kind, expr } => todo!(),
            ExprKind::BinOp { lhs, op, rhs } => {
                let lhs = self.compile_expr_to_float(lhs)?;
                let rhs = self.compile_expr_to_float(rhs)?;
                match op {
                    BinOpKind::Mul => Ok(self.builder.build_float_mul(lhs, rhs, "tmpMul")?),
                    BinOpKind::Div => Ok(self.builder.build_float_div(lhs, rhs, "tmpDiv")?),
                    BinOpKind::Mod => Ok(self.builder.build_float_rem(lhs, rhs, "tmpMod")?),
                    BinOpKind::Add => Ok(self.builder.build_float_add(lhs, rhs, "tmpAdd")?),
                    BinOpKind::Sub => Ok(self.builder.build_float_sub(lhs, rhs, "tmpSub")?),
                    BinOpKind::ShiftL => todo!(),
                    BinOpKind::ShiftR => todo!(),
                    BinOpKind::BitAnd => todo!(),
                    BinOpKind::BitXor => todo!(),
                    BinOpKind::BitOr => todo!(),
                    BinOpKind::Eq => todo!(),
                    BinOpKind::Ne => todo!(),
                    BinOpKind::Lt => todo!(),
                    BinOpKind::Le => todo!(),
                    BinOpKind::Gt => todo!(),
                    BinOpKind::Ge => todo!(),
                    BinOpKind::And => todo!(),
                    BinOpKind::Or => todo!(),
                    BinOpKind::Range => todo!(),
                    BinOpKind::RangeInclusive => todo!(),
                }
            },
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::BinOpAssign { lhs, op, rhs } => todo!(),
        }
    }

    /// Creates a new stack allocation instruction in the entry block of the
    /// function. <https://github.com/TheDan64/inkwell/blob/5c9f7fcbb0a667f7391b94beb65f1a670ad13221/examples/kaleidoscope/implementation_typed_pointers.rs#L845-L857>
    fn create_entry_block_alloca(
        &self,
        fn_value: FunctionValue<'ctx>,
        name: &str,
    ) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();

        let entry = fn_value.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(self.context.f64_type(), name).unwrap()
    }
}

/// <https://github.com/TheDan64/inkwell/blob/5c9f7fcbb0a667f7391b94beb65f1a670ad13221/examples/kaleidoscope/main.rs#L82-L109>
fn run_passes_on(module: &Module) {
    Target::initialize_all(&InitializationConfig::default());
    let target_triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&target_triple).unwrap();
    let target_machine = target
        .create_target_machine(
            &target_triple,
            "generic",
            "",
            OptimizationLevel::None,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .unwrap();

    let passes: &[&str] = &[
        "instcombine",
        "reassociate",
        "gvn",
        "simplifycfg",
        // "basic-aa",
        "mem2reg",
    ];

    module
        .run_passes(passes.join(",").as_str(), &target_machine, PassBuilderOptions::create())
        .unwrap();
}
