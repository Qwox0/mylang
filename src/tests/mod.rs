use crate::error::Error;

mod logic_op;
mod parse_array;
mod parse_function;
mod todo;

macro_rules! jit_run_test {
    ($code:expr => $ret_type:ty) => {
        $crate::tests::jit_run_test_impl::<$ret_type>($code)
    };
}
pub(crate) use jit_run_test;

pub fn jit_run_test_impl<RetTy>(code: &impl std::fmt::Display) -> Result<RetTy, Error> {
    use crate::{codegen::llvm, compiler::Compiler, parser::StmtIter, sema::Sema};

    let alloc = bumpalo::Bump::new();

    let code = format!("test :: -> {code};");
    let code = code.as_ref();
    let stmts = StmtIter::parse_all_or_fail(code, &alloc);

    let sema = Sema::new(code, &alloc);
    let context = inkwell::context::Context::create();
    let codegen = llvm::Codegen::new_module(&context, "test", &alloc);
    let mut compiler = Compiler::new(sema, codegen);

    let _ = compiler.compile_stmts(&stmts);

    if !compiler.sema.errors.is_empty() {
        if compiler.sema.errors.len() > 2 {
            todo!("multiple sema errors")
        }
        return Err(Error::Sema(compiler.sema.errors.into_iter().next().unwrap()));
    }

    let target_machine = llvm::Codegen::init_target_machine();
    compiler.optimize(&target_machine, 0)?;

    println!("{}", compiler.codegen.module.print_to_string().to_string());

    Ok(compiler.codegen.jit_run_fn::<RetTy>("test", inkwell::OptimizationLevel::None)?)
}
