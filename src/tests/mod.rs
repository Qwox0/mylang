use crate::{error::Error, parser::parser_helper::ParserInterface, util::display_spanned_error};

mod binop;
mod enum_;
mod for_loop;
mod function;
mod if_;
mod initializer;
mod logic_binop;
mod parse_array;
mod parse_function;
mod ptr;
mod struct_;
mod todo;
mod union_;
mod while_loop;

macro_rules! jit_run_test {
    (raw $code:expr => $ret_type:ty) => {
        $crate::tests::jit_run_test_impl::<$ret_type>($code.as_ref())
    };
    ($code:expr => $ret_type:ty) => {{
        let code = format!("test :: -> {{ {} }};", $code);
        $crate::tests::jit_run_test_impl::<$ret_type>(code.as_ref())
    }};
}
pub(crate) use jit_run_test;

const DEBUG_TOKENS: bool = false;
const DEBUG_AST: bool = false;
const DEBUG_TYPES: bool = true;
const LLVM_OPTIMIZATION_LEVEL: u8 = 0;
const PRINT_ERROR: bool = true;

pub fn jit_run_test_impl<RetTy>(code: &crate::parser::lexer::Code) -> Result<RetTy, Error> {
    let res = jit_run_test_impl_inner(code);
    if PRINT_ERROR && let Err(err) = res.as_ref() {
        display_spanned_error(err, code);
    }
    res
}

pub fn jit_run_test_impl_inner<RetTy>(code: &crate::parser::lexer::Code) -> Result<RetTy, Error> {
    use crate::{
        codegen::llvm,
        compiler::Compiler,
        parser::{StmtIter, lexer::Lexer},
        sema::Sema,
    };

    let alloc = bumpalo::Bump::new();

    if DEBUG_TOKENS {
        println!("### Tokens:");
        let mut lex = Lexer::new(code);
        while let Some(t) = lex.next() {
            println!("{:?}", t)
        }
        println!();
    }

    if DEBUG_AST {
        println!("### AST Nodes:");
        if let Err(()) = StmtIter::parse_and_debug(code) {
            panic!("Parsing Error")
        }
        println!();
    }

    let stmts = match StmtIter::try_parse_all(code, &alloc) {
        Ok(stmts) => stmts,
        Err(errs) => {
            println!("### Parsing Errors:");
            for e in errs {
                display_spanned_error(&e, code);
            }
            panic!("Parsing Failed")
        },
    };

    let sema = Sema::<DEBUG_TYPES>::new(code, &alloc);
    let context = inkwell::context::Context::create();
    let codegen = llvm::Codegen::new_module(&context, "test", &alloc);
    let mut compiler = Compiler::new(sema, codegen);

    let _ = compiler.compile_stmts(&stmts);

    match compiler.sema.errors.len() {
        0 => {},
        1 => return Err(Error::Sema(compiler.sema.errors.into_iter().next().unwrap())),
        _ => {
            eprintln!("### Sema Errors:");
            for e in compiler.sema.errors {
                display_spanned_error(&e, code);
            }
            panic!("multiple sema errors")
        },
    }

    let target_machine = llvm::Codegen::init_target_machine();
    compiler.optimize(&target_machine, LLVM_OPTIMIZATION_LEVEL)?;

    println!("{}", compiler.codegen.module.print_to_string().to_string());

    Ok(compiler.codegen.jit_run_fn::<RetTy>("test", inkwell::OptimizationLevel::None)?)
}
