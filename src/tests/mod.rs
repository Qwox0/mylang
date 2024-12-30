use crate::{
    codegen::llvm,
    error::Error,
    parser::{StmtIter, lexer::Lexer, parser_helper::ParserInterface},
    sema::Sema,
    util::display_spanned_error,
};

mod alignment;
mod binop;
mod call_conv_c;
mod defer;
mod enum_;
mod for_loop;
mod function;
mod if_;
mod initializer;
mod logic_binop;
mod parse_array;
mod parse_function;
mod ptr;
mod slice;
mod string;
mod struct_;
mod todo;
mod union_;
mod while_loop;

macro_rules! jit_run_test {
    (raw $code:expr => $ret_type:ty,llvm_module) => {
        $crate::tests::jit_run_test_impl::<$ret_type>($code.as_ref())
    };
    ($code:expr => $ret_type:ty,llvm_module) => {{
        let code = format!("test :: -> {{ {} }};", $code);
        $crate::tests::jit_run_test_impl::<$ret_type>(code.as_ref())
    }};
    (raw $code:expr => $ret_type:ty) => {
        $crate::tests::jit_run_test_impl::<$ret_type>($code.as_ref()).map(|(out, _)| out)
    };
    ($code:expr => $ret_type:ty) => {{
        let code = format!("test :: -> {{ {} }};", $code);
        $crate::tests::jit_run_test_impl::<$ret_type>(code.as_ref()).map(|(out, _)| out)
    }};
}
pub(crate) use jit_run_test;

const DEBUG_TOKENS: bool = false;
const DEBUG_AST: bool = true;
const DEBUG_TYPES: bool = true;
const LLVM_OPTIMIZATION_LEVEL: u8 = 0;
const PRINT_ERROR: bool = true;
const PRINT_LLVM_MODULE: bool = true;

pub fn jit_run_test_impl<RetTy>(
    code: &crate::parser::lexer::Code,
) -> Result<(RetTy, String), Error>
where [(); std::mem::size_of::<RetTy>()]: Sized {
    let context = inkwell::context::Context::create();
    let res = compile_test(&context, code).and_then(|codegen| {
        let llvm_module_text = codegen.module.print_to_string().to_string();
        if PRINT_LLVM_MODULE {
            println!("{}", llvm_module_text);
        }
        let out = codegen.jit_run_fn::<RetTy>("test", inkwell::OptimizationLevel::None)?;
        Ok((out, llvm_module_text))
    });
    if PRINT_ERROR && let Err(err) = res.as_ref() {
        display_spanned_error(err, code);
    }
    res
}

pub fn compile_test<'ctx, RetTy>(
    context: &'ctx inkwell::context::Context,
    code: &crate::parser::lexer::Code,
) -> Result<llvm::Codegen<'ctx>, Error>
where
    [(); std::mem::size_of::<RetTy>()]: Sized,
{
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

    let mut sema = Sema::new(code, &alloc, DEBUG_TYPES);
    let order = sema.analyze_all(&stmts);

    match sema.errors.len() {
        0 => {},
        1 => return Err(Error::Sema(sema.errors.into_iter().next().unwrap())),
        _ => {
            eprintln!("### Sema Errors:");
            for e in sema.errors {
                display_spanned_error(&e, code);
            }
            panic!("multiple sema errors")
        },
    }

    let mut codegen = llvm::Codegen::new_module(&context, "dev");
    codegen.compile_all(&stmts, &order);

    let target_machine = llvm::Codegen::init_target_machine(None);
    codegen.optimize_module(&target_machine, LLVM_OPTIMIZATION_LEVEL)?;

    Ok(codegen)
}
