extern crate test;

use crate::{
    cli::BuildArgs,
    compiler::{CompileMode, compile_ctx},
    context::CompilationContext,
    ptr::Ptr,
};
use test::*;

mod aoc2024;
mod parser;

#[bench]
fn bench_frontend1(b: &mut Bencher) {
    b.iter(|| {
        let code = "
pub sub :: (a: f64, b: f64, ) -> -b + a;

Sub :: struct {
    a: f64,
    b: f64,
}
pub sub2 :: (values: Sub) -> values.a - values.b;

mymain :: -> {
    mut a := test(1);
    mut a := 10.0;
    a = a == 0 |> if test(1) else (10.sub(1)) |> sub(3);
    b := test();

    if defer_test() return 10000.0;

    return sub2(Sub.{ a = a, b });
    // this is unreachable but is checked anyway
    x: f64 = 1 + 1;
    x
};
pub test :: (mut x := 1.0) -> {
    x += 1;
    420;
    1+2*x
};

pub defer_test :: -> {
    mut out := 1;
    {
        defer out *= 10;
        out += 1;
    }
    t1 := out == 20;
    out = 1;
    {
        defer out += 1;
        defer out *= 10;
    }
    t2 := out == 11;
    return t1 && t2;
};";
        let ctx = CompilationContext::for_tests(
            BuildArgs::comp_bench_args(None),
            Ptr::from_ref(code),
            false,
        );
        compile_ctx(ctx.0, CompileMode::Check);
    })
}

bench_compilation! { parse_std: "./lib/std/mod.mylang", CompileMode::Parse }
bench_compilation! { check_std: "./lib/std/mod.mylang", CompileMode::Check }

/// ```rust
/// bench_compilation!(bench1: no_prelude "file.mylang", CompileMode::Build)
/// bench_compilation!(bench2: "file.mylang", CompileMode::Build)
/// bench_compilation!(bench3: no_prelude "file.mylang", codegen_only)
/// bench_compilation!(bench4: "file.mylang", codegen_only)
/// ```
macro_rules! bench_compilation {
    ($test_name:ident : $($rem:tt)*) => {
        #[bench]
        fn $test_name(b: &mut Bencher) {
            bench_compilation! { @body@ b; $($rem)* }
        }
    };
    (@body@ $b:expr; $code_file_path:expr, codegen_only) => {
        use crate::diagnostics::DiagnosticReporter;
        let ctx = CompilationContext::basic($crate::cli::BuildArgs::comp_bench_args(Some($code_file_path))).unwrap();
        let mut stmts = $crate::parser::parse_files(ctx.0);
        assert!(!ctx.do_abort_compilation());

        $crate::sema::analyze(ctx.0, &mut stmts);
        assert!(!ctx.do_abort_compilation());

        $b.iter(|| {
            let context = Context::create();
            let mut codegen = llvm::Codegen::new(&context, "dev");
            codegen.compile_all(&stmts).unwrap();
        });
    };
    (@body@ $b:expr; $code_file_path:expr, $mode:expr) => {
        use crate::diagnostics::DiagnosticReporter;
        let args = $crate::cli::BuildArgs::comp_bench_args(Some($code_file_path));
        $b.iter(|| {
            let ctx = CompilationContext::basic(args.clone()).unwrap();
            black_box($crate::compiler::compile_ctx(ctx.0, $mode));
            debug_assert!(!ctx.do_abort_compilation());
        });
    };
}
pub(crate) use bench_compilation;
