//! ```
//! test benches::parser::bench_parse1                 ... bench:       1,593.98 ns/iter (+/- 176.31)
//! test benches::parser::bench_parse2                 ... bench:       2,182.07 ns/iter (+/- 167.26)
//! test benches::parser::bench_parse3                 ... bench:       2,649.51 ns/iter (+/- 129.28)
//! test benches::parser::bench_parse4                 ... bench:       3,291.12 ns/iter (+/- 326.39)
//! ```

extern crate test;

use crate::{
    cli::BuildArgs,
    compiler::{CompileMode, compile_file},
    context::CompilationContext,
    tests::test_file_mock,
};
use std::path::PathBuf;
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
        let code = code.as_ref();
        let ctx = CompilationContext::new();
        compile_file(ctx.0, test_file_mock(code), CompileMode::Check, &BuildArgs::bench_args());
    })
}

bench_compilation! {
    compile_libc_example:
        no_prelude "../../examples/libc/basic/main.mylang",
        crate::compiler::CompileMode::Build
}

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
            bench_compilation! { @code@ b; $($rem)* }
        }
    };
    (@code@ $b:expr; no_prelude $code_file_path:expr, $($rem:tt)*) => {
        let code = include_str!($code_file_path);
        bench_compilation! { @body@ $b;code; $($rem)* }
    };
    (@code@ $b:expr; $code_file_path:expr, $($rem:tt)*) => {
        let code = concat!(include_str!("../../lib/prelude.mylang"), "\n", include_str!($code_file_path));
        bench_compilation! { @body@ $b; code; $($rem)* }
    };
    (@body@ $b:expr; $code:expr; codegen_only) => {
        let code = $code.as_ref();
        let ctx = CompilationContext::new();
        let test_file = $crate::tests::test_file_mock(code);
        let stmts = $crate::parser::parse(ctx.0, test_file, true);
        assert!(!ctx.do_abort_compilation());

        let order = $crate::sema::analyze(ctx.0, &stmts);
        assert!(!ctx.do_abort_compilation());

        $b.iter(|| {
            let context = Context::create();
            let mut codegen = llvm::Codegen::new(&context, "dev");
            codegen.compile_all(&stmts, &order);
        });
    };
    (@body@ $b:expr; $code:expr; $mode:expr) => {
        let code = $code.as_ref();
        let test_file = $crate::tests::test_file_mock(code);
        $b.iter(|| {
            let ctx = CompilationContext::new();
            black_box($crate::compiler::compile_file(ctx.0, black_box(test_file), $mode, &crate::cli::BuildArgs {
                path: PathBuf::new(),
                optimization_level: 0,
                target_triple: None,
                out: crate::cli::OutKind::None,
                no_prelude: true,
                print_compile_time: false,
                debug_ast: false,
                debug_types: false,
                debug_typed_ast: false,
                debug_functions: false,
                debug_llvm_ir_unoptimized: false,
                debug_llvm_ir_optimized: false,
            }));
        });
    };
}
pub(crate) use bench_compilation;
