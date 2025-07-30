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
        let ctx = CompilationContext::new(BuildArgs::comp_bench_args());
        ctx.0.set_test_root(Ptr::from_ref(code.as_ref())).unwrap();
        compile_ctx(ctx.0, CompileMode::Check);
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
        use crate::diagnostics::DiagnosticReporter;
        let code = $code.as_ref();
        let code = $crate::ptr::Ptr::from_ref(code);
        let ctx = CompilationContext::new($crate::cli::BuildArgs::comp_bench_args());
        ctx.0.set_test_root(code).unwrap();
        let stmts = $crate::parser::parse_files_in_ctx(ctx.0);
        assert!(!ctx.do_abort_compilation());

        $crate::sema::analyze(ctx.0, &stmts);
        assert!(!ctx.do_abort_compilation());

        $b.iter(|| {
            let context = Context::create();
            let mut codegen = llvm::Codegen::new(&context, "dev");
            codegen.compile_all(ctx.0, &stmts);
        });
    };
    (@body@ $b:expr; $code:expr; $mode:expr) => {
        let code = $crate::ptr::Ptr::from_ref($code.as_ref());
        $b.iter(|| {
            let ctx = CompilationContext::new($crate::cli::BuildArgs::comp_bench_args());
            ctx.0.set_test_root(code).unwrap();
            black_box($crate::compiler::compile_ctx(ctx.0, $mode));
        });
    };
}
pub(crate) use bench_compilation;
