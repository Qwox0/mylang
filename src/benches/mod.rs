//! old (clone in Lexer::peek)
//! ```
//! test benches::bench_parse  ... bench:       2,898.31 ns/iter (+/- 229.93)
//! test benches::bench_parse2 ... bench:       4,115.37 ns/iter (+/- 397.96)
//! test benches::bench_parse3 ... bench:       4,569.32 ns/iter (+/- 537.69)
//! test benches::bench_parse4 ... bench:       6,576.30 ns/iter (+/- 638.84)
//! ```
//!
//! next_tok field in Lexer
//! ```
//! test benches::bench_parse  ... bench:       1,880.97 ns/iter (+/- 386.20)
//! test benches::bench_parse2 ... bench:       2,780.48 ns/iter (+/- 851.97)
//! test benches::bench_parse3 ... bench:       2,908.87 ns/iter (+/- 408.08)
//! test benches::bench_parse4 ... bench:       4,063.92 ns/iter (+/- 567.65)
//! ```

extern crate test;

use crate::{
    codegen::llvm,
    compiler::Compiler,
    parser::StmtIter,
    sema,
    util::{collect_all_result_errors, display_spanned_error},
};
use inkwell::context::Context;
use std::path::PathBuf;
use test::*;

#[inline]
fn bench_parse(code: &str) {
    let alloc = bumpalo::Bump::new();
    let code = code.as_ref();
    let mut stmts = StmtIter::parse(code, &alloc);
    while let Some(res) = black_box(StmtIter::next(black_box(&mut stmts))) {
        res.unwrap();
    }
}

/// old: 11ms  <- so bad
/// new: 3000ns  (3667x faster)
#[bench]
fn bench_parse1(b: &mut Bencher) {
    let code = "
test :: x -> 1+2+x;
main :: -> test(1) + test(2);
//main :: -> if true test(1) else test(2);
/*
main :: -> {
    a := test(1);
    b := test(2);
    a + b
};
*/
";
    b.iter(|| bench_parse(code));
}

/// old: 23ms  <- so bad
/// new: 3900ns  (5897x faster)
#[bench]
fn bench_parse2(b: &mut Bencher) {
    let code = "
test :: x -> {
    a := (x + 3 * 2) * x + 1;
    b := x * 2 * x;
    a + b
};
main :: -> test(10);
";
    b.iter(|| bench_parse(code));
}

#[bench]
fn bench_parse3(b: &mut Bencher) {
    let code = "
pub test :: x -> 1+2*x;
pub sub :: (a, b) -> -b + a;
main :: -> false | if test(1) else (10 | sub(3));
";
    b.iter(|| bench_parse(code));
}

#[bench]
fn bench_parse4(b: &mut Bencher) {
    let code = "
pub test :: (x := 2) -> 1+2*x;
pub sub :: (a, mut b) -> -b + a;
mymain :: -> {
    mut a := test(1);
    mut a := 10;
    a = 100;
    b := test(2);
    a + b
};
";
    b.iter(|| bench_parse(code));
}

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
    a = a == 0 | if test(1) else (10 | sub(1)) | sub(3);
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
    }; // TODO: no `;` here
    t1 := out == 20;
    out = 1;
    {
        defer out += 1;
        defer out *= 10;
    };
    t2 := out == 11;
    return t1 && t2;
};";
        let code = code.as_ref();
        let alloc = bumpalo::Bump::new();
        let stmts = StmtIter::parse(code, &alloc);

        let stmts = collect_all_result_errors(stmts).unwrap_or_else(|errors| {
            for e in errors {
                display_spanned_error(&e, code);
            }
            panic!("Parse ERROR")
        });

        let sema = sema::Sema::new(code, &alloc, false);
        let context = Context::create();
        let codegen = llvm::Codegen::new_module(&context, "dev");
        let mut compiler = Compiler::new(sema, codegen);

        let _ = compiler.compile_stmts(&stmts);

        if !compiler.sema.errors.is_empty() {
            for e in compiler.sema.errors {
                display_spanned_error(&e, code);
            }
            panic!("Semantic analysis ERROR")
        }
    })
}

#[bench]
#[ignore = "unfinished test"]
fn compile_libc_example(b: &mut Bencher) {
    b.iter(|| {
        crate::compiler::compile2(crate::compiler::CompileMode::Build, &crate::cli::BuildArgs {
            path: PathBuf::from("./examples/libc/main.mylang"),
            optimization_level: 0,
            target_triple: None,
            out: crate::cli::OutKind::None,
            no_prelude: true,
            debug_tokens: false,
            debug_ast: false,
            debug_types: false,
            debug_typed_ast: false,
            debug_functions: false,
            debug_llvm_ir_unoptimized: false,
            debug_llvm_ir_optimized: false,
        });
    });
}
