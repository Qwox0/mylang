extern crate test;

use crate::{
    cli::BuildArgs, context::CompilationContext, diagnostics::DiagnosticReporter, parser, ptr::Ptr,
};
use test::*;

#[inline]
fn bench_parse(code: &str) {
    let ctx = CompilationContext::for_tests(BuildArgs::comp_bench_args(), Ptr::from_ref(code), false);
    parser::parse_files(ctx.0);
    assert!(!ctx.do_abort_compilation());
    assert!(ctx.diagnostic_reporter.diagnostics.is_empty());
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
main :: -> false |> if test(1) else (10 |> sub(3));
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
