extern crate test;

use super::bench_compilation;
use crate::{
    codegen::llvm,
    compiler::{CompileMode, compile, parse},
    sema,
};
use inkwell::context::Context;
use std::path::PathBuf;
use test::*;

// test benches::aoc2024::aoc2024_day01_1parse        ... bench:      85,420.30 ns/iter (+/- 3,077.73)
// test benches::aoc2024::aoc2024_day01_2sema         ... bench:     120,548.58 ns/iter (+/- 7,971.38)
// test benches::aoc2024::aoc2024_day01_3codegen      ... bench:     546,167.50 ns/iter (+/- 33,548.42)
// test benches::aoc2024::aoc2024_day01_3codegen_only ... bench:     404,325.22 ns/iter (+/- 36,847.22)
// test benches::aoc2024::aoc2024_day01_4build        ... bench:     674,020.79 ns/iter (+/- 60,503.05)
//
// test benches::aoc2024::aoc2024_day02_2sema         ... bench:     132,859.66 ns/iter (+/- 8,238.47)
// test benches::aoc2024::aoc2024_day02_3codegen_only ... bench:     492,613.70 ns/iter (+/- 23,070.20)
// test benches::aoc2024::aoc2024_day03_2sema         ... bench:     150,378.88 ns/iter (+/- 14,213.83)
// test benches::aoc2024::aoc2024_day03_3codegen_only ... bench:     535,681.41 ns/iter (+/- 15,414.45)
// test benches::aoc2024::aoc2024_day04_2sema         ... bench:     124,310.08 ns/iter (+/- 53,777.82)
// test benches::aoc2024::aoc2024_day04_3codegen_only ... bench:     433,634.70 ns/iter (+/- 33,858.12)
// test benches::aoc2024::aoc2024_day05_2sema         ... bench:     170,964.08 ns/iter (+/- 7,431.85)
// test benches::aoc2024::aoc2024_day05_3codegen_only ... bench:     731,735.40 ns/iter (+/- 2,453.26)
// test benches::aoc2024::aoc2024_day06_2sema         ... bench:     240,012.95 ns/iter (+/- 2,438.12)
// test benches::aoc2024::aoc2024_day06_3codegen_only ... bench:   1,052,486.90 ns/iter (+/- 47,273.48)
// test benches::aoc2024::aoc2024_day07_2sema         ... bench:     151,257.50 ns/iter (+/- 6,375.68)
// test benches::aoc2024::aoc2024_day07_3codegen_only ... bench:     594,622.40 ns/iter (+/- 63,771.25)
// test benches::aoc2024::aoc2024_day08_2sema         ... bench:     192,948.08 ns/iter (+/- 10,596.19)
// test benches::aoc2024::aoc2024_day08_3codegen_only ... bench:     795,067.60 ns/iter (+/- 48,674.45)

// Compiling file at "../aoc2024/day01.mylang"
//   Frontend:              886.806µs
//     Lexer, Parser:         138.69µs
//     Semantic Analysis:     90.62µs
//     LLVM IR Codegen:       657.496µs
//   Backend:               18.580226ms
//     LLVM Setup:            794.935µs
//     LLVM pass pipeline:    121.178µs
//     writing obj file:      17.664113ms
//   Linking with gcc:      21.314106ms
//   Total:                 40.781138ms

bench_compilation! { aoc2024_day01_1parse: "../../../aoc2024/day01.mylang", CompileMode::Parse }
bench_compilation! { aoc2024_day01_2sema: "../../../aoc2024/day01.mylang", CompileMode::Check } // parse + sema
bench_compilation! { aoc2024_day01_3codegen: "../../../aoc2024/day01.mylang", CompileMode::Codegen } // parse + sema + codegen
bench_compilation! { aoc2024_day01_3codegen_only: "../../../aoc2024/day01.mylang", codegen_only } // codegen
bench_compilation! { aoc2024_day01_4build: "../../../aoc2024/day01.mylang", CompileMode::Build } // parse + sema + codegen + llvm passes

bench_compilation! { aoc2024_day02_2sema: "../../../aoc2024/day02.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day02_3codegen_only: "../../../aoc2024/day02.mylang", codegen_only }
bench_compilation! { aoc2024_day03_2sema: "../../../aoc2024/day03.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day03_3codegen_only: "../../../aoc2024/day03.mylang", codegen_only }
bench_compilation! { aoc2024_day04_2sema: "../../../aoc2024/day04.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day04_3codegen_only: "../../../aoc2024/day04.mylang", codegen_only }
bench_compilation! { aoc2024_day05_2sema: "../../../aoc2024/day05.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day05_3codegen_only: "../../../aoc2024/day05.mylang", codegen_only }
bench_compilation! { aoc2024_day06_2sema: "../../../aoc2024/day06.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day06_3codegen_only: "../../../aoc2024/day06.mylang", codegen_only }
bench_compilation! { aoc2024_day07_2sema: "../../../aoc2024/day07.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day07_3codegen_only: "../../../aoc2024/day07.mylang", codegen_only }
bench_compilation! { aoc2024_day08_2sema: "../../../aoc2024/day08.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day08_3codegen_only: "../../../aoc2024/day08.mylang", codegen_only }
