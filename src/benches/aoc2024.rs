extern crate test;

use super::bench_compilation;
use crate::{
    codegen::llvm, compiler::CompileMode, context::CompilationContext,
};
use inkwell::context::Context;
use test::*;

bench_compilation! { aoc2024_day01_1parse: "../../../../aoc2024/day01.mylang", CompileMode::Parse }
bench_compilation! { aoc2024_day01_2sema: "../../../../aoc2024/day01.mylang", CompileMode::Check } // parse + sema
bench_compilation! { aoc2024_day01_3codegen: "../../../../aoc2024/day01.mylang", CompileMode::Codegen } // parse + sema + codegen
bench_compilation! { aoc2024_day01_3codegen_only: "../../../../aoc2024/day01.mylang", codegen_only } // codegen
bench_compilation! { aoc2024_day01_4build: "../../../../aoc2024/day01.mylang", CompileMode::Build } // parse + sema + codegen + llvm passes

bench_compilation! { aoc2024_day02_2sema: "../../../../aoc2024/day02.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day02_3codegen_only: "../../../../aoc2024/day02.mylang", codegen_only }
bench_compilation! { aoc2024_day03_2sema: "../../../../aoc2024/day03.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day03_3codegen_only: "../../../../aoc2024/day03.mylang", codegen_only }
bench_compilation! { aoc2024_day04_2sema: "../../../../aoc2024/day04.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day04_3codegen_only: "../../../../aoc2024/day04.mylang", codegen_only }
bench_compilation! { aoc2024_day05_2sema: "../../../../aoc2024/day05.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day05_3codegen_only: "../../../../aoc2024/day05.mylang", codegen_only }
bench_compilation! { aoc2024_day06_2sema: "../../../../aoc2024/day06.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day06_3codegen_only: "../../../../aoc2024/day06.mylang", codegen_only }
bench_compilation! { aoc2024_day07_2sema: "../../../../aoc2024/day07.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day07_3codegen_only: "../../../../aoc2024/day07.mylang", codegen_only }
bench_compilation! { aoc2024_day08_2sema: "../../../../aoc2024/day08.mylang", CompileMode::Check }
bench_compilation! { aoc2024_day08_3codegen_only: "../../../../aoc2024/day08.mylang", codegen_only }
