#![feature(test)]
#![feature(try_trait_v2)]
#![feature(never_type)]
#![feature(iter_intersperse)]
#![feature(new_range_api)]
#![feature(unsigned_is_multiple_of)]
#![feature(let_chains)]
#![feature(try_blocks)]
#![feature(assert_matches)]
#![allow(unreachable_code)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(iter_collect_into)]
#![feature(path_add_extension)]

pub mod ast;
pub mod cli;
pub mod codegen;
pub mod compiler;
pub mod defer_stack;
pub mod error;
pub mod parser;
pub mod ptr;
mod scratch_pool;
pub mod sema;
pub mod symbol_table;
pub mod type_;
pub mod util;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;
