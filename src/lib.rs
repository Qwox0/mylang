#![feature(test)]
#![feature(try_trait_v2)]
#![feature(never_type)]
#![feature(iter_intersperse)]
#![feature(new_range_api)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(try_blocks)]
#![feature(assert_matches)]
#![allow(unreachable_code)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(iter_collect_into)]
#![feature(path_add_extension)]
#![feature(exit_status_error)]
#![feature(non_null_from_ref)]
#![feature(ptr_as_ref_unchecked)]
#![feature(thread_local)]
#![feature(formatting_options)]

pub mod arena_allocator;
pub mod ast;
pub mod cli;
pub mod codegen;
pub mod compiler;
pub mod context;
pub mod diagnostic_reporter;
pub mod display_code;
pub mod error;
pub mod literals;
pub mod parser;
pub mod ptr;
pub mod scoped_stack;
mod scratch_pool;
pub mod sema;
pub mod source_file;
pub mod symbol_table;
pub mod type_;
pub mod util;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;
