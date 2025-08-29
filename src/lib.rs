#![feature(test)]
#![feature(try_trait_v2)]
#![feature(never_type)]
#![feature(iter_intersperse)]
#![feature(new_range_api)]
#![feature(if_let_guard)]
#![feature(try_blocks)]
#![feature(assert_matches)]
#![feature(path_add_extension)]
#![feature(exit_status_error)]
#![feature(ptr_as_ref_unchecked)]
#![feature(thread_local)]
#![feature(formatting_options)]
#![feature(maybe_uninit_slice)]
#![feature(arbitrary_self_types)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_write_slice)]
#![feature(type_changing_struct_update)]

pub mod arena_allocator;
pub mod ast;
pub mod cli;
pub mod codegen;
pub mod compiler;
pub mod context;
pub mod diagnostics;
pub mod display_code;
pub mod error;
mod intern_pool;
pub mod literals;
pub mod parser;
pub mod ptr;
mod scope;
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
