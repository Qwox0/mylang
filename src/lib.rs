// Cannot use Rust nightly >= 1.92.0, because of <https://github.com/rust-lang/rust/pull/145838>

#![cfg_attr(not(test), allow(dead_code))]
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
#![feature(vec_push_within_capacity)]

mod arena_allocator;
mod ast;
mod buffer;
pub mod cli;
mod codegen;
pub mod compiler;
mod context;
mod diagnostics;
mod display_code;
mod intern_pool;
mod literals;
mod parser;
mod ptr;
mod scope;
mod scoped_stack;
mod scratch_pool;
mod sema;
mod source_file;
mod type_;
mod util;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;
