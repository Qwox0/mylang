#![feature(try_trait_v2)]
#![feature(never_type)]
#![feature(iter_intersperse)]
#![feature(control_flow_enum)]
#![feature(type_changing_struct_update)]
#![feature(new_range_api)]
#![feature(ptr_as_ref_unchecked)]
#![feature(unsigned_is_multiple_of)]
#![feature(let_chains)]
#![allow(unreachable_code)]

pub mod ast;
pub mod cli;
pub mod codegen;
pub mod error;
pub mod parser;
pub mod ptr;
mod scratch_pool;
pub mod sema;
pub mod symbol_table;
pub mod util;
