use crate::{
    ast::{Expr, debug::DebugAst},
    codegen::llvm::{self, CodegenError},
    parser::lexer::Code,
    ptr::Ptr,
    sema::{Sema, SemaError, SemaResult},
    util::display_spanned_error,
};
use std::time::{Duration, Instant};

pub struct Compiler<'c, 'ctx, 'alloc> {
    pub sema: Sema<'c, 'alloc>,
    pub codegen: llvm::Codegen<'ctx>,
}

impl<'c, 'ctx, 'alloc> Compiler<'c, 'ctx, 'alloc> {
    pub fn new(sema: Sema<'c, 'alloc>, codegen: llvm::Codegen<'ctx>) -> Compiler<'c, 'ctx, 'alloc> {
        Compiler { sema, codegen }
    }

    pub fn compile_stmts(&mut self, stmts: &[Ptr<Expr>]) -> Result<(), ()> {
        for s in stmts.iter().copied() {
            self.sema.preload_top_level(s);
        }

        let mut finished = vec![false; stmts.len()];
        let mut remaining_count = stmts.len();
        let mut order = Vec::with_capacity(stmts.len());
        while finished.iter().any(std::ops::Not::not) {
            let old_remaining_count = remaining_count;
            debug_assert!(stmts.len() == finished.len());
            remaining_count = 0;
            for (idx, (&s, finished)) in stmts.iter().zip(finished.iter_mut()).enumerate() {
                if *finished {
                    continue;
                }
                let res = self.sema.analyze_top_level(s);
                *finished = res != SemaResult::NotFinished;
                match res {
                    SemaResult::Ok(_) => order.push(idx),
                    SemaResult::NotFinished => remaining_count += 1,
                    SemaResult::Err(_) => {},
                }
            }
            // println!("finished statements: {:?}", finished);
            if remaining_count == old_remaining_count {
                panic!("cycle detected") // TODO: find location of cycle
            }
        }

        if !self.sema.errors.is_empty() {
            return Err(());
        }

        for idx in order {
            let s = stmts[idx];
            self.codegen.compile_top_level(s);
        }

        Ok(())
    }

    pub fn compile_stmts_dev(
        &mut self,
        stmts: &[Ptr<Expr>],
        code: &Code,
        debug_typed_ast: bool,
    ) -> (Duration, Duration) {
        let sema_start = Instant::now();
        for s in stmts.iter().copied() {
            self.sema.preload_top_level(s);
        }

        let mut finished = vec![false; stmts.len()];
        let mut remaining_count = stmts.len();
        let mut order = Vec::with_capacity(stmts.len());
        while finished.iter().any(std::ops::Not::not) {
            let old_remaining_count = remaining_count;
            debug_assert!(stmts.len() == finished.len());
            remaining_count = 0;
            for (idx, (&s, finished)) in stmts.iter().zip(finished.iter_mut()).enumerate() {
                if *finished {
                    continue;
                }
                let res = self.sema.analyze_top_level(s);
                *finished = res != SemaResult::NotFinished;
                match res {
                    SemaResult::Ok(_) => order.push(idx),
                    SemaResult::NotFinished => remaining_count += 1,
                    SemaResult::Err(_) => {},
                }
            }
            // println!("finished statements: {:?}", finished);
            if remaining_count == old_remaining_count {
                panic!("cycle detected") // TODO: find location of cycle
            }
        }

        if !self.sema.errors.is_empty() {
            for e in self.sema.errors.iter() {
                display_spanned_error(e, code);
            }
            std::process::exit(1);
        }

        let sema_duration = sema_start.elapsed();

        #[cfg(debug_assertions)]
        if debug_typed_ast {
            println!("\n### Typed AST Nodes:");
            for s in stmts.iter().copied() {
                println!("stmt @ {:?}", s);
                s.print_tree();
            }
            println!();
        }

        let codegen_start = Instant::now();

        for idx in order {
            let s = stmts[idx];
            self.codegen.compile_top_level(s);
        }

        (sema_duration, codegen_start.elapsed())
    }

    pub fn get_sema_errors(&self) -> &[SemaError] {
        &self.sema.errors
    }

    pub fn optimize(
        &self,
        target_machine: &llvm::TargetMachine,
        level: u8,
    ) -> Result<(), CodegenError> {
        self.codegen.optimize_module(target_machine, level)
    }
}
