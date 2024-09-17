use crate::{
    ast::Expr,
    codegen::llvm,
    parser::{lexer::Code, DebugAst},
    ptr::Ptr,
    sema::{Sema, SemaError},
    util::display_spanned_error,
};
use std::time::{Duration, Instant};

pub struct Compiler<'c, 'ctx> {
    pub sema: Sema<'c>,
    pub codegen: llvm::Codegen<'ctx>,
}

impl<'c, 'ctx> Compiler<'c, 'ctx> {
    pub fn new(sema: Sema<'c>, codegen: llvm::Codegen<'ctx>) -> Compiler<'c, 'ctx> {
        Compiler { sema, codegen }
    }

    pub fn compile_stmts(&mut self, stmts: &[Ptr<Expr>]) {
        for s in stmts.iter().copied() {
            self.sema.preload_top_level(s);
        }

        let mut finished = vec![false; stmts.len()];
        let mut remaining_count = stmts.len();
        while finished.iter().any(std::ops::Not::not) {
            let old_remaining_count = remaining_count;
            debug_assert!(stmts.len() == finished.len());
            remaining_count = 0;
            for (&s, finished) in stmts.iter().zip(finished.iter_mut()) {
                if *finished {
                    continue;
                }
                *finished = self.sema.analyze_top_level(s);
                if *finished {
                    // println!("{:#?}", s );
                    self.codegen.compile_top_level(s);
                } else {
                    remaining_count += 1;
                }
            }
            // println!("finished statements: {:?}", finished);
            if remaining_count == old_remaining_count {
                panic!("cycle detected") // TODO: find location of cycle
            }
        }
    }

    pub fn compile_stmts_dev<const DEBUG_TYPED_AST: bool>(
        &mut self,
        stmts: &[Ptr<Expr>],
        code: &Code,
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
                *finished = self.sema.analyze_top_level(s);
                if *finished {
                    order.push(idx);
                } else {
                    remaining_count += 1;
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
            panic!("Semantic analysis ERROR")
        }

        let sema_duration = sema_start.elapsed();

        if DEBUG_TYPED_AST {
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
}
