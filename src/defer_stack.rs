use crate::{ast::Expr, ptr::Ptr, util::UnwrapDebug};
use core::fmt;
use std::mem;

#[derive(Debug, Default)]
pub struct DeferStack {
    values: Vec<DeferStackValue>,
    cur_len: usize,
}

impl DeferStack {
    pub fn open_scope(&mut self) {
        self.values.push(DeferStackValue { prev_len: self.cur_len });
        self.cur_len = 0;
    }

    pub fn close_scope(&mut self) {
        unsafe { self.values.set_len(self.values.len() - self.cur_len) };
        self.cur_len = unsafe { self.values.pop().unwrap_debug().prev_len }; // unwrap: closed when no scope was open
    }

    pub fn push_expr(&mut self, expr_ptr: Ptr<Expr>) {
        self.values.push(DeferStackValue { expr_ptr });
        self.cur_len += 1;
    }

    pub fn get_cur_scope(&self) -> &[Ptr<Expr>] {
        unsafe { mem::transmute(&self.values[self.values.len() - self.cur_len..]) }
    }
}

union DeferStackValue {
    expr_ptr: Ptr<Expr>,
    prev_len: usize,
}

impl fmt::Debug for DeferStackValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#x}", unsafe { self.prev_len })
    }
}

#[cfg(test)]
mod benches {
    extern crate test;
    use super::*;
    use test::*;

    /// `DeferStack` using `Vec<Vec<Ptr<Expr>>>`:
    /// ```
    /// test defer_stack::benches::bench_defer_stack ... bench:       2,132.09 ns/iter (+/- 374.07)
    /// ```
    ///
    /// `DeferStack` using `Vec<DeferStackValue>`:
    /// ```
    /// test defer_stack::benches::bench_defer_stack ... bench:         465.60 ns/iter (+/- 68.57)
    /// ```
    #[bench]
    fn bench_defer_stack(b: &mut Bencher) {
        let n = "".into();
        let e = crate::ast::Expr::new(
            crate::ast::ExprKind::Ident(n),
            crate::parser::lexer::Span::new(0, 0),
        );
        let p = Ptr::from(&e);
        b.iter(|| {
            let mut defer_stack = DeferStack::default();
            for _ in 0..10 {
                defer_stack.open_scope();
                for _ in 0..10 {
                    defer_stack.push_expr(p);
                }
            }
            for _ in 0..10 {
                for _ in 0..10 {
                    defer_stack.push_expr(p);
                }
                defer_stack.close_scope();
            }
        })
    }
}
