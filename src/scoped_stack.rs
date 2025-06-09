use crate::util::UnwrapDebug;
use core::fmt;
use std::{
    iter::FusedIterator,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::Range,
};

#[derive(Debug)]
pub struct ScopedStack<T> {
    values: Vec<ScopedStackValue<T>>,
    cur_len: usize,
}

impl<T> Default for ScopedStack<T> {
    fn default() -> Self {
        // this starting value is required for `ScopedStackScopeIter`.
        Self { values: vec![ScopedStackValue { prev_len: 0 }], cur_len: 0 }
    }
}

impl<T> ScopedStack<T> {
    pub fn open_scope(&mut self) {
        self.values.push(ScopedStackValue { prev_len: self.cur_len });
        self.cur_len = 0;
    }

    pub fn close_scope(&mut self) {
        unsafe { self.values.set_len(self.values.len() - self.cur_len) };
        self.cur_len = unsafe { self.values.pop().u().prev_len }; // unwrap: closed when no scope was open
    }

    pub fn push(&mut self, val: T) {
        self.values.push(ScopedStackValue { val: ManuallyDrop::new(val) });
        self.cur_len += 1;
    }

    pub fn get_cur_scope(&self) -> &[T] {
        let start = self.values.len() - self.cur_len;
        unsafe { mem::transmute::<&[ScopedStackValue<T>], &[T]>(&self.values[start..]) }
    }

    pub fn get_cur_scope_mut(&mut self) -> &mut [T] {
        let start = self.values.len() - self.cur_len;
        unsafe { mem::transmute::<&mut [ScopedStackValue<T>], &mut [T]>(&mut self.values[start..]) }
    }

    /// Iterates over the scopes of the stack in pop order.
    #[inline]
    pub fn iter_scopes(&self) -> ScopedStackScopeIter<'_, T> {
        let Range { start, end } = self.values.as_ptr_range();
        ScopedStackScopeIter { start, end, cur_len: self.cur_len, _lifetime: PhantomData }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    pub fn is_empty(&self) -> bool {
        self.values.len() == 1
    }
}

union ScopedStackValue<T> {
    val: ManuallyDrop<T>,
    prev_len: usize,
}

impl<T: std::fmt::Debug> fmt::Debug for ScopedStackValue<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#x}", unsafe { self.prev_len })
    }
}

pub struct ScopedStackScopeIter<'a, T> {
    start: *const ScopedStackValue<T>,
    end: *const ScopedStackValue<T>,
    cur_len: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Iterator for ScopedStackScopeIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        unsafe {
            let scope_start = self.end.sub(self.cur_len);
            let scope = std::slice::from_raw_parts(scope_start, self.cur_len);

            self.end = self.end.sub(self.cur_len + 1);
            debug_assert!(self.end >= self.start);
            self.cur_len = (*self.end).prev_len;

            Some(mem::transmute::<&[ScopedStackValue<T>], &[T]>(scope))
        }
    }
}

impl<T> FusedIterator for ScopedStackScopeIter<'_, T> {}

/*
#[cfg(test)]
mod benches {
    extern crate test;
    use super::*;
    use crate::ptr::Ptr;
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
        let e = crate::ast::Ast::new(
            crate::ast::ExprKind::Ident(n),
            crate::parser::lexer::Span::new(0, 0),
        );
        let p = Ptr::from(&e);
        b.iter(|| {
            let mut defer_stack = ScopedStack::default();
            for _ in 0..10 {
                defer_stack.open_scope();
                for _ in 0..10 {
                    defer_stack.push(p);
                }
            }
            for _ in 0..10 {
                for _ in 0..10 {
                    defer_stack.push(p);
                }
                defer_stack.close_scope();
            }
        })
    }
}
*/
