#![allow(unused)]

use std::{
    convert::Infallible,
    ops::{ControlFlow, FromResidual},
};

pub trait TupleMap0 {
    type Val0;
    type Mapped0<T>;
    fn map0<U>(self, f: impl FnOnce(Self::Val0) -> U) -> Self::Mapped0<U>;
}

impl<A, B> TupleMap0 for (A, B) {
    type Mapped0<T> = (T, B);
    type Val0 = A;

    fn map0<U>(self, f: impl FnOnce(Self::Val0) -> U) -> Self::Mapped0<U> {
        (f(self.0), self.1)
    }
}

pub trait OptionExt<T>: Sized {
    fn ignore(self) {}
    fn do_(self, f: impl FnOnce(T));
}

impl<T> OptionExt<T> for Option<T> {
    fn do_(self, f: impl FnOnce(T)) {
        self.map(f);
    }
}

pub trait MutChain: Sized {
    #[inline]
    fn mut_chain(mut self, f: impl FnOnce(&mut Self)) -> Self {
        f(&mut self);
        self
    }
}

impl<T> MutChain for T {}

pub struct Batching<I, F> {
    f: F,
    iter: I,
}

impl<B, F, I> Iterator for Batching<I, F>
where
    I: Iterator,
    F: FnMut(&mut I) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (self.f)(&mut self.iter)
    }
}

pub enum TryBatchingFoldResult<T, E> {
    Continue(T),
    Finish(T),
    /// break because of an error
    Err(E),
}

impl<T, E> FromResidual<Result<Infallible, E>> for TryBatchingFoldResult<T, E> {
    fn from_residual(residual: Result<Infallible, E>) -> Self {
        match residual {
            Ok(_) => unreachable!(),
            Err(err) => TryBatchingFoldResult::Err(err),
        }
    }
}

pub trait IteratorBatching: Iterator + Sized {
    fn batching<B, F>(self, f: F) -> Batching<Self, F>
    where F: FnMut(&mut Self) -> Option<B> {
        Batching { f, iter: self }
    }

    fn batching_fold<B, F>(mut self, init: B, mut f: F) -> B
    where F: FnMut(B, &mut Self) -> ControlFlow<B, B> {
        let mut acc = init;
        loop {
            match f(acc, &mut self) {
                ControlFlow::Continue(new) => acc = new,
                ControlFlow::Break(acc) => return acc,
            }
        }
    }

    fn try_batching_fold<B, F, E>(&mut self, init: B, mut f: F) -> Result<B, E>
    where F: FnMut(B, &mut Self) -> TryBatchingFoldResult<B, E> {
        let mut acc = init;
        loop {
            match f(acc, self) {
                TryBatchingFoldResult::Continue(new) => acc = new,
                TryBatchingFoldResult::Finish(ok) => return Ok(ok),
                TryBatchingFoldResult::Err(err) => return Err(err),
            }
        }
    }
}

impl<I: Iterator> IteratorBatching for I {}
