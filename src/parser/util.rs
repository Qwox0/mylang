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

pub trait IteratorBatching: Iterator + Sized {
    fn batching<B, F>(self, f: F) -> Batching<Self, F>
    where F: FnMut(&mut Self) -> Option<B> {
        Batching { f, iter: self }
    }
}

impl<I: Iterator> IteratorBatching for I {}
