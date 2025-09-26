use crate::util::UnwrapDebug;
use std::{mem::MaybeUninit, ops::Deref};

pub type Buffer<T> = Box<[T]>;

/// Can't exceed it's initial capacity.
pub struct CappedVec<T>(Vec<T>);

impl<T> From<Vec<T>> for CappedVec<T> {
    fn from(inner: Vec<T>) -> Self {
        CappedVec(inner)
    }
}

impl<T> CappedVec<T> {
    pub fn new(capacity: usize) -> Self {
        CappedVec::from(Vec::with_capacity(capacity))
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.0.push_within_capacity(value).ok().u();
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.0.len() == self.0.capacity()
    }

    pub fn into_boxed_slice(self) -> Buffer<T> {
        self.0.into_boxed_slice()
    }
}

impl<T> Deref for CappedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<T> IntoIterator for CappedVec<T> {
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;
    type Item = <Vec<T> as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// [`FixedVec`] which supports initialization in a random order.
pub struct UnorderedInitBuf<T> {
    buf: Buffer<MaybeUninit<T>>,
    #[cfg(debug_assertions)]
    was_initialized: Buffer<bool>,
}

impl<T> UnorderedInitBuf<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buf = Vec::with_capacity(capacity);
        unsafe { buf.set_len(capacity) };
        Self {
            buf: buf.into_boxed_slice(),
            #[cfg(debug_assertions)]
            was_initialized: vec![false; capacity].into(),
        }
    }

    #[inline]
    pub fn set(&mut self, idx: usize, value: T) {
        debug_assert!(idx < self.buf.len());
        *unsafe { self.buf.get_unchecked_mut(idx) } = MaybeUninit::new(value);
        #[cfg(debug_assertions)]
        {
            self.was_initialized[idx] = true;
        }
    }

    #[inline]
    pub fn assume_init(self) -> Buffer<T> {
        #[cfg(debug_assertions)]
        debug_assert!(self.was_initialized.iter().all(|b| *b));
        unsafe { self.buf.assume_init() }
    }
}
