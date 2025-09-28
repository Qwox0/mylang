use crate::ptr::Ptr;
use std::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

pub type Buffer<T> = Ptr<[T]>;

/// Can't exceed it's initial capacity.
pub struct CappedVec<T> {
    buf: UnorderedInitBuf<T>,
    len: usize,
}

impl<T> CappedVec<T> {
    pub fn with_buf(buf: Buffer<MaybeUninit<T>>) -> Self {
        let buf = UnorderedInitBuf::with_buf(buf);
        Self { buf, len: 0 }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        debug_assert!(!self.is_full());
        self.buf.set(self.len, value);
        self.len += 1;
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.buf.capacity() == self.len
    }

    pub fn into_full_buf(self) -> Buffer<T> {
        debug_assert!(self.is_full());
        self.buf.assume_init()
    }
}

impl<T> Deref for CappedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { self.buf.buf[..self.len].assume_init_ref() }
    }
}

impl<T> DerefMut for CappedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.buf.buf[..self.len].assume_init_mut() }
    }
}

/// [`FixedVec`] which supports initialization in a random order.
pub struct UnorderedInitBuf<T> {
    buf: Buffer<MaybeUninit<T>>,
    #[cfg(debug_assertions)]
    was_initialized: Box<[bool]>,
}

impl<T> UnorderedInitBuf<T> {
    pub fn with_buf(buf: Buffer<MaybeUninit<T>>) -> Self {
        Self {
            #[cfg(debug_assertions)]
            was_initialized: vec![false; buf.len()].into_boxed_slice(),
            buf,
        }
    }

    #[inline]
    pub fn set(&mut self, idx: usize, value: T) {
        debug_assert!(idx < self.buf.len(), "{idx} < {}", self.buf.len());
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
        self.buf.assume_init()
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }
}
