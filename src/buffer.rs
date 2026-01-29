use crate::{
    ptr::{OPtr, Ptr},
    scratch_allocator::ScratchAllocator, util::debug_only_assert,
};
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
    #[allow(unused)]
    pub fn with_buf(buf: Buffer<MaybeUninit<T>>, in_scratch: OPtr<ScratchAllocator>) -> Self {
        Self { buf: UnorderedInitBuf::with_buf(buf, in_scratch), len: 0 }
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

    #[cfg(debug_assertions)]
    in_scratch: OPtr<ScratchAllocator>,
}

impl<T> UnorderedInitBuf<T> {
    pub fn with_buf(
        buf: Buffer<MaybeUninit<T>>,
        #[allow(unused)] in_scratch: OPtr<ScratchAllocator>,
    ) -> Self {
        #[cfg(debug_assertions)]
        if let Some(scratch) = in_scratch {
            scratch.as_mut().alive_allocations += 1;
        }

        Self {
            buf,
            #[cfg(debug_assertions)]
            was_initialized: vec![false; buf.len()].into_boxed_slice(),
            #[cfg(debug_assertions)]
            in_scratch,
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
        debug_only_assert!(self.was_initialized.iter().all(|b| *b));
        self.buf.assume_init()
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(debug_assertions)]
impl<T> Drop for UnorderedInitBuf<T> {
    fn drop(&mut self) {
        if let Some(scratch) = self.in_scratch {
            scratch.as_mut().alive_allocations -= 1;
        }
    }
}
