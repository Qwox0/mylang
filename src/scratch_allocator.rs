use crate::{
    arena_allocator::{AllocErr, Arena},
    ast,
    buffer::{CappedVec, UnorderedInitBuf},
    ptr::Ptr,
    util::debug_only_assert_eq,
};
#[cfg(debug_assertions)]
use std::any::type_name;
use std::ops::{Deref, DerefMut};

pub struct ScratchAllocator {
    arena: Arena,

    #[cfg(debug_assertions)]
    pub alive_allocations: usize,
}

#[cfg(debug_assertions)]
const DEBUG_SCRATCH: bool = false;
#[cfg(debug_assertions)]
const SKIP_ALLOC: bool = false;

macro_rules! debug_scratch {
    ($fmt:literal $(, $val:expr)* $(,)?) => {
        #[cfg(debug_assertions)]
        if DEBUG_SCRATCH && !(SKIP_ALLOC && $fmt.starts_with("alloc")) {
            println!($fmt, $($val),*);
        }
    };
}

impl ScratchAllocator {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let buf = bumpalo::Bump::with_capacity(capacity);
        //buf.set_allocation_limit(Some(buf.allocated_bytes()));
        Self {
            arena: Arena(buf),
            #[cfg(debug_assertions)]
            alive_allocations: 0,
        }
    }

    pub fn alloc<T>(&self, val: T) -> Result<TmpPtr<T>, AllocErr> {
        Ok(TmpPtr::new(self.arena.alloc(val)?, self))
    }

    pub fn alloc_capped_vec<T>(&self, capacity: usize) -> Result<CappedVec<T>, AllocErr> {
        debug_scratch!("alloc CappedVec<{}> (capacity: {capacity})", type_name::<T>());
        Ok(CappedVec::with_buf(self.arena.alloc_uninit_slice(capacity)?, Some(Ptr::from_ref(self))))
    }

    pub fn alloc_unordered_init_buf<T>(
        &self,
        capacity: usize,
    ) -> Result<UnorderedInitBuf<T>, AllocErr> {
        debug_scratch!("alloc UnorderedInitBuf<{}> (capacity: {capacity})", type_name::<T>());
        Ok(UnorderedInitBuf::with_buf(
            self.arena.alloc_uninit_slice(capacity)?,
            Some(Ptr::from_ref(self)),
        ))
    }

    pub fn alloc_slice_fill_iter<T>(
        &self,
        iter: impl ExactSizeIterator<Item = T>,
    ) -> Result<TmpPtr<[T]>, AllocErr> {
        debug_scratch!("alloc_slice_fill_iter<{}> (iter len: {})", type_name::<T>(), iter.len());
        Ok(TmpPtr::from_ref(self.arena.0.try_alloc_slice_fill_iter(iter)?, self))
    }

    #[allow(unused_variables)]
    pub fn reset(&mut self, s: Ptr<ast::Ast>) {
        #[cfg(debug_assertions)]
        if self.arena.0.iter_allocated_chunks().count() > 1 {
            use crate::{diagnostics::cwarn, display_code::debug_expr, parser::lexer::Span};

            /// This type is private in [`bumpalo`]
            #[repr(C)]
            struct ChunkFooter {
                data: Ptr<u8>,
                layout: std::alloc::Layout,
                prev: std::cell::Cell<Ptr<ChunkFooter>>,
                ptr: std::cell::Cell<Ptr<u8>>,
                allocated_bytes: usize,
            }

            let mut last = unsafe {
                std::mem::transmute::<_, Ptr<ChunkFooter>>(self.arena.0.iter_allocated_chunks_raw())
            };
            while let prev = last.prev.get()
                && prev.prev.get() != prev
            {
                last = prev;
            }
            let cap = last.allocated_bytes;
            let needed = self.count_allocated_bytes();
            cwarn!(Span::ZERO, "exceeded scratch capacity (capacity: {cap}; needed: {needed})");
            debug_expr!(s);
        }

        debug_only_assert_eq!(
            self.alive_allocations,
            0,
            "Cannot reset scratch if allocations are still alive"
        );

        debug_scratch!(
            "reset scratch (used: {}; chunks: {})",
            self.count_allocated_bytes(),
            self.arena.0.iter_allocated_chunks().count()
        );

        self.arena.0.reset();
    }

    pub fn count_allocated_bytes(&self) -> usize {
        unsafe { self.arena.0.iter_allocated_chunks_raw() }.map(|(_, len)| len).sum()
    }
}

pub struct TmpPtr<T: ?Sized> {
    ptr: Ptr<T>,

    #[cfg(debug_assertions)]
    in_scratch: Ptr<ScratchAllocator>,
}

impl<T: ?Sized> TmpPtr<T> {
    pub fn new(ptr: Ptr<T>, #[allow(unused)] in_scratch: &ScratchAllocator) -> Self {
        Self {
            ptr,
            #[cfg(debug_assertions)]
            in_scratch: {
                let in_scratch = Ptr::from_ref(in_scratch);
                in_scratch.as_mut().alive_allocations += 1;
                in_scratch
            },
        }
    }

    pub fn from_ref(r: &T, #[allow(unused)] in_scratch: &ScratchAllocator) -> Self {
        Self::new(Ptr::from_ref(r), in_scratch)
    }
}

impl<T: ?Sized> Deref for TmpPtr<T> {
    type Target = Ptr<T>;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl<T: ?Sized> DerefMut for TmpPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}

#[cfg(debug_assertions)]
impl<T: ?Sized> Drop for TmpPtr<T> {
    fn drop(&mut self) {
        self.in_scratch.alive_allocations -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alive_allocations_counter_with_tmp_ptr_deref() {
        let alloc = ScratchAllocator::new(100);
        debug_only_assert_eq!(alloc.alive_allocations, 0);

        let tmp_ptr = alloc.alloc(123_i64).unwrap();
        debug_only_assert_eq!(alloc.alive_allocations, 1);

        let ptr = *tmp_ptr.deref(); // TODO: force this to be explicit?
        debug_only_assert_eq!(alloc.alive_allocations, 1);
        drop(tmp_ptr);
        debug_only_assert_eq!(alloc.alive_allocations, 0);

        assert_eq!(*ptr, 123);
    }
}
