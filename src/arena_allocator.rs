use crate::{
    ast,
    buffer::{CappedVec, UnorderedInitBuf},
    diagnostics::{HandledErr, cerror},
    parser::lexer::Span,
    ptr::Ptr,
};
#[cfg(debug_assertions)]
use std::any::type_name;
use std::{alloc::Layout, mem::MaybeUninit};

#[derive(Debug)]
pub struct Arena(pub bumpalo::Bump, #[cfg(debug_assertions)] ArenaKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArenaKind {
    Normal,
    Scratch,
}

pub type AllocErr = HandledErr;

pub fn handle_alloc_err(e: bumpalo::AllocErr) -> AllocErr {
    cerror!(Span::ZERO, "allocation failed: {e}")
}

const DEBUG_SCRATCH: bool = false;
const SKIP_ALLOC: bool = false;

macro_rules! debug_scratch {
    ($self:ident, $fmt:literal $(, $val:expr)* $(,)?) => {
        #[cfg(debug_assertions)]
        if DEBUG_SCRATCH && $self.1 == ArenaKind::Scratch && !(SKIP_ALLOC && $fmt.starts_with("alloc")) {
            println!($fmt, $($val),*);
        }
    };
}

impl Arena {
    pub const BUMP_OVERHEAD: usize = 64;

    #[inline]
    #[rustfmt::skip]
    #[allow(unused_variables)]
    fn new_(buf: bumpalo::Bump, kind: ArenaKind) -> Self {
        Self(buf, #[cfg(debug_assertions)] kind)
    }

    #[inline]
    pub fn new() -> Self {
        Self::new_(bumpalo::Bump::new(), ArenaKind::Normal)
    }

    #[inline]
    pub fn new_scratch(capacity: usize) -> Self {
        let buf = bumpalo::Bump::with_capacity(capacity);
        //buf.set_allocation_limit(Some(buf.allocated_bytes()));
        Self::new_(buf, ArenaKind::Scratch)
    }

    #[inline]
    pub fn alloc<T>(&self, val: T) -> Result<Ptr<T>, AllocErr> {
        debug_scratch!(self, "alloc {} (size: {})", type_name::<T>(), size_of::<T>());
        Ok(Ptr::from_ref(self.0.try_alloc(val)?))
    }

    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> Result<Ptr<u8>, AllocErr> {
        debug_scratch!(self, "alloc {layout:?}");
        Ok(Ptr::new(self.0.try_alloc_layout(layout)?))
    }

    #[inline]
    pub fn alloc_uninit<T>(&self) -> Result<Ptr<MaybeUninit<T>>, AllocErr> {
        let raw = self.0.try_alloc_layout(Layout::new::<T>())?;
        Ok(Ptr::new(raw).cast::<MaybeUninit<T>>())
    }

    #[inline]
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> Result<Ptr<[T]>, AllocErr> {
        Ok(Ptr::from_ref(self.0.try_alloc_slice_copy(slice)?))
    }

    #[inline]
    pub fn alloc_one_val_slice<T>(&self, val: T) -> Result<Ptr<[T]>, AllocErr> {
        let ptr = self.alloc(val)?;
        Ok(unsafe { core::slice::from_raw_parts_mut(ptr.as_mut() as *mut T, 1) }.into())
    }

    pub fn alloc_uninit_slice<T>(&self, len: usize) -> Result<Ptr<[MaybeUninit<T>]>, AllocErr> {
        let layout = Layout::array::<T>(len).unwrap();
        let dst = self.alloc_layout(layout)?.cast::<MaybeUninit<T>>();
        Ok(Ptr::from(unsafe { core::slice::from_raw_parts_mut(dst.raw(), len) }))
    }

    pub fn alloc_slice_default<T: Default + Copy>(&self, len: usize) -> Result<Ptr<[T]>, AllocErr> {
        Ok(Ptr::from(self.0.try_alloc_slice_fill_copy(len, T::default())?))
    }

    pub fn alloc_capped_vec<T>(&self, capacity: usize) -> Result<CappedVec<T>, AllocErr> {
        debug_scratch!(self, "alloc CappedVec<{}> (capacity: {capacity})", type_name::<T>());
        Ok(CappedVec::with_buf(self.alloc_uninit_slice(capacity)?))
    }

    pub fn alloc_unordered_init_buf<T>(
        &self,
        capacity: usize,
    ) -> Result<UnorderedInitBuf<T>, AllocErr> {
        debug_scratch!(self, "alloc UnorderedInitBuf<{}> (capacity: {capacity})", type_name::<T>());
        Ok(UnorderedInitBuf::with_buf(self.alloc_uninit_slice(capacity)?))
    }

    pub fn alloc_slice_fill_iter<T>(
        &self,
        iter: impl ExactSizeIterator<Item = T>,
    ) -> Result<Ptr<[T]>, AllocErr> {
        Ok(Ptr::from_ref(self.0.try_alloc_slice_fill_iter(iter)?))
    }

    pub fn reset(&mut self) {
        debug_scratch!(
            self,
            "reset scratch (used: {}; chunks: {})",
            self.count_allocated_bytes(),
            self.0.iter_allocated_chunks().count()
        );
        self.0.reset();
    }

    #[allow(unused_variables)]
    pub fn reset_scratch(&mut self, s: Ptr<ast::Ast>) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.1, ArenaKind::Scratch);

        #[cfg(debug_assertions)]
        if self.0.iter_allocated_chunks().count() > 1 {
            use crate::{diagnostics::cwarn, display_code::debug_expr};

            /// This type is private in [`bumpalo`]
            #[repr(C)]
            struct ChunkFooter {
                data: Ptr<u8>,
                layout: Layout,
                prev: std::cell::Cell<Ptr<ChunkFooter>>,
                ptr: std::cell::Cell<Ptr<u8>>,
                allocated_bytes: usize,
            }

            let mut last = unsafe {
                std::mem::transmute::<_, Ptr<ChunkFooter>>(self.0.iter_allocated_chunks_raw())
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

        self.reset();
    }

    pub fn count_allocated_bytes(&self) -> usize {
        unsafe { self.0.iter_allocated_chunks_raw() }.map(|(_, len)| len).sum()
    }
}

impl From<bumpalo::AllocErr> for AllocErr {
    fn from(e: bumpalo::AllocErr) -> Self {
        handle_alloc_err(e)
    }
}

#[cfg(test)]
mod benches {
    extern crate test;
    use crate::arena_allocator::Arena;
    use test::*;

    macro_rules! bench_alloc_one_val {
        ($name:ident, $ref_name:ident, $val:expr) => {
            bench_alloc_one_val! { _ $name, alloc_one_val_slice($val) }
            bench_alloc_one_val! { _ $ref_name, alloc_slice(&[$val]) }
        };
        (_ $bench_name:ident, $f:ident($val:expr)) => {
            #[bench]
            fn $bench_name(b: &mut Bencher) {
                let alloc = Arena::new();
                b.iter(|| {
                    for _ in 0..100 {
                        let _ = black_box(black_box(&alloc).$f(black_box($val)));
                    }
                });
            }
        };
    }

    #[derive(Clone, Copy)]
    #[allow(unused)]
    struct Big([u8; 128]);
    const BIG_VAL: Big = Big([1; 128]);

    bench_alloc_one_val! { alloc_one_val_slice_u8, alloc_one_val_slice_u8_ref, 1u8 }
    bench_alloc_one_val! { alloc_one_val_slice_i64, alloc_one_val_slice_i64_ref, 1i64 }
    bench_alloc_one_val! { alloc_one_val_slice_ptr, alloc_one_val_slice_ptr_ref, std::ptr::null::<Big>() }
    bench_alloc_one_val! { alloc_one_val_slice_big_struct, alloc_one_val_slice_big_struct_ref, BIG_VAL }
}
