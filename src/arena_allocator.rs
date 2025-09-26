use crate::{
    diagnostics::{HandledErr, cerror},
    parser::lexer::Span,
    ptr::Ptr,
};
use std::{alloc::Layout, mem::MaybeUninit, ptr::NonNull};

#[derive(Debug)]
pub struct Arena(pub bumpalo::Bump);

pub type AllocErr = HandledErr;

pub fn handle_alloc_err(e: bumpalo::AllocErr) -> AllocErr {
    cerror!(Span::ZERO, "allocation failed: {e}")
}

impl Arena {
    #[inline]
    pub fn new() -> Self {
        Self(bumpalo::Bump::new())
    }

    #[inline]
    pub fn alloc<T>(&self, val: T) -> Result<Ptr<T>, AllocErr> {
        Ok(Ptr::from_ref(self.0.try_alloc(val)?))
    }

    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        self.0.try_alloc_layout(layout).map_err(handle_alloc_err)
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
        Ok(Ptr::from(unsafe { core::slice::from_raw_parts_mut(dst.as_ptr(), len) }))
    }

    pub fn alloc_slice_default<T: Default + Copy>(&self, len: usize) -> Result<Ptr<[T]>, AllocErr> {
        Ok(Ptr::from(self.0.try_alloc_slice_fill_copy(len, T::default())?))
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
