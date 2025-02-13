use crate::ptr::Ptr;
use std::{alloc::Layout, ptr::NonNull};

#[derive(Debug)]
pub struct Arena(pub bumpalo::Bump);

pub type AllocErr = bumpalo::AllocErr;

impl Arena {
    #[inline]
    pub fn new() -> Self {
        Self(bumpalo::Bump::new())
    }

    #[inline]
    pub fn alloc<T>(&self, val: T) -> Result<Ptr<T>, AllocErr> {
        self.0.try_alloc(val).map(Ptr::from)
    }

    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        self.0.try_alloc_layout(layout)
    }

    #[inline]
    pub fn alloc_uninit<T>(&self) -> Result<Ptr<T>, AllocErr> {
        let raw = self.0.try_alloc_layout(Layout::new::<T>())?;
        Ok(Ptr::new(raw).cast::<T>())
    }

    /// # Source
    ///
    /// see [`bumpalo::Bump::alloc_slice_copy`]
    #[inline]
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> Result<Ptr<[T]>, AllocErr> {
        let layout = core::alloc::Layout::for_value(slice);
        let dst = self.alloc_layout(layout)?.cast::<T>();

        Ok(Ptr::from(unsafe {
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst.as_ptr(), slice.len());
            core::slice::from_raw_parts_mut(dst.as_ptr(), slice.len())
        }))
    }
}
