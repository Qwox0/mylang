use crate::ptr::Ptr;
use std::{alloc::Layout, mem::MaybeUninit, ptr::NonNull};

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
    pub fn alloc_uninit<T>(&self) -> Result<Ptr<MaybeUninit<T>>, AllocErr> {
        let raw = self.0.try_alloc_layout(Layout::new::<T>())?;
        Ok(Ptr::new(raw).cast::<MaybeUninit<T>>())
    }

    /// # Source
    ///
    /// see [`bumpalo::Bump::alloc_slice_copy`]
    #[inline]
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> Result<Ptr<[T]>, AllocErr> {
        let dst = self.alloc_uninit_slice(slice.len())?;
        unsafe {
            let mut dst = std::mem::transmute::<Ptr<[MaybeUninit<T>]>, Ptr<[T]>>(dst);
            core::ptr::copy_nonoverlapping(slice.as_ptr(), dst.as_mut_ptr(), slice.len());
            Ok(dst)
        }
    }

    pub fn alloc_uninit_slice<T>(&self, len: usize) -> Result<Ptr<[MaybeUninit<T>]>, AllocErr> {
        let layout = Layout::array::<T>(len).unwrap();
        let dst = self.alloc_layout(layout)?.cast::<MaybeUninit<T>>();
        Ok(Ptr::from(unsafe { core::slice::from_raw_parts_mut(dst.as_ptr(), len) }))
    }

    pub fn alloc_slice_from_unsized_iter<T, I: Iterator<Item = T> + Clone>(
        &self,
        iter: I,
    ) -> Result<Ptr<[T]>, AllocErr> {
        let len = iter.clone().count();
        let mut slice = self.alloc_uninit_slice::<T>(len)?;
        for (idx, t) in iter.enumerate() {
            debug_assert!(idx < slice.len());
            unsafe { slice.get_unchecked_mut(idx) }.write(t);
        }
        Ok(slice.cast_slice::<T>())
    }
}
