use crate::ptr::Ptr;
use std::{alloc::Layout, fmt, mem::MaybeUninit, ptr::NonNull};

#[derive(Debug)]
pub struct Arena(pub bumpalo::Bump);

#[derive(Debug, Clone)]
pub struct AllocErr(bumpalo::AllocErr);

impl Arena {
    #[inline]
    pub fn new() -> Self {
        Self(bumpalo::Bump::new())
    }

    #[inline]
    pub fn alloc<T>(&self, val: T) -> Result<Ptr<T>, AllocErr> {
        match self.0.try_alloc(val) {
            Ok(t) => Ok(Ptr::from(t)),
            Err(e) => Err(AllocErr(e)),
        }
    }

    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        self.0.try_alloc_layout(layout).map_err(AllocErr)
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

    pub fn alloc_uninit_slice<T>(&self, len: usize) -> Result<Ptr<[MaybeUninit<T>]>, AllocErr> {
        let layout = Layout::array::<T>(len).unwrap();
        let dst = self.alloc_layout(layout)?.cast::<MaybeUninit<T>>();
        Ok(Ptr::from(unsafe { core::slice::from_raw_parts_mut(dst.as_ptr(), len) }))
    }
}

impl From<bumpalo::AllocErr> for AllocErr {
    fn from(e: bumpalo::AllocErr) -> Self {
        AllocErr(e)
    }
}

impl fmt::Display for AllocErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
