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
