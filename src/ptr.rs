use std::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// [`NonNull`] but without the requirement for the `unsafe` keyword.
#[repr(transparent)]
pub struct Ptr<T: ?Sized>(NonNull<T>);

impl<T: ?Sized> Clone for Ptr<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Ptr<T> {}

impl<T: ?Sized> Deref for Ptr<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: ?Sized> DerefMut for Ptr<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T: ?Sized> From<&T> for Ptr<T> {
    #[inline]
    fn from(value: &T) -> Self {
        Ptr(value.into())
    }
}

impl<T: ?Sized> From<&mut T> for Ptr<T> {
    #[inline]
    fn from(value: &mut T) -> Self {
        Ptr(value.into())
    }
}

impl<T: ?Sized> From<*mut T> for Ptr<T> {
    #[inline]
    fn from(value: *mut T) -> Self {
        debug_assert!((value as *const () as u64) > 0x500);
        debug_assert!(!value.is_null());
        Ptr(unsafe { NonNull::new_unchecked(value) })
    }
}

impl<T: ?Sized> Ptr<T> {
    pub const fn new(ptr: NonNull<T>) -> Ptr<T> {
        Self(ptr)
    }

    #[inline]
    pub const fn as_ref<'a>(&self) -> &'a T {
        unsafe { self.0.as_ref() }
    }

    #[inline]
    pub fn as_mut<'a>(&self) -> &'a mut T {
        unsafe { self.0.as_ptr().as_mut_unchecked() }
    }

    #[inline]
    pub const fn raw(self) -> *mut T {
        self.0.as_ptr()
    }

    #[inline]
    pub const fn cast<U>(self) -> Ptr<U> {
        Ptr(self.0.cast::<U>())
    }

    /*
    pub const fn from_ref(r: &T) -> Ptr<T> {
        Ptr(NonNull::from_ref(r))
    }
    */

    pub fn from_ref(r: &T) -> Ptr<T> {
        let p = Ptr(NonNull::from_ref(r));
        debug_assert!((p.raw() as *const () as usize) > 0x500);
        p
    }

    pub fn drop_in_place(self) {
        unsafe { self.0.drop_in_place() }
    }
}

impl<T> Ptr<T> {
    /// Casts the pointer into a pointer to a slice of length 1.
    #[inline]
    pub const fn as_slice1(self) -> Ptr<[T]> {
        let slice = unsafe { std::slice::from_raw_parts(self.raw(), 1) };
        Ptr::new(NonNull::from_ref(slice))
    }
}

impl<T, U> PartialEq<Ptr<U>> for Ptr<T> {
    #[inline]
    fn eq(&self, other: &Ptr<U>) -> bool {
        std::ptr::eq(self.raw(), other.raw() as *const T)
    }
}

impl<T> Eq for Ptr<T> {}

impl<T: ?Sized + std::fmt::Display> std::fmt::Display for Ptr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.as_ref(), f)
    }
}

const DEFAULT_PTR_DEBUG_DEPTH: usize = 7;

impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for Ptr<T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut opt = f.options();
        let width = opt.get_width().unwrap_or(DEFAULT_PTR_DEBUG_DEPTH);
        if width == 0 {
            write!(f, "{:x?} ...", self.0)
        } else if opt.get_debug_as_hex().is_some() {
            self.0.fmt(f)
        } else {
            if f.alternate() {
                write!(f, "Ptr->")?;
            }
            self.as_ref()
                .fmt(&mut f.with_options(*opt.width(Some(width.saturating_sub(1)))))
        }
    }
}

impl<T: ?Sized> std::fmt::Pointer for Ptr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Pointer::fmt(&self.0, f)
    }
}

impl<T: ?Sized> std::hash::Hash for Ptr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> Ptr<[T]> {
    pub fn as_slice(&self) -> &[T] {
        &self[..]
    }
}

pub type OPtr<T> = Option<Ptr<T>>;

pub trait OPtrExt<T: ?Sized> {
    fn raw(self) -> *mut T
    where T: Sized;
}

impl<T: ?Sized> OPtrExt<T> for OPtr<T> {
    fn raw(self) -> *mut T
    where T: Sized {
        match self {
            Some(p) => p.raw(),
            None => std::ptr::null_mut(),
        }
    }
}

impl<T, U> PartialEq<Ptr<U>> for OPtr<T> {
    #[inline]
    fn eq(&self, other: &Ptr<U>) -> bool {
        match *self {
            Some(p) => p == *other,
            None => false,
        }
    }
}

impl<T, U> PartialEq<OPtr<U>> for Ptr<T> {
    fn eq(&self, other: &OPtr<U>) -> bool {
        other == self
    }
}
