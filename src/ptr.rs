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
        Self(self.0.clone())
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
        // SAFETY: yolo
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
    pub fn as_mut<'a>(&mut self) -> &'a mut T {
        unsafe { self.0.as_mut() }
    }

    #[inline]
    pub fn raw(self) -> *mut T {
        self.0.as_ptr()
    }

    #[inline]
    pub const fn cast<U>(self) -> Ptr<U> {
        Ptr(self.0.cast::<U>())
    }
}

impl<T: ?Sized> PartialEq for Ptr<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

impl<T: ?Sized> Eq for Ptr<T> {}

impl<T: ?Sized> PartialOrd for Ptr<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.as_ptr().cast::<()>().partial_cmp(&other.0.as_ptr().cast::<()>())
    }
}

impl<T: ?Sized> Ord for Ptr<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_ptr().cast::<()>().cmp(&other.0.as_ptr().cast::<()>())
    }
}

impl<T: ?Sized + std::fmt::Debug> std::fmt::Debug for Ptr<T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "Ptr->")?;
            self.as_ref().fmt(f)
        } else {
            self.0.fmt(f)
        }
    }
}

impl<T: ?Sized> std::hash::Hash for Ptr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl ToString for Ptr<str> {
    fn to_string(&self) -> String {
        self.as_ref().to_string()
    }
}
