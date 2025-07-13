use std::{
    hash::Hash,
    mem::MaybeUninit,
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
    pub fn as_ref<'a>(&self) -> &'a T {
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
    pub fn cast<U>(self) -> Ptr<U>
    where T: Sized {
        debug_assert!(self.raw().is_aligned());
        let p = Ptr(self.0.cast::<U>());
        debug_assert!(p.raw().is_aligned());
        p
    }

    pub fn from_ref(r: &T) -> Ptr<T> {
        let p = Ptr(NonNull::from_ref(r));
        debug_assert!((p.raw() as *const () as usize) > 0x500, "ptr ({p:p}) might be invalid");
        p
    }

    pub fn from_ptr(ptr: *mut T) -> OPtr<T> {
        NonNull::new(ptr).map(Ptr::new)
    }

    pub fn drop_in_place(self) {
        unsafe { self.0.drop_in_place() }
    }

    pub fn as_hash_key(self) -> HashKeyPtr<T>
    where T: Hash + Eq {
        HashKeyPtr(self)
    }
}

impl<T> Ptr<[T]> {
    pub fn empty_slice() -> Ptr<[T]> {
        Ptr::from(&[] as &[T])
    }
}

impl<T> Ptr<MaybeUninit<T>> {
    pub fn write(self, val: T) -> Ptr<T> {
        self.as_mut().write(val);
        self.cast::<T>()
    }
}

impl<T> Ptr<[MaybeUninit<T>]> {
    pub fn assume_init(self) -> Ptr<[T]> {
        Ptr::from_ref(unsafe { self.assume_init_ref() })
    }
}

impl<T> Ptr<T> {
    /// Casts the pointer into a pointer to a slice of length 1.
    #[inline]
    pub const fn as_slice1(self) -> Ptr<[T]> {
        let slice = unsafe { std::slice::from_raw_parts(self.raw(), 1) };
        Ptr::new(NonNull::from_ref(slice))
    }

    pub fn p_eq<U>(self, other: Ptr<U>) -> bool {
        std::ptr::eq(self.raw(), other.raw() as *mut T)
    }
}

impl<T: ?Sized> PartialEq<Ptr<T>> for Ptr<T> {
    #[inline]
    fn eq(&self, other: &Ptr<T>) -> bool {
        std::ptr::eq(self.raw(), other.raw())
    }
}

impl<T: ?Sized> Eq for Ptr<T> {}

impl<T: ?Sized + std::fmt::Display> std::fmt::Display for Ptr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.as_ref(), f)
    }
}

const DEFAULT_PTR_DEBUG_DEPTH: u16 = 7;

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

impl<T> PartialEq<Ptr<T>> for OPtr<T> {
    #[inline]
    fn eq(&self, other: &Ptr<T>) -> bool {
        match *self {
            Some(p) => p == *other,
            None => false,
        }
    }
}

impl<T> PartialEq<OPtr<T>> for Ptr<T> {
    fn eq(&self, other: &OPtr<T>) -> bool {
        other == self
    }
}

/// Wrapper around [`Ptr`] which calls the correct [`Hash::hash`] and [`PartialEq::eq`]
/// implementations.
pub struct HashKeyPtr<T: Hash + Eq + ?Sized>(pub Ptr<T>);

impl<T: Hash + Eq + ?Sized> PartialEq for HashKeyPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}
impl<T: Hash + Eq + ?Sized> Eq for HashKeyPtr<T> {}

impl<T: Hash + Eq + ?Sized> Hash for HashKeyPtr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ref().hash(state);
    }
}
