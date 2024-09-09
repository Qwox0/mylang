use bumpalo::{AllocErr, Bump};
use core::slice;
use std::{alloc::Layout, marker::PhantomData, mem, ptr::NonNull};

pub struct ScratchPool<'bump, T: 'bump> {
    //bump: &'bump mut Bump,
    bump: Bump,
    _marker: PhantomData<(&'bump mut Bump, T)>,
    len: usize,
    // chunk_count: usize,
    // prev_cap: usize,
}

impl<'bump, T: 'bump> ScratchPool<'bump, T> {
    // /// resets the `scratch_bump`
    // pub fn new(scratch_bump: &'bump mut Bump) -> ScratchPool<T> {
    //     Self::assert_no_padding();
    //     scratch_bump.reset();
    //     ScratchPool { bump: scratch_bump, _marker: PhantomData, len: 0 }
    // }
    #[inline]
    pub fn new() -> ScratchPool<'bump, T> {
        Self::assert_no_padding();
        ScratchPool { bump: bumpalo::Bump::new(), _marker: PhantomData, len: 0 }
    }

    #[inline]
    pub fn new_with_first_val(val: T) -> Result<ScratchPool<'bump, T>, AllocErr> {
        let mut pool = Self::new();
        pool.push(val)?;
        Ok(pool)
    }

    /// see [`Bump::iter_allocated_chunks`]:
    /// 1. Every object allocated in this arena has the same alignment, and that
    ///    alignment is at most 16.
    /// 2. Every object's size is a multiple of its alignment.
    /// 3. None of the objects allocated in this arena contain any internal
    ///    padding.
    #[inline]
    fn assert_no_padding() {
        let alignment = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>();
        debug_assert!(alignment <= 16);
        debug_assert!(size.is_multiple_of(alignment));
    }

    #[inline]
    pub fn push(&mut self, val: T) -> Result<(), AllocErr> {
        self.bump.try_alloc(val)?;
        self.len += 1;
        Ok(())
    }

    #[inline]
    pub fn get_item_count(&self) -> usize {
        self.len
    }

    /// copies the items from this scratch allocator to the `target` slice.
    ///
    /// # Panics
    ///
    /// Panics (in debug mode) if the length of `target` doesn't match
    /// [`ScratchSliceBuilder::get_item_count`]
    pub fn clone_to_slice(&self, target: &mut [T])
    where T: Clone {
        debug_assert!(target.len() == self.get_item_count());

        let mut rev_target_iter = target.iter_mut().rev();

        for (ptr, len) in unsafe { self.bump.iter_allocated_chunks_raw() } {
            let ptr = ptr as *const T;
            let len = len / mem::size_of::<T>();
            for x in unsafe { slice::from_raw_parts(ptr, len) } {
                let t = rev_target_iter.next().unwrap();
                *t = x.clone();
            }
        }
    }

    pub fn clone_to_slice_in_bump(&self, target_bump: &Bump) -> Result<NonNull<[T]>, AllocErr>
    where T: Clone {
        let len = self.get_item_count();
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = target_bump.alloc_layout(layout).cast::<T>();
        let target_slice = unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), len) };
        self.clone_to_slice(target_slice);
        Ok(NonNull::from(target_slice))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clone_to_slice() {
        // let mut scratch_alloc = Bump::new();
        // let mut scratch = ScratchPool::<i32>::new(&mut scratch_alloc);
        let mut scratch = ScratchPool::<i32>::new();

        let count = 10000;

        for x in 0..count as i32 {
            scratch.push(x).unwrap();
        }

        assert!(scratch.len == count);

        let mut full_vec = vec![0; count];

        scratch.clone_to_slice(&mut full_vec[0..count]);

        assert!(full_vec.iter().enumerate().all(|(idx, x)| idx == *x as usize));
    }

    #[test]
    fn test_clone_to_slice_in_bump() {
        // let mut scratch_alloc = Bump::new();
        // let mut scratch = ScratchPool::<i32>::new(&mut scratch_alloc);
        let mut scratch = ScratchPool::<i32>::new();

        let count = 10000;

        for x in 0..count as i32 {
            scratch.push(x).unwrap();
        }

        assert!(scratch.len == count);

        let mut target_bump = Bump::new();
        let result_slice =
            unsafe { scratch.clone_to_slice_in_bump(&mut target_bump).unwrap().as_ref() };

        println!("{:?}", result_slice);

        assert!(result_slice.iter().enumerate().all(|(idx, x)| idx == *x as usize));
    }

    /*
    #[test]
    fn test_no_padding() {
        let mut scratch_alloc = Bump::new();
        let mut scratch = ScratchPool::<Expr>::new(&mut scratch_alloc);

        for _ in 0..2 {
            scratch
                .push(Expr::new(
                    ExprKind::Fn {
                        params: NonNull::from(&[(Ident { span: Span::new(1, 3) }, None)]),
                        ret_type: None,
                        body: NonNull::new(100000 as *mut _).unwrap(),
                    },
                    Span::new(1234, usize::MAX),
                ))
                .unwrap();
        }

        println!("print data for debugging");
        unsafe {
            for (ptr, len) in scratch.bump.iter_allocated_chunks_raw() {
                let bytes = slice::from_raw_parts(ptr, len);
                println!("bytes: {:x?}", bytes);

                let ptr = ptr as *const Expr;
                let len = len / mem::size_of::<Expr>();
                let items = slice::from_raw_parts(ptr, len);
                println!("items: {:?}", items);
            }
        }

        const EXPR_SIZE: usize = mem::size_of::<Expr>();

        let bytes = unsafe {
            let (ptr, len) =
                scratch.bump.iter_allocated_chunks_raw().next().expect("test does allocations");
            slice::from_raw_parts(ptr, len)
        };

        let first_item_first_byte = bytes[0];
        let first_item_last_byte = bytes[EXPR_SIZE - 1];
        let second_item_first_byte = bytes[EXPR_SIZE];
        assert!(first_item_last_byte == u8::MAX);
        assert!(first_item_first_byte == second_item_first_byte);
    }
    */

    /// old idea for the [`ScratchSliceBuilder`] data struct
    #[test]
    fn test_chunk_count() {
        // let mut scratch_alloc = Bump::new();
        // let mut scratch = ScratchPool::<i32>::new(&mut scratch_alloc);
        let mut scratch = ScratchPool::<i32>::new();

        let mut chunk_count = unsafe { scratch.bump.iter_allocated_chunks_raw() }.count();
        let mut prev_cap = scratch.bump.chunk_capacity();

        for x in 0..10000 {
            scratch.push(x).unwrap();

            let cap = scratch.bump.chunk_capacity();
            if cap > prev_cap {
                chunk_count += 1;
            }
            prev_cap = cap;
        }

        println!(
            "{} == {}",
            unsafe { scratch.bump.iter_allocated_chunks_raw() }.count(),
            chunk_count
        );
        assert!(unsafe { scratch.bump.iter_allocated_chunks_raw() }.count() == chunk_count);
    }
}
