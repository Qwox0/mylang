use super::jit_run_test;

#[derive(Debug)]
#[repr(C)]
struct Slice {
    ptr: *const i32,
    len: usize,
}

#[test]
fn create_slice_from_array() {
    let out = jit_run_test!("
array: [7]i32 = .[1, 2, 4, 8, 16, 32, 64];
slice: []i32 = array[3..6];
if slice[0] != 8 || slice[1] != 16 || slice[2] != 32
    return slice[0..0];
slice" => Slice)
    .unwrap();

    assert!(!out.ptr.is_null());
    assert_eq!(out.len, 6 - 3);
}

#[test]
fn create_slice_from_slice() {
    let out = jit_run_test!("
array: [7]i32 = .[1, 2, 4, 8, 16, 32, 64];
slice: []i32 = array[1..6];
slice2: []i32 = slice[0..2];
if slice2[0] != 2 || slice2[1] != 4
    return slice[0..0];
slice2" => Slice)
    .unwrap();

    assert!(!out.ptr.is_null());
    assert_eq!(out.len, 2 - 0);
}

/*
#[test]
fn arr_to_slice() {
    let out = jit_run_test!("
slice: []i32 = .[1, 2, 4, 8, 16, 32, 64].&;
slice" => Slice)
    .unwrap();

    assert!(!out.ptr.is_null());
    assert_eq!(out.len, 7);
}

#[test]
fn empty_slice() {
    let out = jit_run_test!("
slice: []i32 = &.[];
slice" => Slice)
    .unwrap();

    assert_eq!(out.len, 0);
}
*/
