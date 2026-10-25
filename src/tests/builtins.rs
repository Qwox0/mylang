use crate::tests::*;

// (mylang ordering, llvm ir ordering, valid for load, valid for store, valid for fail_ordering of cmpxchg)
const ATOMIC_ORDERINGS: [(&'static str, &'static str, bool, bool, bool); 5] = [
    ("Relaxed", "monotonic", true, true, true),
    ("Acquire", "acquire", true, false, true),
    ("Release", "release", false, true, false),
    ("AcqRel", "acq_rel", false, false, false),
    ("SeqCst", "seq_cst", true, true, true),
];

// (mylang type, llvm ir cffi type, llvm ir atomic operation type, align)
const TYPES: [(&'static str, &'static str, &'static str, usize); 2] = [
    ("u32", "i32", "i32", 4),
    ("bool", "i1", "i8", 1),
    //("f64", "double", "double", 8), // atomic load and stores are supported, but not cmpxchg/fetch_*
];

#[test]
fn test_atomic_load() {
    for (ordering, llvm_ir_ordering, valid_ordering, ..) in ATOMIC_ORDERINGS {
        for (mylang_ty, llvm_ty, llvm_atomic_ty, align) in TYPES {
            #[rustfmt::skip]
            let code = format!("
atomic :: #import \"../../lib/std/atomic.mylang\";
test :: (val: *{mylang_ty}) -> atomic.load(val, .{ordering});
");
            if valid_ordering {
                let res = test_body(code).with_prelude().compile_no_err();
                assert_contains!(
                    res.llvm_ir(),
                    "{llvm_ty} @load.{mylang_ty}.{ordering}(ptr noundef %ptr) #0"
                );
                assert_contains!(
                    res.llvm_ir(),
                    "load atomic {llvm_atomic_ty}, ptr %ptr {llvm_ir_ordering}, align {align}"
                );

                assert_contains!(res.llvm_ir(), "attributes #0 = {{ \"alwaysinline\" }}");
            } else {
                test_body(code).with_prelude().error(
                    format!(
                        "no variant or associated constant `{ordering}` on enum type \
                         `LoadOrdering`"
                    ),
                    substr!(ordering),
                );
            }
        }
    }
}

#[test]
fn test_atomic_store() {
    for (ordering, llvm_ir_ordering, _, valid_ordering, ..) in ATOMIC_ORDERINGS {
        for (mylang_ty, llvm_ty, llvm_atomic_ty, align) in TYPES {
            #[rustfmt::skip]
            let code = format!("
atomic :: #import \"../../lib/std/atomic.mylang\";
mut static val: {mylang_ty};
set :: (new: {mylang_ty}) -> atomic.store(&mut val, new, .{ordering});
");
            if valid_ordering {
                let res = test_body(code).with_prelude().compile_no_err();
                assert_contains!(
                    res.llvm_ir(),
                    "void @store.{mylang_ty}.{ordering}(ptr noundef %ptr, {llvm_ty} noundef %val) \
                     #0"
                );
                let var_name = if mylang_ty == "bool" { "0" } else { "val" };
                assert_contains!(
                    res.llvm_ir(),
                    "store atomic {llvm_atomic_ty} %{var_name}, ptr %ptr {llvm_ir_ordering}, \
                     align {align}"
                );

                assert_contains!(res.llvm_ir(), "attributes #0 = {{ \"alwaysinline\" }}");
            } else {
                test_body(code).with_prelude().error(
                    format!(
                        "no variant or associated constant `{ordering}` on enum type \
                         `StoreOrdering`"
                    ),
                    substr!(ordering),
                );
            }
        }
    }
}

#[test]
fn test_atomic_cmpxchg() {
    for (ord_ok_idx, (ord_ok, llvm_ord_ok, ..)) in ATOMIC_ORDERINGS.into_iter().enumerate() {
        for (ord_fail_idx, (ord_fail, llvm_ord_fail, _, _, valid_fail_ord)) in
            ATOMIC_ORDERINGS.into_iter().enumerate()
        {
            for (mylang_ty, llvm_ty, llvm_atomic_ty, align) in TYPES {
                #[rustfmt::skip]
                let code = format!("
atomic :: #import \"../../lib/std/atomic.mylang\";
mut static val: {mylang_ty};
s :: (expect: {mylang_ty}, new: {mylang_ty}) -> atomic.cmpxchg_strong(&mut val, expect, new, .{ord_ok}, .{ord_fail});
w :: (expect: {mylang_ty}, new: {mylang_ty}) -> atomic.cmpxchg_weak(&mut val, expect, new, .{ord_ok}, .{ord_fail});
");
                if ord_ok_idx < ord_fail_idx {
                    //test_body(code).with_prelude().error(/*TODO*/);
                } else if !valid_fail_ord {
                    test_body(code).with_prelude().error(
                        format!(
                            "no variant or associated constant `{ord_fail}` on enum type \
                             `CmpxchgFailOrdering`"
                        ),
                        substr!(ord_fail;skip=(ord_ok == ord_fail) as usize),
                    );
                } else {
                    let res = test_body(code).with_prelude().compile_no_err();
                    assert_contains!(
                        res.llvm_ir(),
                        "@cmpxchg_strong.{mylang_ty}.{ord_ok}.{ord_fail}(ptr noundef %ptr, \
                         {llvm_ty}"
                    );
                    assert_contains!(
                        res.llvm_ir(),
                        "@cmpxchg_weak.{mylang_ty}.{ord_ok}.{ord_fail}(ptr noundef %ptr, {llvm_ty}"
                    );

                    let var1 = if mylang_ty == "bool" { "0" } else { "expected" };
                    let var2 = if mylang_ty == "bool" { "1" } else { "new" };
                    assert_contains!(
                        res.llvm_ir(),
                        "cmpxchg ptr %ptr, {llvm_atomic_ty} %{var1}, {llvm_atomic_ty} %{var2} \
                         {llvm_ord_ok} {llvm_ord_fail}, align {align}"
                    );
                    assert_contains!(
                        res.llvm_ir(),
                        "cmpxchg weak ptr %ptr, {llvm_atomic_ty} %{var1}, {llvm_atomic_ty} \
                         %{var2} {llvm_ord_ok} {llvm_ord_fail}, align {align}"
                    );

                    assert_contains!(res.llvm_ir(), "attributes #0 = {{ \"alwaysinline\" }}");
                }
            }
        }
    }
}
