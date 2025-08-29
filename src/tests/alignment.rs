use crate::tests::test_body;

#[test]
fn correct_alignment_on_store_instruction() {
    let res = test_body("mut a := 1;").ok(());
    let store_instructions = res
        .llvm_ir()
        .lines()
        .filter(|l| l.trim_start().starts_with("store"))
        .collect::<Vec<_>>();
    assert_eq!(
        store_instructions.len(),
        1,
        "the test code should result in llvm ir with a store instruction"
    );
    assert!(store_instructions[0].ends_with("align 8"))
}
