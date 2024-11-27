use crate::tests::jit_run_test;

#[test]
fn correct_alignment_on_store_instruction() {
    let (_, llvm_module_text) = jit_run_test!("mut a := 1;" => (), llvm_module).unwrap();
    let store_instructions = llvm_module_text
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
