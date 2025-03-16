use mylang::{
    cli::BuildArgs,
    compiler::{CompileMode, CompileResult},
    context::CompilationContext,
    ptr::Ptr,
};
use std::path::Path;

fn test_file(file_path: &str) {
    std::env::set_current_dir(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))).unwrap();
    let ctx = CompilationContext::new();
    let mut args = BuildArgs {
        path: file_path.into(),
        print_compile_time: false,
        //debug_llvm_ir_unoptimized: true,
        ..BuildArgs::default()
    };
    let res = mylang::compiler::compile(Ptr::from_ref(&ctx), CompileMode::Run, &mut args);
    //assert!(res.ok());
    assert!(matches!(res, CompileResult::Ok))
}

macro_rules! file_test {
    ($file:ident) => {
        #[test]
        fn $file() {
            test_file(concat!(stringify!($file), ".mylang"))
        }
    };
}

file_test! { import }
file_test! { function_call }

#[test]
fn error_no_main() {
    std::env::set_current_dir(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))).unwrap();
    let ctx = CompilationContext::new();
    let mut args = BuildArgs {
        path: "./no_main.mylang".into(),
        print_compile_time: false,
        ..BuildArgs::default()
    };
    let res = mylang::compiler::compile(Ptr::from_ref(&ctx), CompileMode::Run, &mut args);
    assert!(matches!(res, CompileResult::Err));
    // TODO: check the error message
}
