use mylang::{
    cli::BuildArgs,
    compiler::{CompileMode, CompileResult},
    context::CompilationContext,
    ptr::Ptr,
};
use std::{
    path::Path,
    process::{Command, Output, Stdio},
};

fn test_file(file_path: &str) {
    std::env::set_current_dir(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))).unwrap();
    let ctx = CompilationContext::new();
    let mut args = BuildArgs {
        path: file_path.into(),
        debug_ast: false,
        quiet: true,
        debug_llvm_ir_unoptimized: true,
        ..BuildArgs::default()
    };
    let res = mylang::compiler::compile(Ptr::from_ref(&ctx), CompileMode::Run, &mut args);
    if !matches!(res, CompileResult::Ok) {
        panic!("Compilation of '{file_path}' failed!")
    }
}

fn test_shell_script(sh_path: &str) -> Output {
    let test_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(sh_path);
    println!("testing shell script {:?}: \"\"\"", sh_path);
    let out = Command::new("bash")
        .arg(test_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .unwrap();
    println!("\"\"\"");
    assert!(out.status.success());
    out
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
file_test! { default_args }
file_test! { named_call_args }

#[test]
fn error_no_main() {
    std::env::set_current_dir(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))).unwrap();
    let ctx = CompilationContext::new();
    let mut args =
        BuildArgs { path: "./no_main.mylang".into(), quiet: true, ..BuildArgs::default() };
    let res = mylang::compiler::compile(Ptr::from_ref(&ctx), CompileMode::Run, &mut args);
    assert!(matches!(res, CompileResult::Err));
    // TODO: check the error message
}

#[test]
fn c_ffi_take_array_arg() {
    let out = test_shell_script("tests/c_ffi_take_array_arg/run.sh");
    assert!(
        String::from_utf8_lossy(&out.stdout)
            .trim_end()
            .ends_with("got array: [1,2,3,4,]")
    );
}
