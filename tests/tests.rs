use mylang::{
    cli::BuildArgs,
    compiler::{CompileMode, CompileResult},
    context::CompilationContext,
    ptr::Ptr,
};
use std::{
    io::{BufRead, BufReader},
    path::Path,
    process::{Command, Stdio},
    thread,
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

#[allow(unused)]
struct Output {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn test_shell_script(sh_path: &str) -> Output {
    let test_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(sh_path);
    println!("testing shell script {:?}: \"\"\"", sh_path);

    // <https://stackoverflow.com/a/72831067>
    let mut child = Command::new("bash")
        .arg(test_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let child_stdout = child.stdout.take().expect("Internal error, could not take stdout");
    let child_stderr = child.stderr.take().expect("Internal error, could not take stderr");

    let (stdout_tx, stdout_rx) = std::sync::mpsc::channel();
    let (stderr_tx, stderr_rx) = std::sync::mpsc::channel();

    let stdout_thread = thread::spawn(move || {
        for line in BufReader::new(child_stdout).lines() {
            let line = line.unwrap();
            println!("{}", line);
            stdout_tx.send(line).unwrap();
        }
    });

    let stderr_thread = thread::spawn(move || {
        for line in BufReader::new(child_stderr).lines() {
            let line = line.unwrap();
            eprintln!("{}", line);
            stderr_tx.send(line).unwrap();
        }
    });

    let status = child.wait().expect("Internal error, failed to wait on child");
    stdout_thread.join().unwrap();
    stderr_thread.join().unwrap();

    println!("\"\"\"");

    let join_lines = |acc, line: String| acc + line.as_str() + "\n";
    let stdout = stdout_rx.into_iter().fold(String::new(), join_lines);
    let stderr = stderr_rx.into_iter().fold(String::new(), join_lines);

    assert!(status.success());
    Output { status, stdout, stderr }
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
    assert!(out.stdout.trim_end().ends_with("got array: [1,2,3,4,]"));
}
