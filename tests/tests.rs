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

fn test_cmd(cmd: &mut Command) -> Output {
    println!(
        "testing command '{}{}': \"\"\"",
        cmd.get_program().display(),
        cmd.get_args()
            .map(std::ffi::OsStr::display)
            .fold(String::new(), |acc, arg| acc + " " + &arg.to_string())
    );

    // <https://stackoverflow.com/a/72831067>
    let mut child = cmd
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests"))
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
    let out = test_cmd(Command::new("mylang").arg("run").arg("./no_main.mylang"));
    assert!(!out.status.success());
    assert!(out.stderr.contains("Couldn't find the entry point 'main' in 'no_main.mylang'"));
}

#[test]
fn c_ffi_take_array_arg() {
    let out = test_cmd(Command::new("bash").arg("./c_ffi_take_array_arg/run.sh"));
    assert!(out.status.success());
    assert!(out.stdout.trim_end().ends_with("got array: [1,2,3,4,]"));
}
