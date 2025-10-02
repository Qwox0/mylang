use mylang::{
    cli::BuildArgs,
    compiler::{CompileMode, CompileResult},
};
use std::{
    borrow::BorrowMut,
    io::{BufRead, BufReader},
    path::Path,
    process::{Command, Stdio},
    thread,
};

fn test_file(file_path: &str) {
    std::env::set_current_dir(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests"))).unwrap();
    let args = BuildArgs {
        path: file_path.into(),
        debug_ast: false,
        quiet: true,
        debug_llvm_ir_unoptimized: true,
        ..BuildArgs::default()
    };
    let res = mylang::compiler::compile(CompileMode::Run, args);
    if !matches!(res, CompileResult::Ok) {
        panic!("Compilation of '{file_path}' failed!")
    }
}

fn path(rel_to_project_root: impl AsRef<Path>) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel_to_project_root)
}

struct Output {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn test_cmd(mut cmd: impl BorrowMut<Command>) -> Output {
    let cmd = cmd.borrow_mut();
    println!(
        "testing command '{}{}': \"\"\"",
        cmd.get_program().display(),
        cmd.get_args()
            .map(std::ffi::OsStr::display)
            .fold(String::new(), |acc, arg| acc + " " + &arg.to_string())
    );

    if cmd.get_current_dir().is_none() {
        cmd.current_dir(path("tests"));
    }

    // <https://stackoverflow.com/a/72831067>
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();

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

fn mylang<S: AsRef<std::ffi::OsStr>>(args: impl IntoIterator<Item = S>) -> Command {
    let mut cmd = Command::new("mylang");
    cmd.args(args);
    cmd
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
    let out = test_cmd(mylang(["run", "./no_main.mylang"]));
    assert!(!out.status.success());
    assert!(out.stderr.contains("Couldn't find the entry point 'main' in 'no_main.mylang'"));
}

#[test]
fn printf_vararg() {
    let out = test_cmd(mylang(["run", "./printf_vararg.mylang"]));
    assert!(out.status.success());
    assert_eq!(out.stdout, "10; 123.456000; Hello World\n");
}

#[test]
fn c_ffi_take_array_arg() {
    let dir = path("./tests/c_ffi_take_array_arg/");

    // build mylang lib
    let out =
        test_cmd(mylang(["build", "take_arr.mylang", "--out=obj", "--lib"]).current_dir(&dir));
    assert!(out.status.success());

    // build & link with `main.c`
    let out = test_cmd(
        Command::new("clang")
            .args(["-o", "./out/main", "./main.c", "./out/take_arr.o"])
            .current_dir(&dir),
    );
    assert!(out.status.success());

    // run
    let out = test_cmd(Command::new("./out/main").current_dir(&dir));
    assert!(out.status.success());
}

#[test]
fn set_impl() {
    let out = test_cmd(mylang(["check", "../lib/std/set.mylang", "--entry-point=set_test"]));
    assert!(out.status.success());
}
