use crate::{
    cli::{BuildArgs, TestArgsOptions},
    codegen::llvm::CodegenModuleExt,
    compiler::{BackendModule, CompileMode, CompileResult, compile_file},
    context::CompilationContext,
    diagnostic_reporter::{DiagnosticReporter, DiagnosticSeverity, SavedDiagnosticMessage},
    parser::lexer::{Code, Span},
    ptr::Ptr,
    source_file::SourceFile,
};
use std::{fmt::Display, path::Path};

mod alignment;
mod args;
mod associated_variables;
mod binop;
mod call_conv_c;
mod defer;
mod enum_;
mod for_loop;
mod function;
mod if_;
mod initializer;
mod logic_binop;
mod parse_array;
mod parse_function;
mod ptr;
mod slice;
mod string;
mod struct_;
mod todo;
mod union_;
mod var_decl;
mod while_loop;

const TEST_OPTIONS: TestArgsOptions = TestArgsOptions {
    debug_ast: true,
    debug_types: false,
    debug_typed_ast: false,
    llvm_optimization_level: 0,
    print_llvm_module: false,
    entry_point: "test",
};

pub struct JitRunTestResult<RetTy> {
    ctx: CompilationContext,
    full_code: String,
    //compilation_out: BackendOut,
    ret: Option<RetTy>,
    //module: Option<inkwell::module::Module<'static>>,
    backend_mod: Option<BackendModule>,
}

impl<RetTy> JitRunTestResult<RetTy> {
    pub fn ok(&self) -> &RetTy {
        debug_assert_eq!(self.ret.is_some(), !self.ctx.diagnostic_reporter.do_abort_compilation());
        self.ret.as_ref().unwrap_or_else(|| panic!("Test failed! Expected no errors"))
    }

    pub fn err(&self) -> &[SavedDiagnosticMessage] {
        let diag = self.ctx.diagnostic_reporter.diagnostics.as_slice();
        debug_assert_eq!(self.ret.is_some(), diag.is_empty());
        if self.ret.is_some() {
            panic!("Test failed! Expected compiler error, but compilation succeded.")
        }
        diag
    }

    pub fn one_err(&self) -> &SavedDiagnosticMessage {
        let errors = self.err();
        debug_assert!(errors.iter().filter(|e| e.severity.aborts_compilation()).count() == 1);
        &errors[0]
    }

    pub fn module_text(&self) -> Option<String> {
        self.backend_mod
            .as_ref()
            .map(|o| o.codegen_module().print_to_string().to_string())
    }
}

pub fn test_file_mock(code: &Code) -> SourceFile {
    SourceFile::new(Ptr::from_ref(Path::new("test.mylang")), Ptr::from_ref(code))
}

pub fn jit_run_test<'ctx, RetTy>(code: impl Display) -> JitRunTestResult<RetTy> {
    let code = format!("test :: -> {{ {code} }};");
    jit_run_test_raw(code)
}

pub fn jit_run_test_raw<'ctx, RetTy>(code: impl ToString) -> JitRunTestResult<RetTy> {
    let code = code.to_string();
    let ctx = CompilationContext::new();
    let test_file = test_file_mock(code.as_ref());
    let res =
        compile_file(ctx.0, test_file, CompileMode::TestRun, &BuildArgs::test_args(TEST_OPTIONS));
    let backend_mod = match res {
        CompileResult::ModuleForTesting(backend_module) => Some(backend_module),
        CompileResult::Err => None,
        CompileResult::Ok | CompileResult::RunErr { .. } => unreachable!(),
    };
    let ret = backend_mod.as_ref().map(|m| {
        m.codegen_module()
            .jit_run_fn::<RetTy>(TEST_OPTIONS.entry_point, inkwell::OptimizationLevel::None)
            .unwrap()
    });
    JitRunTestResult { ret, ctx, full_code: code, backend_mod }
}

#[allow(unused)]
pub fn test_compile_err(
    code: impl Display,
    expected_msg_start: &str,
    expected_span: impl FnOnce(&str) -> TestSpan,
) {
    let code = format!("test :: -> {{ {code} }};");
    test_compile_err_raw(code, expected_msg_start, expected_span)
}

pub fn test_compile_err_raw(
    code: impl ToString,
    expected_msg_start: &str,
    expected_span: impl FnOnce(&str) -> TestSpan,
) {
    let code = code.to_string();
    let ctx = CompilationContext::new();
    let test_file = test_file_mock(code.as_ref());
    compile_file(ctx.0, test_file, CompileMode::TestRun, &BuildArgs::test_args(TEST_OPTIONS));
    let res = JitRunTestResult::<()> { ret: None, ctx, full_code: code, backend_mod: None };
    let err = res.one_err();
    debug_assert_eq!(err.severity, DiagnosticSeverity::Error);
    debug_assert!(
        err.msg.starts_with(expected_msg_start),
        "{:?} doesn't start with {:?}",
        err.msg,
        expected_msg_start
    );
    debug_assert_eq!(err.span, expected_span(&res.full_code));
}

#[derive(Debug)]
pub struct TestSpan(Span);

impl TestSpan {
    pub fn new(start: usize, end: usize) -> TestSpan {
        TestSpan(Span::new(start..end, None))
    }

    pub fn with_len(start: usize, len: usize) -> TestSpan {
        TestSpan::new(start, start + len)
    }

    pub fn pos(pos: usize) -> TestSpan {
        TestSpan::new(pos, pos + 1)
    }

    pub fn of_substr(str: &str, substr: &str) -> TestSpan {
        TestSpan::with_len(str.find(substr).unwrap(), substr.len())
    }

    pub fn start(self) -> TestSpan {
        TestSpan(self.0.start_pos())
    }

    pub fn end(self) -> TestSpan {
        TestSpan(self.0.end())
    }
}

impl PartialEq<Span> for TestSpan {
    fn eq(&self, other: &Span) -> bool {
        self.0.range() == other.range()
    }
}

impl PartialEq<TestSpan> for Span {
    fn eq(&self, other: &TestSpan) -> bool {
        other == self
    }
}
