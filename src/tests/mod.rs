use crate::{
    ast,
    cli::{BuildArgs, TestArgsOptions},
    codegen::llvm::CodegenModuleExt,
    compiler::{BackendModule, CompileMode, CompileResult, compile_ctx},
    context::CompilationContext,
    diagnostics::{DiagnosticReporter, DiagnosticSeverity, SavedDiagnosticMessage},
    parser::{self, lexer::Span},
    ptr::Ptr,
    util::IteratorExt,
};
use std::{cell::OnceCell, fmt::Display, iter::FusedIterator};

mod alignment;
mod args;
mod array;
mod associated_consts;
mod binop;
mod call_conv_c;
mod consts;
mod defer;
mod enum_;
mod for_loop;
mod function;
mod global_scope;
mod if_;
mod index;
mod initializer;
mod logic_binop;
mod mut_checks;
mod ordering;
mod parse_number_literals;
mod ptr;
mod range;
mod slice;
mod string;
mod struct_;
mod todo;
mod union_;
mod var_decl;
mod while_loop;

const TEST_OPTIONS: TestArgsOptions = TestArgsOptions {
    debug_ast: false,
    debug_types: false,
    debug_typed_ast: false,
    llvm_optimization_level: 0,
    print_llvm_module: true,
    is_lib: true,
};

pub struct JitRunTestResult<RetTy> {
    ctx: CompilationContext,
    full_code: String,
    //compilation_out: BackendOut,
    ret: Option<RetTy>,
    //module: Option<inkwell::module::Module<'static>>,
    backend_mod: Option<BackendModule>,
    module_text: OnceCell<String>,
}

impl<RetTy> JitRunTestResult<RetTy> {
    #[track_caller]
    pub fn ok(&self) -> &RetTy {
        debug_assert_eq!(self.ret.is_some(), !self.ctx.diagnostic_reporter.do_abort_compilation());
        let Some(ret) = self.ret.as_ref() else {
            panic!("Test failed! Expected no errors")
        };
        ret
    }

    pub fn diagnostics(&self) -> &[SavedDiagnosticMessage] {
        self.ctx.diagnostic_reporter.diagnostics.as_slice()
    }

    pub fn errors(&self) -> impl FusedIterator<Item = &SavedDiagnosticMessage> {
        if self.ret.is_some() {
            panic!("Test failed! Expected compiler error, but compilation succeded.")
        }
        self.diagnostics().iter().filter(|e| e.severity.aborts_compilation())
    }

    pub fn warnings(&self) -> impl FusedIterator<Item = &SavedDiagnosticMessage> {
        self.diagnostics().iter().filter(|e| e.severity == DiagnosticSeverity::Warn)
    }

    fn read_llvm_ir(&self) -> String {
        self.backend_mod
            .as_ref()
            .unwrap()
            .codegen_module()
            .print_to_string()
            .to_string()
    }

    pub fn llvm_ir(&self) -> &str {
        self.module_text.get_or_init(|| self.read_llvm_ir())
    }

    pub fn take_llvm_ir(&mut self) -> String {
        self.module_text.take().unwrap_or_else(|| self.read_llvm_ir())
    }
}

#[track_caller]
pub fn jit_run_test<'ctx, RetTy>(code: impl Display) -> JitRunTestResult<RetTy> {
    let code = format!("test :: -> {{ {code} }};");
    jit_run_test_raw(code)
}

#[track_caller]
pub fn jit_run_test_raw<'ctx, RetTy>(code: impl ToString) -> JitRunTestResult<RetTy> {
    let code = code.to_string();
    let ctx = CompilationContext::new(BuildArgs::test_args(TEST_OPTIONS));
    ctx.0.set_test_root(Ptr::from_ref(code.as_ref())).unwrap();
    let res = compile_ctx(ctx.0, CompileMode::TestRun);
    let backend_mod = match res {
        CompileResult::ModuleForTesting(backend_module) => Some(backend_module),
        CompileResult::Err => None,
        CompileResult::Ok | CompileResult::RunErr { .. } => unreachable!(),
    };
    let ret = backend_mod.as_ref().map(|m| {
        m.codegen_module()
            .jit_run_fn::<RetTy>("test", inkwell::OptimizationLevel::None)
            .unwrap()
    });
    JitRunTestResult { ret, ctx, full_code: code, backend_mod, module_text: OnceCell::new() }
}

#[track_caller]
pub fn test_compile_err(
    code: impl Display,
    expected_msg_start: &str,
    expected_span: impl FnOnce(&str) -> TestSpan,
) {
    let code = format!("test :: -> {{ {code} }};");
    test_compile_err_raw(code, expected_msg_start, expected_span)
}

#[track_caller]
pub fn test_compile_err_raw(
    code: impl ToString,
    expected_msg_start: &str,
    expected_span: impl FnOnce(&str) -> TestSpan,
) {
    let code = code.to_string();
    let ctx = CompilationContext::new(BuildArgs::test_args(TEST_OPTIONS));
    ctx.0.set_test_root(Ptr::from_ref(code.as_ref())).unwrap();
    compile_ctx(ctx.0, CompileMode::TestRun);
    let res = JitRunTestResult::<()> {
        ret: None,
        ctx,
        full_code: code,
        backend_mod: None,
        module_text: OnceCell::new(),
    };
    let err = res
        .errors()
        .one()
        .unwrap_or_else(|e| panic!("Test failed! Compiler didn't emit exactly one error ({e:?})"));
    debug_assert_eq!(err.severity, DiagnosticSeverity::Error);
    debug_assert_eq!(err.msg.as_ref(), expected_msg_start, "incorrect compiler error");
    debug_assert_eq!(err.span, expected_span(&res.full_code));
}

pub fn test_parse(code: &str) -> TestParseRes {
    let ctx = CompilationContext::new(BuildArgs::test_args(TEST_OPTIONS));
    ctx.0.set_test_root(Ptr::from_ref(code.as_ref())).unwrap();
    let stmts = parser::parse_files_in_ctx(ctx.0);
    TestParseRes { ctx, stmts }
}

pub struct TestParseRes {
    #[allow(unused)]
    ctx: CompilationContext,
    stmts: Vec<Ptr<ast::Ast>>,
}

impl TestParseRes {
    pub fn errors(&self) -> impl FusedIterator<Item = &SavedDiagnosticMessage> {
        self.ctx
            .diagnostic_reporter
            .diagnostics
            .iter()
            .filter(|e| e.severity.aborts_compilation())
    }

    pub fn no_error(self) -> Self {
        assert!(self.errors().count() == 0);
        self
    }
}

#[derive(Debug)]
pub struct TestSpan(Span);

impl TestSpan {
    const ZERO: TestSpan = TestSpan(Span::ZERO);

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
        TestSpan::of_nth_substr(str, 0, substr)
    }

    pub fn of_nth_substr(str: &str, mut n: usize, substr: &str) -> TestSpan {
        let mut pos = 0;
        loop {
            let str = &str[pos..];
            pos += str.find(substr).unwrap_or_else(|| {
                panic!("The input text should have at least {} occurrences of {substr:?}", n + 1);
            });
            if n == 0 {
                break;
            }
            n -= 1;
            pos += 1;
        }
        TestSpan::with_len(pos, substr.len())
    }

    pub fn start(self) -> TestSpan {
        TestSpan(self.0.start())
    }

    pub fn start_with_len(self, len: usize) -> TestSpan {
        let mut span = self.0.start();
        span.end = span.start + len;
        TestSpan(span)
    }

    pub fn end(self) -> TestSpan {
        TestSpan(self.0.end())
    }

    pub fn after(self) -> TestSpan {
        TestSpan(self.0.after())
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

/// Checks if `llvm_ir` contains an automatically renamed duplicate of `sym_name`.
fn has_duplicate_symbol(llvm_ir: &str, mut sym_name: &str) -> bool {
    if sym_name.ends_with('\"') {
        sym_name = &sym_name[..sym_name.len() - 1];
    }

    llvm_ir.match_indices(sym_name).any(|(idx, _)| {
        llvm_ir.as_bytes()[idx + sym_name.len()] == b'.'
            && llvm_ir.as_bytes()[idx + sym_name.len() + 1].is_ascii_digit()
    })
}
