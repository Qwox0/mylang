use crate::{
    ast,
    cli::{BuildArgs, TestArgsOptions},
    codegen::llvm::CodegenModuleExt,
    compiler::{BackendModule, CompileMode, CompileResult, compile_ctx},
    context::CompilationContext,
    diagnostics::{DiagnosticReporter, DiagnosticSeverity},
    parser::{self, lexer::Span},
    ptr::Ptr,
};
use std::{
    cell::OnceCell,
    fmt::{self, Display},
    thread,
};

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
mod type_joining;
mod union_;
mod var_decl;
mod vararg;
mod while_loop;

const TEST_OPTIONS: TestArgsOptions = TestArgsOptions {
    debug_ast: false,
    debug_types: false,
    debug_typed_ast: false,
    llvm_optimization_level: 0,
    print_llvm_module: true,
    is_lib: true,
};

fn test(code: impl ToString) -> NewTest {
    NewTest { code: code.to_string() }
}

fn test_body(code_body: impl Display) -> NewTest {
    test(format!("test :: -> {{ {code_body} }};"))
}

fn test_parse(code: impl ToString) -> TestResult<Parsed> {
    let res = test(code).prepare();
    parser::parse_files_in_ctx(res.ctx.ctx.0);
    TestResult { data: Parsed, ..res }
}

fn test_analyzed_struct(struct_code: &str) -> TestResult<Ptr<crate::ast::StructDef>> {
    let res = test(format!("_ :: {struct_code}")).compile_no_err();
    let struct_def = res.stmts()[0]
        .downcast::<crate::ast::Decl>()
        .init
        .unwrap()
        .downcast::<crate::ast::StructDef>();
    TestResult { data: struct_def, ..res }
}

struct NewTest {
    code: String,
}

struct TestCtx {
    ctx: CompilationContext,
    diag_idx: usize,
}

impl Drop for TestCtx {
    #[track_caller]
    fn drop(&mut self) {
        if thread::panicking() {
            return; // Test failed already
        }
        let unhandled_errors = self.ctx.diagnostic_reporter.diagnostics[self.diag_idx..]
            .iter()
            .filter(|diag| diag.severity.aborts_compilation())
            .count();
        if unhandled_errors > 0 {
            panic!("Test failed! Got {unhandled_errors} unexpected compiler errors!");
        }
    }
}

struct TestResult<Kind> {
    ctx: TestCtx,
    code: String,
    data: Kind,
}

struct Parsed;
struct Compiled {
    backend_mod: BackendModule,
    module_text: OnceCell<String>,
}
struct Ok<RetTy> {
    c: Compiled,
    ret: RetTy,
}
struct Err;

impl NewTest {
    fn prepare(self) -> TestResult<()> {
        let ctx = CompilationContext::new(BuildArgs::test_args(TEST_OPTIONS));
        ctx.0.set_test_root(Ptr::from_ref(self.code.as_ref())).unwrap();
        TestResult { ctx: TestCtx { ctx, diag_idx: 0 }, code: self.code, data: () }
    }

    fn compile(self) -> TestResult<CompileResult> {
        let _self = self.prepare();
        TestResult { data: compile_ctx(_self.ctx.ctx.0, CompileMode::TestRun), .._self }
    }

    #[track_caller]
    fn compile_no_err(self) -> TestResult<Compiled> {
        let res = self.compile();
        let CompileResult::ModuleForTesting(backend_mod) = res.data else {
            panic!("Test failed! Expected no compiler errors.")
        };
        TestResult { data: Compiled { backend_mod, module_text: OnceCell::new() }, ..res }
    }

    #[track_caller]
    fn _ok<RetTy>(self) -> TestResult<Ok<RetTy>> {
        let res = self.compile_no_err();
        let ret = res
            .data
            .backend_mod
            .codegen_module()
            .jit_run_fn::<RetTy>("test", inkwell::OptimizationLevel::None)
            .unwrap();
        TestResult { data: Ok { ret, c: res.data }, ..res }
    }

    #[track_caller]
    fn ok<RetTy: PartialEq + fmt::Debug>(self, expected: RetTy) -> TestResult<Ok<RetTy>> {
        let res = self._ok();
        assert_eq!(res.data.ret, expected);
        res
    }

    #[track_caller]
    fn get_out<RetTy: Copy>(self) -> RetTy {
        self._ok().data.ret
    }

    #[track_caller]
    fn error(self, msg: impl AsRef<str>, span: impl FnOnce(&str) -> TestSpan) -> TestResult<Err> {
        let res = self.compile();
        if !(matches!(res.data, CompileResult::Err)
            && res.ctx.ctx.diagnostic_reporter.do_abort_compilation())
        {
            panic!("Test failed! Expected compiler error, but compilation succeded.")
        }
        res.error(msg, span)
    }
}

impl<Res> TestResult<Res> {
    fn read_llvm_ir(&self) -> String
    where Res: AsRef<Compiled> {
        self.data.as_ref().backend_mod.codegen_module().print_to_string().to_string()
    }

    pub fn llvm_ir(&self) -> &str
    where Res: AsRef<Compiled> {
        self.data.as_ref().module_text.get_or_init(|| self.read_llvm_ir())
    }

    pub fn take_llvm_ir(&mut self) -> String
    where Res: AsRef<Compiled> + AsMut<Compiled> {
        self.data.as_mut().module_text.take().unwrap_or_else(|| self.read_llvm_ir())
    }

    #[track_caller]
    fn check_next_diag(
        &mut self,
        sev: DiagnosticSeverity,
        msg: Option<&str>,
        span: impl FnOnce(&str) -> TestSpan,
    ) {
        let ctx = &mut self.ctx;
        let Some(diag) = ctx.ctx.diagnostic_reporter.diagnostics.get(ctx.diag_idx) else {
            panic!("Expected at least {} diagnostics", ctx.diag_idx + 1)
        };
        ctx.diag_idx += 1;
        assert_eq!(diag.severity, sev);
        if let Some(msg) = msg {
            assert_eq!(diag.msg.as_ref(), msg, "incorrect diagnostic message");
        }
        assert_eq!(diag.span, span(&self.code));
    }

    #[track_caller]
    pub fn error<'m>(
        mut self,
        msg: impl AsRef<str>,
        span: impl FnOnce(&str) -> TestSpan,
    ) -> TestResult<Err> {
        self.check_next_diag(DiagnosticSeverity::Error, Some(msg.as_ref()), span);
        TestResult { data: Err, ..self }
    }

    #[track_caller]
    pub fn warn<'m>(
        mut self,
        msg: impl Optional<&'m str>,
        span: impl FnOnce(&str) -> TestSpan,
    ) -> Self {
        self.check_next_diag(DiagnosticSeverity::Warn, msg.as_option(), span);
        self
    }

    #[track_caller]
    pub fn info<'m>(
        mut self,
        msg: impl Optional<&'m str>,
        span: impl FnOnce(&str) -> TestSpan,
    ) -> Self {
        self.check_next_diag(DiagnosticSeverity::Info, msg.as_option(), span);
        self
    }

    pub fn stmts(&self) -> &[Ptr<ast::Ast>] {
        &self.ctx.ctx.stmts
    }
}

trait Optional<T> {
    fn as_option(self) -> Option<T>;
}

impl<T> Optional<T> for T {
    fn as_option(self) -> Option<T> {
        Some(self)
    }
}

impl<T> Optional<T> for Option<T> {
    fn as_option(self) -> Option<T> {
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

    #[track_caller]
    pub fn of_substr(str: &str, substr: &str, mut n: usize) -> TestSpan {
        let mut pos = 0;
        loop {
            let str = &str[pos..];
            let Some(start) = str.find(substr) else {
                panic!("The input text should have at least {} occurrences of {substr:?}", n + 1);
            };
            pos += start;
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

macro_rules! substr {
    ($substr:expr $(; skip=$skip:expr)? $(; . $method:ident($($arg:expr),*))?) => {
        &|code: &str| $crate::tests::TestSpan::of_substr(code, $substr, 0 $(+ $skip)?)$(.$method($($arg),*))?
    };
}
pub(self) use substr;

impl AsRef<Compiled> for Compiled {
    fn as_ref(&self) -> &Compiled {
        self
    }
}

impl AsMut<Compiled> for Compiled {
    fn as_mut(&mut self) -> &mut Compiled {
        self
    }
}

impl<RetTy> AsRef<Compiled> for Ok<RetTy> {
    fn as_ref(&self) -> &Compiled {
        &self.c
    }
}

impl<RetTy> AsMut<Compiled> for Ok<RetTy> {
    fn as_mut(&mut self) -> &mut Compiled {
        &mut self.c
    }
}
