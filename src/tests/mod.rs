use crate::{
    cli::{BuildArgs, OutKind},
    codegen::llvm::CodegenModuleExt,
    compiler::{BackendModule, CompileMode, CompileResult, compile_file},
    context::CompilationContext,
    diagnostic_reporter::SavedDiagnosticMessage,
    parser::lexer::Code,
    ptr::Ptr,
    source_file::SourceFile,
};
use std::{
    fmt::Display,
    path::{Path, PathBuf},
};

mod alignment;
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
mod while_loop;

const DEBUG_AST: bool = true;
const DEBUG_TYPES: bool = false;
const DEBUG_TYPED_AST: bool = false;
const LLVM_OPTIMIZATION_LEVEL: u8 = 0;
const PRINT_LLVM_MODULE: bool = false;

pub struct JitRunTestResult<RetTy> {
    ctx: CompilationContext,
    //compilation_out: BackendOut,
    ret: Option<RetTy>,
    //module: Option<inkwell::module::Module<'static>>,
    backend_mod: Option<BackendModule>,
}

impl<RetTy> JitRunTestResult<RetTy> {
    pub fn ok(&self) -> &RetTy {
        debug_assert_eq!(
            self.ret.is_some(),
            self.ctx.diagnostic_reporter.diagnostics.as_slice().is_empty()
        );
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
        debug_assert!(errors.len() == 1);
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

pub fn jit_run_test_raw<'ctx, RetTy>(code: impl AsRef<Code>) -> JitRunTestResult<RetTy> {
    let ctx = CompilationContext::new();
    let test_file = test_file_mock(code.as_ref());
    let res = compile_file(ctx.0, test_file, CompileMode::TestRun, &BuildArgs {
        path: PathBuf::default(), // irrelevant
        optimization_level: LLVM_OPTIMIZATION_LEVEL,
        target_triple: None,
        out: OutKind::None,
        no_prelude: true,
        print_compile_time: false,
        debug_ast: DEBUG_AST,
        debug_types: DEBUG_TYPES,
        debug_typed_ast: DEBUG_TYPED_AST,
        debug_functions: false,
        debug_llvm_ir_unoptimized: false,
        debug_llvm_ir_optimized: PRINT_LLVM_MODULE,
    });
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
    JitRunTestResult { ret, ctx, backend_mod }
}
