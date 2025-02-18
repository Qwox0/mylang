use crate::{
    cli::{BuildArgs, OutKind},
    codegen::llvm::CodegenModuleExt,
    compiler::{BackendOut, CompileMode, compile_ctx},
    context::CompilationContext,
    diagnostic_reporter::SavedDiagnosticMessage,
    parser::lexer::Code,
    ptr::Ptr,
};
use std::{fmt::Display, path::PathBuf, str::FromStr};

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

const DEBUG_TOKENS: bool = false;
const DEBUG_AST: bool = true;
const DEBUG_TYPES: bool = true;
const DEBUG_TYPED_AST: bool = false;
const LLVM_OPTIMIZATION_LEVEL: u8 = 0;
const PRINT_LLVM_MODULE: bool = false;

pub struct JitRunTestResult<RetTy> {
    ctx: CompilationContext,
    //compilation_out: BackendOut,
    ret: Option<RetTy>,
    //module: Option<inkwell::module::Module<'static>>,
    backend_out: Option<BackendOut>,
}

impl<RetTy> JitRunTestResult<RetTy> {
    pub fn ok(&self) -> &RetTy {
        debug_assert_eq!(
            self.ret.is_some(),
            self.ctx.diagnostic_reporter.diagnostics.borrow().as_slice().is_empty()
        );
        self.ret.as_ref().unwrap_or_else(|| panic!("Test failed! Expected no errors"))
    }

    pub fn err(&self) -> Vec<SavedDiagnosticMessage> {
        let diag = self.ctx.diagnostic_reporter.diagnostics.take();
        debug_assert_eq!(self.ret.is_some(), diag.is_empty());
        if self.ret.is_some() {
            panic!("Test failed! Expected compiler error, but compilation succeded.")
        }
        diag
    }

    pub fn module_text(&self) -> Option<String> {
        self.backend_out
            .as_ref()
            .map(|o| o.codegen_module().print_to_string().to_string())
    }
}

pub fn jit_run_test<'ctx, RetTy>(code: impl Display) -> JitRunTestResult<RetTy> {
    let code = format!("test :: -> {{ {code} }};");
    jit_run_test_raw(code)
}

pub fn jit_run_test_raw<'ctx, RetTy>(code: impl AsRef<Code>) -> JitRunTestResult<RetTy> {
    let mut ctx = CompilationContext::new(Ptr::from_ref(code.as_ref()));
    let res = compile_ctx(&mut ctx, CompileMode::Build, &BuildArgs {
        path: PathBuf::from_str("test").unwrap(),
        optimization_level: LLVM_OPTIMIZATION_LEVEL,
        target_triple: None,
        out: OutKind::None,
        no_prelude: true,
        print_compile_time: false,
        debug_tokens: DEBUG_TOKENS,
        debug_ast: DEBUG_AST,
        debug_types: DEBUG_TYPES,
        debug_typed_ast: DEBUG_TYPED_AST,
        debug_functions: false,
        debug_llvm_ir_unoptimized: false,
        debug_llvm_ir_optimized: PRINT_LLVM_MODULE,
    });
    let backend_out = res.backend_out;
    debug_assert!(!res.ok || backend_out.is_some());
    let ret = backend_out.as_ref().map(|o| {
        o.codegen_module()
            .jit_run_fn::<RetTy>("test", inkwell::OptimizationLevel::None)
            .unwrap()
    });
    JitRunTestResult { ret, ctx, backend_out }
}
