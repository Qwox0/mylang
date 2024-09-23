use super::{Codegen, REPL_EXPR_ANON_FN_NAME};
use inkwell::{
    execution_engine::{ExecutionEngine, FunctionLookupError},
    llvm_sys::orc2::{
        LLVMOrcJITDylibCreateResourceTracker, LLVMOrcJITDylibRef, LLVMOrcResourceTrackerRef,
    },
    module::Module,
    OptimizationLevel,
};

#[derive(Debug, Default)]
pub struct Jit<'ctx> {
    jit: Option<ExecutionEngine<'ctx>>,
}

impl<'ctx> Jit<'ctx> {
    fn jit(&self) -> Result<&ExecutionEngine<'ctx>, JitError> {
        self.jit.as_ref().ok_or(JitError::MustAddAModuleFirst)
    }

    pub fn add_module(&mut self, module: Module<'ctx>) {
        unsafe { self.add_module_ref(&module) }
    }

    /// # SAFETY
    ///
    /// Ensure `module` isn't added a second time.
    unsafe fn add_module_ref(&mut self, module: &Module<'ctx>) {
        match self.jit.as_ref() {
            Some(jit) => jit.add_module(module).expect("module should not be in execution engine"),
            None => {
                self.jit
                    .insert(module.create_jit_execution_engine(OptimizationLevel::None).unwrap());
            },
        }
    }

    pub fn take_module_from<'alloc>(&mut self, compiler: &mut Codegen<'ctx, 'alloc>) {
        compiler.move_module_to(self)
    }

    pub fn run(&mut self, fn_name: &str) -> Result<f64, JitError> {
        Ok(unsafe { self.jit()?.get_function::<unsafe extern "C" fn() -> f64>(fn_name)?.call() })
    }

    pub fn run_repl_expr(&mut self, module: Module<'ctx>) -> Result<f64, JitError> {
        //println!("0 {:?}", unsafe { self.jit()? .get_function::<unsafe extern "C"
        // fn() -> f64>(REPL_EXPR_ANON_FN_NAME) });
        unsafe { self.add_module_ref(&module) };
        let a = self.jit()?.as_mut_ptr();
        let out = self.run(REPL_EXPR_ANON_FN_NAME)?;
        //println!("b {:?}", unsafe { self.add_module_ref(&module) });
        let jit = self.jit()?;
        //println!("1 {:?}", unsafe { jit.get_function::<unsafe extern "C" fn() ->
        // f64>(REPL_EXPR_ANON_FN_NAME) });
        let a = jit.remove_module(&module);
        //println!("2 {:?}", unsafe { jit.get_function::<unsafe extern "C" fn() ->
        // f64>(REPL_EXPR_ANON_FN_NAME) });
        //jit.add_module()
        //println!("c {:?}", unsafe { self.add_module_ref(&module) });
        println!("3 {:?}", a);
        let out2 = self.run(REPL_EXPR_ANON_FN_NAME)?;
        println!("4 {:?}", out2);
        let jit = self.jit()?;
        let a = jit.remove_module(&module);
        println!("5 {:?}", a);
        let out2 = self.run(REPL_EXPR_ANON_FN_NAME)?;
        println!("6 {:?}", out2);
        Ok(out)
    }
}

#[derive(Debug)]
pub enum JitError {
    FunctionLookupError(FunctionLookupError),
    MustAddAModuleFirst,
}

impl From<FunctionLookupError> for JitError {
    fn from(e: FunctionLookupError) -> Self {
        JitError::FunctionLookupError(e)
    }
}
