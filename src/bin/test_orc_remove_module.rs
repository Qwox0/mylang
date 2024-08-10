use inkwell::{
    builder::Builder,
    context::Context,
    llvm_sys::{
        core::LLVMDisposeModule, execution_engine::LLVMRemoveModule,
        orc2::LLVMOrcJITDylibCreateResourceTracker,
    },
    module::Module,
    types::BasicMetadataTypeEnum,
    values::FunctionValue,
    OptimizationLevel,
};
use std::mem;

fn create_my_fn_proto<'ctx>(context: &'ctx Context, module: &Module<'ctx>) -> FunctionValue<'ctx> {
    let ret_type = context.f64_type();
    let args_types = [BasicMetadataTypeEnum::FloatType(context.f64_type())];
    let fn_type = ret_type.fn_type(&args_types, false);
    let fn_val = module.add_function("my_fn", fn_type, None);

    /*
    for (idx, param) in fn_val.get_param_iter().enumerate() {
        param.set_name(&code[params[idx].0.span])
    }
    */

    fn_val
}

/*
fn create_my_fn<'ctx>(context: &'ctx Context, module: &Module<'ctx>, builder: Builder<'ctx>) {
    let func = create_my_fn_proto("my_fn", context, module)
    let entry = context.append_basic_block(func, "entry");
    builder.position_at_end(entry);

    self.variables.reserve(params.len());

    for (idx, param) in func.get_param_iter().enumerate() {
        let pname = &code[params[idx].0.span];

        #[cfg(feature = "use_ptr_values")]
        {
            let alloca = self.create_entry_block_alloca(func, pname);
            self.builder.build_store(alloca, param)?;
            self.variables.insert(pname.to_string(), alloca);
        }

        #[cfg(not(feature = "use_ptr_values"))]
        {
            let alloca = param.as_any_value_enum();
            self.variables.insert(pname.to_string(), alloca);
        }
    }

    let body = self.compile_expr_to_float(body, code)?;
    self.builder.build_return(Some(&body))?;

    if func.verify(true) {
        Ok(func)
    } else {
        unsafe { func.delete() };
        Err(CError::InvalidGeneratedFunction)
    }
}
*/

fn create_expr<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    val: f64,
) {
    let ret_type = context.f64_type();
    let args_types = [BasicMetadataTypeEnum::FloatType(context.f64_type())];
    let fn_type = ret_type.fn_type(&args_types, false);
    let func = module.add_function("__expr", fn_type, None);
    let entry = context.append_basic_block(func, "entry");
    builder.position_at_end(entry);
    let body = {
        let func = module
            .get_function("my_fn")
            .unwrap_or_else(|| create_my_fn_proto(&context, &module));

        let args = [context.f64_type().const_float_from_string(&val.to_string()).into()];

        builder
            .build_call(func, &args, "calltmp")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value()
    };
    builder.build_return(Some(&body)).unwrap();
    if !func.verify(true) {
        unsafe { func.delete() };
        panic!("CError::InvalidGeneratedFunction")
    }

    println!("func: {:?}", func);
}

fn main_inkwell_jit() {
    let context = Context::create();
    let builder = context.create_builder();
    let module = context.create_module("mymodule");

    let func = create_my_fn_proto(&context, &module);
    let entry = context.append_basic_block(func, "entry");
    builder.position_at_end(entry);
    let body = {
        let lhs = func.get_first_param().unwrap().into_float_value();
        let rhs = context.f64_type().const_float_from_string("2");

        builder.build_float_mul(lhs, rhs, "tmpMul").unwrap()
    };
    builder.build_return(Some(&body)).unwrap();
    if !func.verify(true) {
        unsafe { func.delete() };
        panic!("CError::InvalidGeneratedFunction")
    }

    let jit = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();

    let module = context.create_module("mymodule");
    jit.add_module(&module).unwrap();
    create_expr(&context, &module, &builder, 3.0);

    let a = unsafe { jit.get_function::<unsafe extern "C" fn() -> f64>("__expr").unwrap().call() };
    println!("=> {}", a);

    jit.remove_module(&module).unwrap();

    unsafe { func.delete() };
    unsafe { LLVMDisposeModule(module.as_mut_ptr()) };
    mem::forget(module);
    jit.free_fn_machine_code(func);

    let module = context.create_module("mymodule");
    jit.add_module(&module).unwrap();
    create_expr(&context, &module, &builder, 10.0);

    let a = unsafe { jit.get_function::<unsafe extern "C" fn() -> f64>("__expr").unwrap().call() };
    println!("=> {}", a);
}

fn main() {
    println!("inkwell jit",);
    main_inkwell_jit();
}
