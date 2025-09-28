use inkwell::{
    context::{AsContextRef, Context},
    llvm_sys::{
        core::*,
        prelude::{LLVMTypeRef, LLVMValueRef},
    },
    types::{AsTypeRef, FunctionType, StructType},
    values::StructValue,
};

pub fn new_raw_const_struct<'ctx>(
    struct_ty: StructType<'ctx>,
    args: &mut [LLVMValueRef],
) -> StructValue<'ctx> {
    unsafe {
        StructValue::new(LLVMConstNamedStruct(
            struct_ty.as_type_ref(),
            args.as_mut_ptr(),
            args.len() as u32,
        ))
    }
}

pub fn new_fn_type<'ctx>(
    ret_ty: LLVMTypeRef,
    param_types: &mut [LLVMTypeRef],
    is_var_args: bool,
) -> FunctionType<'ctx> {
    unsafe {
        FunctionType::new(LLVMFunctionType(
            ret_ty,
            param_types.as_mut_ptr(),
            param_types.len() as u32,
            is_var_args as i32,
        ))
    }
}

pub fn new_anon_struct_type<'ctx>(
    ctx: &'ctx Context,
    field_types: &mut [LLVMTypeRef],
    packed: bool,
) -> StructType<'ctx> {
    unsafe {
        StructType::new(LLVMStructTypeInContext(
            ctx.as_ctx_ref(),
            field_types.as_mut_ptr(),
            field_types.len() as u32,
            packed as i32,
        ))
    }
}

pub fn set_struct_body<'ctx>(
    struct_ty: StructType<'ctx>,
    field_types: &mut [LLVMTypeRef],
    packed: bool,
) {
    unsafe {
        LLVMStructSetBody(
            struct_ty.as_type_ref(),
            field_types.as_mut_ptr(),
            field_types.len() as u32,
            packed as i32,
        );
    }
}
