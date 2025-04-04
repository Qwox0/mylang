use crate::{
    ast::{
        self, Ast, AstEnum, AstKind, BinOpKind, DeclList, DeclListExt, OptionTypeExt, TypeEnum,
        UnaryOpKind, UpcastToAst, is_pos_arg,
    },
    context::primitives,
    display_code::display,
    literals::replace_escape_chars,
    ptr::Ptr,
    scoped_stack::ScopedStack,
    symbol_table::CodegenSymbolTable,
    type_::{RangeKind, enum_alignment, ty_match, union_size},
    util::{
        self, UnwrapDebug, forget_lifetime, get_aligned_offset, get_padding, is_simple_enum,
        panic_debug, unreachable_debug,
    },
};
pub use inkwell::targets::TargetMachine;
use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    attributes::Attribute,
    basic_block::BasicBlock,
    builder::{Builder, BuilderError},
    context::Context,
    execution_engine::FunctionLookupError,
    llvm_sys::{LLVMType, LLVMValue, prelude::LLVMValueRef},
    module::{Linkage, Module},
    passes::PassBuilderOptions,
    support::LLVMString,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetTriple},
    types::{
        AnyTypeEnum, ArrayType, AsTypeRef, BasicMetadataTypeEnum, BasicType, BasicTypeEnum,
        FloatType, IntType, PointerType, StructType,
    },
    values::{
        AggregateValue, AnyValue, AnyValueEnum, AsValueRef, BasicMetadataValueEnum, BasicValue,
        BasicValueEnum, FloatValue, FunctionValue, GlobalValue, InstructionValue, IntMathValue,
        IntValue, PhiValue, PointerValue, StructValue,
    },
};
use std::{borrow::Cow, collections::HashMap, marker::PhantomData, path::Path};

pub mod jit;

#[derive(Debug, thiserror::Error)]
#[error("{:?}", self)]
pub enum CodegenError {
    BuilderError(BuilderError),
    InvalidGeneratedFunction,
    InvalidCallProduced,
    FunctionLookupError(FunctionLookupError),
    CannotOptimizeModule(LLVMString),
    CannotCompileObjFile(LLVMString),
    CannotCreateJit(LLVMString),
    CannotMutateRegister,

    AllocErr(bumpalo::AllocErr),
}

unsafe impl Send for CodegenError {}
unsafe impl Sync for CodegenError {}

#[cfg(not(debug_assertions))]
pub type CodegenResult<T> = Result<T, CodegenError>;
#[cfg(debug_assertions)]
pub type CodegenResult<T> = Result<T, anyhow::Error>;

impl From<BuilderError> for CodegenError {
    fn from(e: BuilderError) -> Self {
        CodegenError::BuilderError(e)
    }
}

impl From<FunctionLookupError> for CodegenError {
    fn from(e: FunctionLookupError) -> Self {
        CodegenError::FunctionLookupError(e)
    }
}

macro_rules! try_not_never {
    ($sym:expr) => {{
        match $sym {
            s @ Symbol::Never => return Ok(s),
            s => s,
        }
    }};
}

/// Returns [`Symbol::Never`] if it occurs
macro_rules! try_compile_expr_as_val {
    ($codegen:ident, $expr:expr) => {{
        let expr: Ptr<Ast> = $expr;
        match $codegen.compile_expr(expr)? {
            sym @ Symbol::Never => return Ok(sym),
            sym => $codegen.sym_as_val(sym, expr.ty.u())?,
        }
    }};
}

/*
/// Returns [`Symbol::Never`] if it occurs
macro_rules! try_get_symbol_as_val {
    ($codegen:ident, $name:expr, $ty:expr) => {{
        match $codegen.get_symbol($name) {
            sym @ Symbol::Never => return Ok(sym),
            sym => $codegen.sym_as_val(sym, $ty)?,
        }
    }};
}
*/

pub struct Codegen<'ctx> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Option<Module<'ctx>>,

    symbols: CodegenSymbolTable<Symbol<'ctx>>,
    type_table: HashMap<Ptr<ast::Type>, CodegenType<'ctx>>,
    fn_table: HashMap<Ptr<ast::Fn>, FunctionValue<'ctx>>,
    defer_stack: ScopedStack<Ptr<Ast>>,

    cur_fn: Option<FunctionValue<'ctx>>,
    cur_loop: Option<Loop<'ctx>>,
    sret_ptr: Option<PointerValue<'ctx>>,

    return_depth: usize,
    continue_break_depth: usize,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Codegen<'ctx> {
        Codegen {
            context,
            builder: context.create_builder(),
            module: Some(context.create_module(module_name)),
            symbols: CodegenSymbolTable::default(),
            type_table: HashMap::new(),
            fn_table: HashMap::new(),
            defer_stack: ScopedStack::default(),
            cur_fn: None,
            cur_loop: None,
            sret_ptr: None,
            return_depth: 0,
            continue_break_depth: 0,
        }
    }

    pub fn compile_top_level(&mut self, stmt: Ptr<Ast>) {
        debug_assert!(stmt.ty.u().matches_void());
        self.compile_expr(stmt).unwrap();
    }

    fn compile_expr(&mut self, expr: Ptr<Ast>) -> CodegenResult<Symbol<'ctx>> {
        self.compile_expr_with_write_target(expr, None)
    }

    fn compile_expr_with_write_target(
        &mut self,
        expr: Ptr<Ast>,
        mut write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let out = self._compile_expr_inner(expr, &mut write_target);
        if let Some(target) = write_target
            && let Ok(out) = out
        {
            match out {
                Symbol::Stack(ptr) => {
                    // #[cfg(debug_assertions)]
                    // println!("WARN: memcpy to write_target");
                    let layout = expr.ty.u().layout();
                    let alignment = layout.align as u32;
                    let size = self.context.i64_type().const_int(layout.size as u64, false);
                    self.builder.build_memcpy(target, alignment, ptr, alignment, size)?;
                },
                Symbol::Register(val) => {
                    self.build_store(target, val.basic_val(), expr.ty.u().alignment())?;
                },
                _ => {},
            }
        }
        out
    }

    #[inline]
    fn _compile_expr_inner(
        &mut self,
        mut expr: Ptr<Ast>,
        write_target: &mut Option<PointerValue<'ctx>>,
    ) -> CodegenResult<Symbol<'ctx>> {
        // println!("compile {:x?}: {:?} {:?}", expr, expr.kind, ast::debug::DebugAst::to_text(&expr));

        expr = expr.rep();

        let out_ty = expr.ty.u();

        let p = primitives();

        /* // see `If`
        if out_ty == p.never_ty {
            return Ok(Symbol::Never);
        }
        */

        macro_rules! write_target_or {
            ($alt:expr) => {
                if let Some(target) = write_target.take() { target } else { $alt }
            };
        }

        match expr.matchable().as_mut() {
            AstEnum::Ident { decl, .. } => {
                debug_assert!(
                    !expr.is_const_val(),
                    "constants should have been replaced during sema"
                );
                Ok(self.get_symbol(decl.u()))
            },
            // AstEnum::Parenthesis { expr, .. } => self.compile_expr_with_write_target(*expr, write_target) ,
            AstEnum::Block { stmts, has_trailing_semicolon, .. } => {
                self.open_scope();
                let res: CodegenResult<Symbol> = try {
                    let mut out = Symbol::Void;
                    if let Some(last) = stmts.last_mut() {
                        last.ty = Some(out_ty)
                    }
                    for s in stmts.iter() {
                        out = self.compile_expr(*s)?;
                        if out == Symbol::Never {
                            break;
                        }
                    }
                    if !*has_trailing_semicolon || out == Symbol::Never {
                        out
                    } else {
                        debug_assert!(out_ty.matches_void());
                        Symbol::Void
                    }
                };
                self.close_scope(res.as_ref().is_ok_and(|s| *s != Symbol::Never))?;
                res
            },
            AstEnum::PositionalInitializer { lhs, args, .. } => {
                if out_ty.kind.is_struct_kind() {
                    let s_def = match out_ty.kind {
                        AstKind::StructDef => out_ty.downcast::<ast::StructDef>(),
                        AstKind::SliceTy => p.untyped_slice_struct_def,
                        _ => unreachable_debug(),
                    };
                    let s_ty = s_def.upcast_to_type();
                    let struct_ty = self.llvm_type(s_ty).struct_ty();
                    let ptr = write_target_or!(self.build_alloca(struct_ty, "struct", s_ty)?);
                    self.compile_positional_initializer_body(struct_ty, ptr, s_def.fields, args)?;
                    stack_val(ptr)
                } else if let Some(ptr) = out_ty.try_downcast::<ast::PtrTy>() {
                    let lhs = lhs.u();
                    debug_assert_eq!(lhs.ty, out_ty);
                    let s_def = ptr.pointee.downcast::<ast::StructDef>();
                    let struct_ty = self.type_table[&s_def.upcast_to_type()].struct_ty(); // TODO: test if `*struct {...}` syntax works
                    let ptr = try_compile_expr_as_val!(self, lhs).ptr_val();
                    self.compile_positional_initializer_body(struct_ty, ptr, s_def.fields, args)?;
                    reg(ptr)
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::NamedInitializer { lhs, fields: values, .. } => {
                if out_ty.kind.is_struct_kind() {
                    let s_def = match out_ty.kind {
                        AstKind::StructDef => out_ty.downcast::<ast::StructDef>(),
                        AstKind::SliceTy => p.untyped_slice_struct_def,
                        _ => unreachable_debug(),
                    };
                    let s_ty = s_def.upcast_to_type();
                    let struct_ty = self.llvm_type(s_ty).struct_ty();
                    let ptr = write_target_or!(self.build_alloca(struct_ty, "struct", s_ty)?);
                    self.compile_named_initializer_body(struct_ty, ptr, s_def.fields, values)?;
                    stack_val(ptr)
                } else if let Some(ptr_ty) = out_ty.try_downcast::<ast::PtrTy>() {
                    let lhs = lhs.u();
                    debug_assert_eq!(lhs.ty, out_ty);
                    let s_def = ptr_ty.pointee.downcast::<ast::StructDef>();
                    let struct_ty = self.type_table[&s_def.upcast_to_type()].struct_ty(); // TODO: test if `*struct {...}` syntax works
                    let ptr = try_compile_expr_as_val!(self, lhs).ptr_val();
                    self.compile_named_initializer_body(struct_ty, ptr, s_def.fields, values)?;
                    reg(ptr)
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::ArrayInitializer { lhs, elements, .. } => {
                debug_assert!(lhs.is_none(), "todo");
                let arr_ty = out_ty.downcast::<ast::ArrayTy>();
                let elem_ty = arr_ty.elem_ty.downcast_type();
                let elem_cty = self.llvm_type(elem_ty);
                let len = elements.len();
                debug_assert_eq!(len, arr_ty.len.int());
                debug_assert!(u32::try_from(len).is_ok());
                let arr_ty = elem_cty.basic_ty().array_type(len as u32);
                let arr_ptr = write_target_or!(self.build_alloca(arr_ty, "arr", out_ty)?);
                let idx_ty = self.context.i64_type();
                for (idx, elem) in elements.iter_mut().enumerate() {
                    finalize_ty(elem.ty.as_mut().u(), elem_ty);
                    let elem_ptr = unsafe {
                        self.builder.build_in_bounds_gep(
                            arr_ty,
                            arr_ptr,
                            &[idx_ty.const_zero(), idx_ty.const_int(idx as u64, false)],
                            "",
                        )
                    }?;
                    let _ = self.compile_expr_with_write_target(*elem, Some(elem_ptr))?;
                }
                stack_val(arr_ptr)
            },
            AstEnum::ArrayInitializerShort { lhs, val, count, .. } => {
                debug_assert!(lhs.is_none(), "todo");
                let elem_ty = val.ty.u();
                let elem_cty = self.llvm_type(elem_ty).basic_ty();
                let len = count.int();
                let arr_ty = elem_cty.array_type(len);
                let arr_ptr = write_target_or!(self.build_alloca(arr_ty, "arr", out_ty)?);
                let elem_val = try_compile_expr_as_val!(self, *val);

                let idx_ty = self.context.i64_type();
                let for_info = self.build_for(
                    idx_ty,
                    false,
                    idx_ty.const_zero(),
                    idx_ty.const_int(len as u64, false),
                    false,
                )?;

                let elem_ptr = unsafe {
                    self.builder.build_in_bounds_gep(
                        arr_ty,
                        arr_ptr,
                        &[idx_ty.const_zero(), for_info.idx_int],
                        "",
                    )
                }?;
                self.build_store(elem_ptr, elem_val.basic_val(), elem_ty.alignment())?;

                self.build_for_end(for_info, Symbol::Void)?;

                stack_val(arr_ptr)
            },
            AstEnum::Dot { lhs, rhs, .. } => {
                let lhs = lhs.u();
                let lhs_ty = lhs.ty.u();
                if let Some(enum_def) = lhs.try_downcast::<ast::EnumDef>() {
                    debug_assert_eq!(out_ty.kind, AstKind::EnumDef);
                    debug_assert_eq!(lhs_ty, p.type_ty);
                    let _ = self.llvm_type(enum_def.upcast_to_type());
                    let (tag, _) = enum_def.variants.find_field(&rhs.text).u();
                    self.compile_enum_val(enum_def, tag, None, write_target.take())
                } else if lhs_ty.kind == AstKind::StructDef || lhs_ty.kind == AstKind::SliceTy {
                    let s_def = match lhs_ty.kind {
                        AstKind::StructDef => lhs_ty.downcast::<ast::StructDef>(),
                        AstKind::SliceTy => p.untyped_slice_struct_def,
                        _ => unreachable_debug(),
                    };
                    let struct_ty = self.type_table[&s_def.upcast_to_type()].struct_ty();
                    let struct_sym = try_not_never!(self.compile_expr(lhs)?);
                    let (field_idx, _) = s_def.fields.find_field(&rhs.text).u();
                    self.build_struct_access(struct_ty, struct_sym, field_idx as u32, &rhs.text)
                } else if let Some(u_def) = lhs_ty.try_downcast::<ast::UnionDef>() {
                    let union_ty = self.type_table[&lhs_ty].struct_ty();
                    let union_sym = try_not_never!(self.compile_expr(lhs)?);
                    debug_assert!(!(u_def.fields.len() > 0 && union_ty.count_fields() == 0));
                    self.build_struct_access(union_ty, union_sym, 0, &rhs.text)
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::Index { lhs, idx, .. } => {
                let lhs_sym = try_not_never!(self.compile_expr(*lhs)?);
                let idx_val = try_compile_expr_as_val!(self, *idx);

                let (ptr, len, elem_ty) = match lhs.ty.matchable().as_ref() {
                    TypeEnum::SliceTy { elem_ty, .. } => {
                        let (ptr, len) = self.build_slice_field_access(lhs_sym)?;
                        (ptr, len, *elem_ty)
                    },
                    TypeEnum::ArrayTy { elem_ty, len, .. } => {
                        let Symbol::Stack(arr_ptr) = lhs_sym else { unreachable_debug() };
                        let len = self.context.i64_type().const_int(len.int(), false);
                        (arr_ptr, len, *elem_ty)
                    },
                    _ => unreachable_debug(),
                };
                let elem_ty = elem_ty.downcast_type();
                let llvm_elem_ty = self.llvm_type(elem_ty).basic_ty();
                match idx.ty.matchable().as_ref() {
                    TypeEnum::IntTy { .. } => {
                        stack_val(self.build_gep(llvm_elem_ty, ptr, &[idx_val.int_val()])?)
                    },
                    TypeEnum::RangeTy { elem_ty, rkind, .. } => {
                        let i = elem_ty.downcast::<ast::IntTy>();
                        let range_val = idx_val.struct_val();

                        let (ptr, len) = match rkind {
                            RangeKind::Full => (ptr, len),
                            RangeKind::From => {
                                let start = self
                                    .builder
                                    .build_extract_value(range_val, 0, "start")?
                                    .into_int_value();
                                let ptr = self.build_gep(llvm_elem_ty, ptr, &[start])?;
                                let len = self.builder.build_int_sub(len, start, "")?;
                                (ptr, len)
                            },
                            RangeKind::To | RangeKind::ToInclusive => {
                                let mut end = self
                                    .builder
                                    .build_extract_value(range_val, 0, "end")?
                                    .into_int_value();
                                if rkind.is_inclusive() {
                                    end = self.builder.build_int_add(
                                        end,
                                        end.get_type().const_int(1, i.is_signed),
                                        "",
                                    )?;
                                }
                                (ptr, end)
                            },
                            RangeKind::Both | RangeKind::BothInclusive => {
                                let start = self
                                    .builder
                                    .build_extract_value(range_val, 0, "start")?
                                    .into_int_value();
                                let mut end = self
                                    .builder
                                    .build_extract_value(range_val, 1, "end")?
                                    .into_int_value();
                                if rkind.is_inclusive() {
                                    end = self.builder.build_int_add(
                                        end,
                                        end.get_type().const_int(1, i.is_signed),
                                        "",
                                    )?;
                                }
                                let ptr = self.build_gep(llvm_elem_ty, ptr, &[start])?;
                                let len = self.builder.build_int_sub(end, start, "")?;
                                (ptr, len)
                            },
                        };
                        self.build_slice(ptr, len)
                    },
                    _ => unreachable_debug(),
                }
            },
            AstEnum::Cast { operand, target_ty, .. } => {
                debug_assert!(target_ty.rep().is_type());
                self.compile_cast(*operand, target_ty.downcast_type())
            },
            AstEnum::Autocast { operand, .. } => self.compile_cast(*operand, out_ty),
            AstEnum::Call { func, args, .. } => {
                if let Some(f) = func.try_downcast::<ast::Fn>() {
                    let val = if func.kind == AstKind::Fn {
                        let Symbol::Function { val } = self.compile_expr(*func)? else {
                            unreachable_debug()
                        };
                        val
                    } else {
                        *self.fn_table.get(&f).u()
                    };
                    self.compile_call(f, val, args.iter().copied(), write_target.take())
                } else if func.ty == p.method_stub {
                    let dot = func.downcast::<ast::Dot>();
                    let f = dot.rhs.rep().downcast::<ast::Fn>();
                    let val = *self.fn_table.get(&f).u();
                    let args = std::iter::once(dot.lhs.u()).chain(args.iter().copied());
                    self.compile_call(f, val, args, write_target.take())
                } else if func.ty == p.enum_variant {
                    let dot = func.downcast::<ast::Dot>();
                    let enum_ty = out_ty.downcast::<ast::EnumDef>();
                    debug_assert_eq!(dot.lhs.u().downcast_type(), enum_ty.upcast_to_type());
                    let idx = enum_ty.variants.find_field(&dot.rhs.text).u().0;
                    debug_assert!(args.len() <= 1);
                    self.compile_enum_val(enum_ty, idx, args.get(0).copied(), write_target.take())
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::UnaryOp { op, operand, .. } => {
                op.finalize_arg_type(operand.ty.as_mut().u(), out_ty);
                let sym = try_not_never!(self.compile_expr(*operand)?);
                match op {
                    UnaryOpKind::AddrOf | UnaryOpKind::AddrMutOf => {
                        reg(self.build_ptr_to_sym(sym, out_ty)?)
                    },
                    UnaryOpKind::Deref => {
                        stack_val(self.sym_as_val(sym, p.never_ptr_ty)?.ptr_val())
                    },
                    UnaryOpKind::Not => {
                        debug_assert!(
                            operand.ty == p.bool || operand.ty.u().kind == AstKind::IntTy
                        );
                        debug_assert!(out_ty == p.bool || out_ty.kind == AstKind::IntTy);
                        let v = self.sym_as_val(sym, out_ty)?;
                        reg(self.builder.build_not(v.bool_val(), "not")?)
                    },
                    UnaryOpKind::Neg => {
                        let v = self.sym_as_val(sym, out_ty)?;
                        if out_ty.kind == AstKind::IntTy {
                            debug_assert!(out_ty.downcast::<ast::IntTy>().is_signed);
                            reg(self.builder.build_int_neg(v.int_val(), "neg")?)
                        } else if out_ty.kind == AstKind::FloatTy {
                            reg(self.builder.build_float_neg(v.float_val(), "neg")?)
                        } else {
                            todo!("neg for other types")
                        }
                    },
                    UnaryOpKind::Try => todo!(),
                }
            },
            AstEnum::BinOp { lhs, op, rhs, .. } => {
                let op = *op;
                op.finalize_arg_type(lhs.ty.as_mut().u(), rhs.ty.as_mut().u(), out_ty);
                let arg_ty = lhs.ty.u();
                let lhs_sym = try_not_never!(self.compile_expr(*lhs)?);
                if matches!(op, BinOpKind::And | BinOpKind::Or) {
                    debug_assert_eq!(arg_ty, p.bool);
                    let lhs = self.sym_as_val(lhs_sym, arg_ty)?.bool_val();
                    return self.build_bool_short_circuit_binop(lhs, *rhs, op);
                }

                let rhs_sym = try_not_never!(self.compile_expr(*rhs)?);
                if let Some(e) = arg_ty.try_downcast::<ast::EnumDef>()
                    && is_simple_enum(e.variants)
                    && matches!(op, BinOpKind::Eq | BinOpKind::Ne)
                {
                    let enum_ty = self.llvm_type(arg_ty);
                    let tag_ty = enum_ty.struct_ty().get_field_type_at_index(0).u();
                    let tag_align = enum_alignment(e.variants);
                    let lhs = self.build_struct_access(enum_ty.struct_ty(), lhs_sym, 0, "")?;
                    let lhs = self.sym_as_val_with_llvm_ty(lhs, tag_ty, tag_align)?.int_val();
                    let rhs = self.build_struct_access(enum_ty.struct_ty(), rhs_sym, 0, "")?;
                    let rhs = self.sym_as_val_with_llvm_ty(rhs, tag_ty, tag_align)?.int_val();
                    return self.build_int_binop(lhs, rhs, false, op).map(reg_sym);
                }

                let lhs_val = self.sym_as_val(lhs_sym, arg_ty)?;
                let rhs_val = self.sym_as_val(rhs_sym, arg_ty)?;
                if arg_ty == p.bool {
                    return self.build_bool_binop(lhs_val.bool_val(), rhs_val.bool_val(), op);
                }
                match arg_ty.matchable().as_ref() {
                    TypeEnum::IntTy { is_signed, .. } => self
                        .build_int_binop(lhs_val.int_val(), rhs_val.int_val(), *is_signed, op)
                        .map(reg_sym),
                    TypeEnum::PtrTy { .. } => self
                        .build_int_binop(lhs_val.ptr_val(), rhs_val.ptr_val(), false, op)
                        .map(reg_sym),
                    TypeEnum::OptionTy { ty, .. } if ty.u().kind == AstKind::PtrTy => self
                        .build_int_binop(lhs_val.int_val(), rhs_val.int_val(), false, op)
                        .map(reg_sym),
                    TypeEnum::FloatTy { .. } => self
                        .build_float_binop(lhs_val.float_val(), rhs_val.float_val(), op)
                        .map(reg_sym),
                    _ => {
                        display(expr.full_span()).finish();
                        todo!("binop {op:?} for {}", arg_ty)
                    },
                }
            },
            AstEnum::Range { start, end, .. } => {
                let r = out_ty.downcast::<ast::RangeTy>();
                let mut range = self.range_type(r).get_undef().as_aggregate_value_enum();
                if let Some(start) = start {
                    finalize_ty(start.ty.as_mut().u(), r.elem_ty);
                    let val = try_compile_expr_as_val!(self, *start);
                    range = self.builder.build_insert_value(range, val.basic_val(), 0, "")?;
                }
                if let Some(end) = end {
                    finalize_ty(end.ty.as_mut().u(), r.elem_ty);
                    let val = try_compile_expr_as_val!(self, *end);
                    let end_idx = r.rkind.get_field_count() as u32 - 1;
                    range = self.builder.build_insert_value(range, val.basic_val(), end_idx, "")?;
                }
                reg(range)
            },
            &mut AstEnum::Assign { lhs, rhs, .. } => {
                //debug_assert_eq!(lhs.ty, rhs.ty);
                debug_assert!(ty_match(lhs.ty.u(), rhs.ty.u()));
                let lhs_sym = self.compile_expr(lhs)?;
                let stack_ptr = self.build_ptr_to_sym(lhs_sym, lhs.ty.u())?;
                let _ = self.compile_expr_with_write_target(rhs, Some(stack_ptr))?;
                Ok(Symbol::Void)
            },
            &mut AstEnum::BinOpAssign { lhs, op, rhs, .. } => {
                debug_assert_eq!(lhs.ty, rhs.ty);
                let arg_ty = lhs.ty.u();
                let Symbol::Stack(stack_var) = self.compile_expr(lhs)? else {
                    todo!("variable mut check during sema");
                    unreachable_debug()
                };
                let lhs_llvm_ty = self.llvm_type(arg_ty).basic_ty();
                let lhs_val = self.build_load(lhs_llvm_ty, stack_var, "lhs", arg_ty.alignment())?;

                if matches!(op, BinOpKind::And | BinOpKind::Or) {
                    debug_assert_eq!(arg_ty, p.bool);
                    return self.build_bool_short_circuit_binop(lhs_val.into_int_value(), rhs, op);
                }

                debug_assert_eq!(rhs.ty, arg_ty);
                let rhs_val = try_compile_expr_as_val!(self, rhs);
                let binop_res = match arg_ty.matchable().as_ref() {
                    TypeEnum::IntTy { is_signed, .. } => self.build_int_binop(
                        lhs_val.into_int_value(),
                        rhs_val.int_val(),
                        *is_signed,
                        op,
                    )?,
                    TypeEnum::FloatTy { .. } => {
                        self.build_float_binop(lhs_val.into_float_value(), rhs_val.float_val(), op)?
                    },
                    /*
                    TypeEnum::Bool { .. } => {
                        panic!("{:#?}", expr.kind);
                        //self.build_bool_binop(lhs.into_int_value(),
                        // rhs.bool_val(), op)?
                    },
                    */
                    t => todo!("{:?}", t),
                };
                self.build_store(stack_var, binop_res.basic_val(), arg_ty.alignment())?;
                Ok(Symbol::Void)
            },
            AstEnum::Decl {
                markers, ident, on_type, var_ty, init, is_const, is_extern, ..
            } => {
                let var_ty = var_ty.u();
                if let Some(init) = init {
                    init.ty = Some(var_ty);
                }

                if *is_const {
                    if var_ty == p.fn_val {
                        let f = init.u().downcast::<ast::Fn>();
                        if init.u().kind != AstKind::Fn {
                            // don't need to compile an alias again
                            debug_assert!(self.fn_table.contains_key(&f));
                        } else {
                            let name = match on_type {
                                Some(ty) => {
                                    let ty = ty.downcast_type();
                                    Cow::Owned(format!("{}.{}", ty, &*ident.text)) // TODO: use correct type name
                                },
                                None => Cow::Borrowed(ident.text.as_ref()),
                            };
                            if *is_extern {
                                self.compile_prototype(name.as_ref(), f);
                            } else {
                                self.compile_fn(name.as_ref(), f)?;
                            }
                        }
                    } else if var_ty == p.type_ty {
                        self.llvm_type(init.u().downcast_type());
                    }

                    // compile time values are inlined during sema. We don't have to add those to
                    // the symbol table.
                } else {
                    debug_assert_ne!(var_ty, p.fn_val);
                    debug_assert_ne!(var_ty, p.type_ty);
                    debug_assert_ne!(expr.kind, AstKind::Fn);
                    debug_assert_ne!(expr.rep().kind, AstKind::Fn);
                    debug_assert!(on_type.is_none());

                    const ENABLE_NON_MUT_TO_REG: bool = false;

                    let sym = if *is_extern {
                        let ty = self.llvm_type(var_ty).basic_ty();
                        let val = self.module().add_global(ty, None, &ident.text);
                        Symbol::Global { val }
                    } else if ENABLE_NON_MUT_TO_REG
                        && let Some(init) = init
                        && !markers.is_mut
                    {
                        try_not_never!(self.compile_expr(*init)?)
                    } else {
                        let stack_ty = self.llvm_type(var_ty).basic_ty();
                        let stack_ptr = self.build_alloca(stack_ty, &ident.text, var_ty)?;
                        if let Some(init) = init {
                            finalize_ty(init.ty.as_mut().u(), var_ty);
                            let _init = try_not_never!(
                                self.compile_expr_with_write_target(*init, Some(stack_ptr))?
                            );
                        }
                        Symbol::Stack(stack_ptr)
                    };

                    self.symbols.push((ident.decl.u(), sym));
                }

                debug_assert!(out_ty.matches_void());
                Ok(Symbol::Void)
            },
            /*
            AstEnum::Extern { ident, ty, .. } => {
                debug_assert!(ty.u().ty == p.type_ty);
                let sym = if let Some(f) = ty.u().try_downcast::<ast::FunctionTy>() {
                    Symbol::Function { val: self.compile_prototype(&ident.text, f.func).0 }
                } else {
                    let ty = self.llvm_type(ty.u()).basic_ty();
                    let val = self.module.add_global(ty, None, &ident.text);
                    Symbol::Global { val }
                };
                let _ = self.symbols.insert(ident.text, sym);
                Ok(Symbol::Void)
            },
            */
            AstEnum::If { condition, then_body, else_body, .. } => {
                let func = self.cur_fn.u();
                debug_assert!(condition.ty == p.bool);
                let condition = try_compile_expr_as_val!(self, *condition).bool_val();
                let condition = if condition.get_type().get_bit_width() > 1 {
                    let i1 = self.context.custom_width_int_type(1);
                    // TODO: truncate the bool on load
                    self.builder.build_int_truncate(condition, i1, "")?
                } else {
                    condition
                };

                let write_target = write_target.take();
                if write_target.is_some() {
                    debug_assert!(else_body.is_some());
                }

                let mut then_bb = self.context.append_basic_block(func, "then");
                let mut else_bb = self.context.append_basic_block(func, "else");
                let merge_bb = self.context.append_basic_block(func, "ifmerge");

                self.builder.build_conditional_branch(condition, then_bb, else_bb)?;

                self.builder.position_at_end(then_bb);
                then_body.ty = Some(out_ty);
                let then_sym = self.compile_expr_with_write_target(*then_body, write_target)?;
                let then_val = self.sym_as_val_checked(then_sym, out_ty)?;
                if then_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_sym = if let Some(else_body) = else_body {
                    else_body.ty = Some(out_ty);
                    self.compile_expr_with_write_target(*else_body, write_target)?
                } else {
                    Symbol::Void
                };
                let else_val = self.sym_as_val_checked(else_sym, out_ty)?;
                if else_sym != Symbol::Never {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                else_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);

                if out_ty == p.void_ty {
                    return Ok(Symbol::Void);
                } else if out_ty == p.never {
                    self.builder.build_unreachable()?;
                    return Ok(Symbol::Never);
                }

                if let Some(write_target) = write_target {
                    stack_val(write_target)
                } else {
                    let branch_ty = self.llvm_type(out_ty).basic_ty();
                    let phi = self.builder.build_phi(branch_ty, "ifexpr")?;
                    debug_assert!(then_val.is_some() || else_val.is_some());
                    if let Some(then_val) = then_val {
                        phi.add_incoming(&[(&then_val.basic_val(), then_bb)]);
                    }
                    if let Some(else_val) = else_val {
                        phi.add_incoming(&[(&else_val.basic_val(), else_bb)]);
                    }
                    reg(phi)
                }
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { source, iter_var, body, .. } => {
                let outer_continue_break_depth = self.continue_break_depth;
                self.continue_break_depth = 0;

                let source_ty = self.llvm_type(source.ty.u());
                let source_sym = try_not_never!(self.compile_expr(*source)?);

                match source.ty.matchable().as_ref() {
                    TypeEnum::ArrayTy { len, .. } => {
                        let idx_ty = self.context.i64_type();
                        let len = idx_ty.const_int(len.int(), false);
                        let for_info =
                            self.build_for(idx_ty, false, idx_ty.const_zero(), len, false)?;

                        let Symbol::Stack(arr_ptr) = source_sym else { panic!() };
                        let iter_var_sym =
                            Symbol::Stack(self.build_gep(source_ty.arr_ty(), arr_ptr, &[
                                idx_ty.const_zero(),
                                for_info.idx_int,
                            ])?);
                        self.symbols.push((iter_var.decl.u(), iter_var_sym));
                        debug_assert!(body.ty.u().matches_void());
                        let out = self.compile_expr(*body)?;
                        self.build_for_end(for_info, out)?
                    },
                    TypeEnum::SliceTy { elem_ty, .. } => {
                        let idx_ty = self.context.i64_type();
                        let (ptr, len) = self.build_slice_field_access(source_sym)?;

                        let for_info =
                            self.build_for(idx_ty, false, idx_ty.const_zero(), len, false)?;
                        let elem_ty = self.llvm_type(elem_ty.downcast_type()).basic_ty();
                        let iter_var_sym =
                            Symbol::Stack(self.build_gep(elem_ty, ptr, &[for_info.idx_int])?);
                        self.symbols.push((iter_var.decl.u(), iter_var_sym));
                        debug_assert!(body.ty.u().matches_void());
                        let out = self.compile_expr(*body)?;
                        self.build_for_end(for_info, out)?
                    },
                    TypeEnum::RangeTy { elem_ty, rkind, .. } if rkind.has_start() => {
                        let i = elem_ty.downcast::<ast::IntTy>();
                        let elem_llvm_ty = self.llvm_type(*elem_ty).int_ty();
                        let range_ty = source_ty.struct_ty();
                        let start = self.build_struct_access(range_ty, source_sym, 0, "start")?;
                        let start = self.sym_as_val(start, *elem_ty)?.int_val();

                        let end = if rkind.has_end() {
                            let idx = rkind.get_field_count() as u32 - 1;
                            let end = self.build_struct_access(range_ty, source_sym, idx, "end")?;
                            self.sym_as_val(end, *elem_ty)?.int_val()
                        } else {
                            self.max_int(elem_llvm_ty, i.is_signed)?
                        };

                        let for_info = self.build_for(
                            elem_llvm_ty,
                            i.is_signed,
                            start,
                            end,
                            rkind.is_inclusive(),
                        )?;
                        let iter_var_sym = reg_sym(for_info.idx);
                        self.symbols.push((iter_var.decl.u(), iter_var_sym));
                        let out = self.compile_expr(*body)?;
                        self.build_for_end(for_info, out)?
                    },
                    _ => panic_debug("for loop over other types"),
                };

                self.continue_break_depth = outer_continue_break_depth;
                debug_assert!(out_ty.matches_void());
                Ok(Symbol::Void)
            },
            AstEnum::While { condition, body, .. } => {
                let func = self.cur_fn.u();
                let cond_bb = self.context.append_basic_block(func, "while.cond");
                let body_bb = self.context.append_basic_block(func, "while.body");
                let end_bb = self.context.append_basic_block(func, "while.end");

                let outer_continue_break_depth = self.continue_break_depth;
                self.continue_break_depth = 0;

                // entry
                self.builder.build_unconditional_branch(cond_bb)?;

                // cond
                self.builder.position_at_end(cond_bb);
                debug_assert!(condition.ty.u().matches_bool());
                let cond = try_compile_expr_as_val!(self, *condition).bool_val();
                self.builder.build_conditional_branch(cond, body_bb, end_bb)?;

                // body
                self.builder.position_at_end(body_bb);
                let outer_loop = self.cur_loop.replace(Loop { continue_bb: cond_bb, end_bb });
                debug_assert!(body.ty.u().matches_void());
                let out = self.compile_expr(*body)?;
                self.cur_loop = outer_loop;
                if !matches!(out, Symbol::Never) {
                    self.builder.build_unconditional_branch(cond_bb)?;
                }

                // end
                self.builder.position_at_end(end_bb);
                self.continue_break_depth = outer_continue_break_depth;
                debug_assert!(out_ty.matches_void());
                Ok(Symbol::Void)
            },
            // AstEnum::Catch { .. } => todo!(),
            AstEnum::Defer { stmt, .. } => {
                self.defer_stack.push(*stmt);
                debug_assert!(out_ty.matches_void());
                Ok(Symbol::Void)
            },
            AstEnum::Return { val, parent_fn, .. } => {
                let f = parent_fn.u();
                if let Some(val) = val {
                    finalize_ty(val.ty.as_mut().u(), f.ret_ty.u());
                    let sym = self.compile_expr(*val)?;
                    if self.compile_multiple_defer_scopes(self.return_depth)? {
                        self.build_return(sym, val.ty.u())?;
                    }
                } else {
                    debug_assert_eq!(f.ret_ty, p.void_ty);
                    if self.compile_multiple_defer_scopes(self.return_depth)? {
                        self.builder.build_return(None)?;
                    }
                }
                Ok(Symbol::Never)
            },
            AstEnum::Break { val, .. } => {
                if val.is_some() {
                    todo!("break with expr")
                }
                let _ = self.compile_multiple_defer_scopes(self.continue_break_depth)?;
                let bb = self.cur_loop.u().end_bb;
                self.builder.build_unconditional_branch(bb)?;
                Ok(Symbol::Never)
            },
            AstEnum::Continue { .. } => {
                let _ = self.compile_multiple_defer_scopes(self.continue_break_depth)?;
                let bb = self.cur_loop.u().continue_bb;
                self.builder.build_unconditional_branch(bb)?;
                Ok(Symbol::Never)
            },
            AstEnum::ImportDirective { .. } => Ok(Symbol::Module),

            AstEnum::IntVal { val, .. } => match expr.ty.matchable().as_ref() {
                TypeEnum::IntTy { bits, is_signed, .. } => {
                    reg(self.int_type(*bits).const_int(expr.int(), *is_signed))
                },
                TypeEnum::FloatTy { bits, .. } => {
                    let float = *val as f64;
                    if float as i64 != *val {
                        panic!("literal precision loss")
                    }
                    reg(self.float_type(*bits).const_float(float))
                },
                _ => unreachable_debug(),
            },
            AstEnum::FloatVal { val, .. } => {
                let float_ty = expr.ty.downcast::<ast::FloatTy>();
                reg(self.float_type(float_ty.bits).const_float(*val))
            },
            AstEnum::BoolVal { val, .. } => {
                debug_assert!(expr.ty == p.bool);
                reg(self.bool_val(*val))
            },
            AstEnum::CharVal { val, .. } => {
                //debug_assert!(expr.ty == p.char);
                debug_assert!(expr.ty == p.u8);
                reg(self.int_type(8).const_int(*val as u8 as u64, false)) // TODO: real char type
            },
            // AstEnum::BCharLit { val, .. } => reg(self.int_type(8).const_int(*val as u64, false)),
            AstEnum::StrVal { text, .. } => {
                debug_assert!(expr.ty.u().matches_str());
                let value = replace_escape_chars(&text);
                let ptr = self.builder.build_global_string_ptr(&value, "")?;
                let len = self.int_type(64).const_int(value.len() as u64, false);
                self.build_slice(ptr.as_pointer_value(), len)
            },
            AstEnum::PtrVal { val, .. } => {
                if *val == 0 {
                    reg(self.ptr_type().const_null())
                } else {
                    todo!("other const ptrs")
                }
            },
            AstEnum::SimpleTy { .. }
            | AstEnum::IntTy { .. }
            | AstEnum::FloatTy { .. }
            | AstEnum::PtrTy { .. }
            | AstEnum::SliceTy { .. }
            | AstEnum::ArrayTy { .. }
            | AstEnum::StructDef { .. }
            | AstEnum::UnionDef { .. }
            | AstEnum::EnumDef { .. }
            | AstEnum::RangeTy { .. }
            | AstEnum::OptionTy { .. } => todo!("runtime type"),
            AstEnum::Fn { .. } => {
                Ok(Symbol::Function { val: *self.fn_table.get(&expr.downcast::<ast::Fn>()).u() })
            },
        }
    }

    fn compile_call(
        &mut self,
        f: Ptr<ast::Fn>,
        fn_val: FunctionValue<'ctx>,
        args: impl IntoIterator<Item = Ptr<Ast>>,
        mut write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let ret_ty = f.ret_ty.u();
        let use_sret = self.ret_type(ret_ty) == RetType::SRetParam;
        let sret_arg = if !use_sret {
            None
        } else if let Some(write_target) = write_target.take() {
            Some(write_target)
        } else {
            let llvm_ty = self.llvm_type(ret_ty).basic_ty();
            Some(self.build_alloca(llvm_ty, "out", ret_ty)?)
        };
        let has_sret = sret_arg.is_some() as usize;
        let args_count = f.params.len() + has_sret;
        let mut arg_values = Vec::with_capacity(args_count);
        unsafe { arg_values.set_len(args_count) };
        #[cfg(debug_assertions)]
        let mut arg_values_was_initialized = vec![false; arg_values.len()];

        macro_rules! set_arg_val {
            ($idx:expr, $val:expr) => {
                arg_values[$idx] = $val;
                #[cfg(debug_assertions)]
                {
                    arg_values_was_initialized[$idx] = true;
                }
            };
        }

        if let Some(sret_arg) = sret_arg {
            set_arg_val!(0, BasicMetadataValueEnum::from(sret_arg));
        }

        for_each_call_arg(f.params, args, |arg, param, mut p_idx| {
            let sym = self.compile_expr(arg)?;
            let val = if param.var_ty.u().pass_arg_as_ptr() {
                let ptr = match sym {
                    Symbol::Stack(ptr) => ptr,
                    Symbol::Register(val) => {
                        let llvm_ty = self.llvm_type(param.var_ty.u()).basic_ty();
                        let ptr =
                            self.build_alloca(llvm_ty, &param.ident.text, param.var_ty.u())?;
                        self.build_store(ptr, val.basic_val(), param.var_ty.u().alignment())?;
                        ptr
                    },
                    Symbol::Global { val } => val.as_pointer_value(),
                    _ => unreachable_debug(),
                };
                BasicMetadataValueEnum::from(ptr)
            } else {
                self.sym_as_val(sym, param.var_ty.u())?.basic_metadata_val()
            };
            p_idx += has_sret;
            set_arg_val!(p_idx, val);
            Ok(())
        })?;
        debug_assert_eq!(arg_values.len() as u32, fn_val.count_params());
        #[cfg(debug_assertions)]
        debug_assert!(arg_values_was_initialized.iter().all(|b| *b));
        let ret = self.builder.build_call(fn_val, &arg_values, "call")?;
        if let Some(write_target) = write_target.take() {
            let ret = CodegenValue::new(ret.as_value_ref()).basic_val();
            self.build_store(write_target, ret, ret_ty.alignment())?;
        }
        let p = primitives();
        if ret_ty == p.never {
            self.builder.build_unreachable()?;
            Ok(Symbol::Never)
        } else if ret_ty == p.void_ty {
            Ok(Symbol::Void)
        } else if use_sret {
            stack_val(sret_arg.u())
        } else {
            reg(ret)
        }
    }

    fn compile_enum_val(
        &mut self,
        enum_def: Ptr<ast::EnumDef>,
        variant_idx: usize,
        data: Option<Ptr<Ast>>,
        write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let enum_ty = self.type_table[&enum_def.upcast_to_type()].struct_ty();
        let tag_ty = self.enum_tag_type(enum_def.variants);
        let tag_val = tag_ty.const_int(variant_idx as u64, false);
        if write_target.is_some() || data.is_some() {
            let enum_ptr = if let Some(ptr) = write_target {
                ptr
            } else {
                self.build_alloca(enum_ty, "enum", enum_def.upcast_to_type())?
            };

            // set tag
            let tag_ptr = enum_ptr;
            self.build_store(tag_ptr, tag_val, enum_alignment(enum_def.variants))?;

            // set data
            if let Some(data) = data {
                let data_ptr = self.builder.build_struct_gep(enum_ty, enum_ptr, 1, "enum_data")?;
                debug_assert_eq!(data.ty, enum_def.variants[variant_idx].var_ty.u());
                let _ = self.compile_expr_with_write_target(data, Some(data_ptr))?;
            }

            stack_val(enum_ptr)
        } else {
            reg(self.builder.build_insert_value(enum_ty.get_poison(), tag_val, 0, "")?)
        }
    }

    fn compile_fn(&mut self, name: &str, f: Ptr<ast::Fn>) -> CodegenResult<FunctionValue<'ctx>> {
        let prev_bb = self.builder.get_insert_block();

        let (fn_val, use_sret) = self.compile_prototype(name, f);

        let prev_sret_ptr = if use_sret {
            let sret_ptr = fn_val.get_first_param().u().into_pointer_value();
            self.sret_ptr.replace(sret_ptr)
        } else {
            self.sret_ptr.take()
        };

        let val = self.compile_fn_body(fn_val, f, use_sret)?;

        self.sret_ptr = prev_sret_ptr;

        if let Some(prev_bb) = prev_bb {
            self.builder.position_at_end(prev_bb);
        }

        Ok(val)
    }

    fn compile_prototype(&mut self, name: &str, f: Ptr<ast::Fn>) -> (FunctionValue<'ctx>, bool) {
        debug_assert!(
            self.fn_table.get(&f).is_none(),
            "called compile_prototype multiple times on the same function ('{name}')"
        );
        let ptr_type = BasicMetadataTypeEnum::from(self.ptr_type());
        let ret_type = self.ret_type(f.ret_ty.u());
        let use_sret = ret_type == RetType::SRetParam;
        let mut param_types = if use_sret { vec![ptr_type] } else { Vec::new() };
        f.params
            .iter()
            .map(|d| {
                let var_ty = d.var_ty.u();
                if var_ty.pass_arg_as_ptr() {
                    ptr_type
                } else {
                    self.llvm_type(var_ty).basic_metadata_ty()
                }
            })
            .collect_into(&mut param_types);
        let fn_type = match ret_type.into_basic() {
            Some(ret_type) => ret_type.fn_type(&param_types, false),
            None => self.context.void_type().fn_type(&param_types, false),
        };
        let fn_val = self.module().add_function(name, fn_type, Some(Linkage::External));
        let mut params_iter = fn_val.get_param_iter();

        if use_sret {
            let llvm_ty = self.llvm_type(f.ret_ty.u()).any_ty();
            let sret = self
                .context
                .create_type_attribute(Attribute::get_named_enum_kind_id("sret"), llvm_ty);
            fn_val.add_attribute(inkwell::attributes::AttributeLoc::Param(0), sret);
            params_iter.next().u().set_name("sret");
        }

        for (idx, param) in params_iter.enumerate() {
            param.set_name(&f.params[idx].ident.text)
        }

        let prev = self.fn_table.insert(f, fn_val);
        debug_assert!(prev.is_none());
        (fn_val, use_sret)
    }

    pub fn compile_fn_body(
        &mut self,
        func: FunctionValue<'ctx>,
        f: Ptr<ast::Fn>,
        use_sret: bool,
    ) -> CodegenResult<FunctionValue<'ctx>> {
        let entry = self.context.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let outer_fn = self.cur_fn.replace(func);
        let outer_return_depth = self.return_depth;
        self.return_depth = 0;

        self.open_scope();
        let res = try {
            self.symbols.reserve(f.params.len());

            for (param, param_def) in
                func.get_param_iter().skip(use_sret as usize).zip(f.params.iter())
            {
                let param = CodegenValue::new(param.as_value_ref());
                let param_ty = param_def.var_ty.u();
                let s = if param_ty.pass_arg_as_ptr() {
                    Symbol::Stack(param.ptr_val())
                } else {
                    if param_def.markers.is_mut {
                        self.position_builder_at_start(func.get_first_basic_block().u());
                        Symbol::Stack(self.build_alloca_value(param.basic_val(), param_ty)?)
                    } else {
                        Symbol::Register(param)
                    }
                };
                self.symbols.push((*param_def, s));
            }

            f.body.u().as_mut().ty = Some(f.ret_ty.u());
            let body = self.compile_expr(f.body.u())?;
            self.build_return(body, f.ret_ty.u())?;

            if func.verify(true) {
                func
            } else {
                #[cfg(debug_assertions)]
                self.module().print_to_stderr();
                unsafe { func.delete() };
                panic_debug("invalid generated function");
                Err(CodegenError::InvalidGeneratedFunction)?
            }
        };
        self.close_scope(true)?; // TODO: is `true` correct?
        self.return_depth = outer_return_depth;
        self.cur_fn = outer_fn;
        self.builder.clear_insertion_position();
        res
    }

    fn compile_positional_initializer_body(
        &mut self,
        struct_ty: StructType<'ctx>,
        struct_ptr: PointerValue<'ctx>,
        fields: Ptr<[Ptr<ast::Decl>]>,
        args: &[Ptr<Ast>],
    ) -> CodegenResult<Symbol<'ctx>> {
        for_each_call_arg(
            fields,
            args.iter().copied(),
            |val: Ptr<Ast>, field_def: Ptr<ast::Decl>, f_idx: usize| {
                let field_ptr = self.builder.build_struct_gep(
                    struct_ty,
                    struct_ptr,
                    f_idx as u32,
                    &field_def.ident.text,
                )?;
                finalize_ty(val.as_mut().ty.as_mut().u(), field_def.var_ty.u());
                self.compile_expr_with_write_target(val, Some(field_ptr))?;
                CodegenResult::Ok(())
            },
        )?;
        stack_val(struct_ptr)
    }

    fn compile_named_initializer_body(
        &mut self,
        struct_ty: StructType<'ctx>,
        struct_ptr: PointerValue<'ctx>,
        fields: Ptr<[Ptr<ast::Decl>]>,
        values: &[(Ptr<ast::Ident>, Option<Ptr<Ast>>)],
    ) -> CodegenResult<()> {
        let mut is_initialized_field = vec![false; fields.len()];
        for (f, init) in values.iter() {
            let (f_idx, field_def) = fields.find_field(&f.text).u();
            is_initialized_field[f_idx] = true;

            let field_ptr =
                self.builder.build_struct_gep(struct_ty, struct_ptr, f_idx as u32, &*f.text)?;

            match init {
                Some(init) => {
                    finalize_ty(init.as_mut().ty.as_mut().u(), field_def.var_ty.u());
                    let _ = self.compile_expr_with_write_target(*init, Some(field_ptr));
                },
                None => {
                    let var_ty = field_def.var_ty.u();
                    let sym = self.get_symbol(f.decl.u());
                    let val = self.sym_as_val(sym, var_ty)?.basic_val();
                    self.build_store(field_ptr, val, var_ty.alignment())?;
                },
            };
        }
        for (f_idx, _) in
            is_initialized_field.into_iter().enumerate().filter(|(_, is_init)| !is_init)
        {
            let field = fields[f_idx];
            let field_ptr = self.builder.build_struct_gep(
                struct_ty,
                struct_ptr,
                f_idx as u32,
                &field.ident.text,
            )?;
            finalize_ty(field.as_mut().init.as_mut().u().ty.as_mut().u(), field.var_ty.u()); // TODO: don't finalize the types here
            let _ = self.compile_expr_with_write_target(field.init.u(), Some(field_ptr))?;
        }
        Ok(())
    }

    fn compile_cast(
        &mut self,
        expr: Ptr<Ast>,
        target_ty: Ptr<ast::Type>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let p = primitives();
        let expr_ty = expr.ty.u();
        let sym = self.compile_expr(expr)?;

        if expr_ty == target_ty {
            return Ok(sym);
        }

        if target_ty.kind == AstKind::OptionTy && expr_ty.is_non_null() {
            return Ok(sym);
        }

        if expr_ty.kind == AstKind::PtrTy && target_ty.kind == AstKind::PtrTy {
            return Ok(sym);
        }

        // TODO: remove this rule
        if let Some(expr_ty) = expr_ty.try_downcast::<ast::OptionTy>() {
            if let Some(target_ty) = target_ty.try_downcast::<ast::OptionTy>()
                && (expr_ty.inner_ty == target_ty.inner_ty
                    || (expr_ty.inner_ty.kind == AstKind::PtrTy
                        && target_ty.inner_ty.kind == AstKind::PtrTy))
            {
                return Ok(sym);
            } else if expr_ty.inner_ty.kind == AstKind::PtrTy && target_ty.kind == AstKind::PtrTy {
                return Ok(sym);
            }
        }

        if let Some(i_ty) = expr_ty.try_downcast::<ast::IntTy>() {
            let int = self.sym_as_val(sym, expr_ty)?.int_val();
            let target = self.llvm_type(target_ty);
            return if target_ty.kind == AstKind::IntTy {
                let rhs_ty = target.int_ty();
                // The documentation of `build_int_cast_sign_flag` is wrong. The signedness of the
                // source type, not the target type, is relevant.
                let is_signed = i_ty.is_signed;
                reg(self.builder.build_int_cast_sign_flag(int, rhs_ty, is_signed, "")?)
            } else if target_ty.kind == AstKind::FloatTy {
                let float_ty = target.float_ty();
                reg(if i_ty.is_signed {
                    self.builder.build_signed_int_to_float(int, float_ty, "")?
                } else {
                    self.builder.build_unsigned_int_to_float(int, float_ty, "")?
                })
            } else {
                unreachable_debug()
            };
        }

        if expr_ty == p.bool
            && let Some(i_ty) = target_ty.try_downcast::<ast::IntTy>()
        {
            let lhs = self.sym_as_val(sym, p.bool)?.bool_val();
            let int_ty = self.llvm_type(i_ty.upcast_to_type()).int_ty();
            return reg(if i_ty.is_signed {
                //self.builder.build_int_s_extend_or_bit_cast(int_value, int_type, "")
                self.builder.build_int_s_extend(lhs, int_ty, "")?
            } else {
                //self.builder.build_int_z_extend_or_bit_cast(int_value, int_type, "")
                self.builder.build_int_z_extend(lhs, int_ty, "")?
            });
        }

        if expr_ty.kind == AstKind::FloatTy {
            let float = self.sym_as_val(sym, expr_ty)?.float_val();
            let target = self.llvm_type(target_ty);
            return if let Some(target_i_ty) = target_ty.try_downcast::<ast::IntTy>() {
                let int_ty = target.int_ty();
                return reg(if target_i_ty.is_signed {
                    self.builder.build_float_to_signed_int(float, int_ty, "")?
                } else {
                    self.builder.build_float_to_unsigned_int(float, int_ty, "")?
                });
            } else {
                unreachable_debug()
            };
        }

        /*
            (p @ TypeEnum::PtrTy { .. }, i @ TypeEnum::IntTy { .. }) => {
                let ptr = self.sym_as_val(sym, &p)?.ptr_val();
                let int_ty = self.llvm_type(&i).int_ty();
                reg(self.builder.build_ptr_to_int(ptr, int_ty, "")?)
            },
            (TypeEnum::Option { ty: p, .. }, i @ TypeEnum::IntTy { .. })
                if matches!(*p, TypeEnum::PtrTy { .. }) =>
            {
                let ptr = self.sym_as_val(sym, &p)?.ptr_val();
                let int_ty = self.llvm_type(&i).int_ty();
                reg(self.builder.build_ptr_to_int(ptr, int_ty, "")?)
            },
        */
        if (expr_ty.kind == AstKind::PtrTy
            || expr_ty
                .try_downcast::<ast::OptionTy>()
                .is_some_and(|opt| opt.inner_ty.kind == AstKind::PtrTy))
            && let Some(i_ty) = target_ty.try_downcast::<ast::IntTy>()
        {
            let ptr = self.sym_as_val(sym, expr_ty)?.ptr_val();
            let int_ty = self.llvm_type(i_ty.upcast_to_type()).int_ty();
            return reg(self.builder.build_ptr_to_int(ptr, int_ty, "")?);
        }

        /*
            (l @ TypeEnum::IntTy { .. }, r @ TypeEnum::IntTy { is_signed, .. }) => {
                let lhs = self.sym_as_val(sym, &l)?.int_val();
                let rhs_ty = self.llvm_type(&r).int_ty();
                reg(self.builder.build_int_cast_sign_flag(lhs, rhs_ty, is_signed, "")?)
            },

            (i @ TypeEnum::IntTy { is_signed, .. }, f @ TypeEnum::FloatTy { .. }) => {
                let int = self.sym_as_val(sym, &i)?.int_val();
                let float_ty = self.llvm_type(&f).float_ty();
                reg(if is_signed {
                    self.builder.build_signed_int_to_float(int, float_ty, "")?
                } else {
                    self.builder.build_unsigned_int_to_float(int, float_ty, "")?
                })
            },
            (f @ TypeEnum::FloatTy { .. }, i @ TypeEnum::IntTy { is_signed, .. }) => {
                let float = self.sym_as_val(sym, &f)?.float_val();
                let int_ty = self.llvm_type(&i).int_ty();
                reg(if is_signed {
                    self.builder.build_float_to_signed_int(float, int_ty, "")?
                } else {
                    self.builder.build_float_to_unsigned_int(float, int_ty, "")?
                })
            },
        */

        /*

            (e @ TypeEnum::EnumDef { variants, .. }, i @ TypeEnum::IntTy { .. })
                if is_simple_enum(variants) =>
            {
                let lhs = self.sym_as_val(sym, &e)?.struct_val();
                let tag = self.builder.build_extract_value(lhs, 0, "")?.into_int_value();
                let rhs_ty = self.llvm_type(&i).int_ty();
                reg(self.builder.build_int_cast_sign_flag(tag, rhs_ty, false, "")?)
            },

            (l, t) => panic!("cannot cast {l} to {t}"),
        }
        */

        display(expr.full_span()).finish();

        todo!("cast {} to {}", expr_ty, target_ty);
    }

    // -----------------------

    /// Note: alloca in a loop results in a stack overflow because llvm doesn't cleanup alloca
    /// until the end of the function
    ///
    /// See <https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas>
    fn build_alloca(
        &self,
        llvm_ty: impl BasicType<'ctx>,
        name: &str,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let prev_pos = self.builder.get_insert_block();
        let fn_entry_bb = self.cur_fn.u().get_first_basic_block().u();
        self.position_builder_at_start(fn_entry_bb);

        let ptr = self.builder.build_alloca(llvm_ty, name)?;
        set_alignment(ptr, ty.alignment());

        if let Some(prev) = prev_pos {
            self.builder.position_at_end(prev);
        }
        Ok(ptr)
    }

    fn build_alloca_value(
        &self,
        val: impl BasicValue<'ctx>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let val = val.as_basic_value_enum();
        let alloca = self.build_alloca(val.get_type(), "", ty).u();
        self.build_store(alloca, val, ty.alignment())?;
        Ok(alloca)
    }

    fn build_ptr_to_sym(
        &self,
        sym: Symbol<'ctx>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<PointerValue<'ctx>> {
        Ok(match sym {
            Symbol::Stack(ptr_value) => ptr_value,
            Symbol::Register(val) => {
                #[cfg(debug_assertions)]
                println!("INFO: doing stack allocation for register value: {val:?}");
                self.build_alloca_value(val.basic_val(), ty)?
            },
            Symbol::Function { val, .. } => val.as_global_value().as_pointer_value(),
            Symbol::Global { val } => val.as_pointer_value(),
            Symbol::Void | Symbol::Never | Symbol::Module => unreachable_debug(),
        })
    }

    fn build_store(
        &self,
        ptr: PointerValue<'ctx>,
        value: impl BasicValue<'ctx>,
        alignment: usize,
    ) -> CodegenResult<InstructionValue<'ctx>> {
        let build_instruction = self.builder.build_store(ptr, value)?;
        set_alignment(build_instruction, alignment);
        Ok(build_instruction)
    }

    fn build_load(
        &self,
        pointee_ty: impl BasicType<'ctx>,
        ptr: PointerValue<'ctx>,
        name: &str,
        alignment: usize,
    ) -> CodegenResult<BasicValueEnum<'ctx>> {
        let out = self.builder.build_load(pointee_ty, ptr, name)?;
        set_alignment(out, alignment);
        Ok(out)
    }

    fn build_return(
        &mut self,
        ret_sym: Symbol<'ctx>,
        ret_ty: Ptr<ast::Type>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let ret = match (ret_sym, self.ret_type(ret_ty)) {
            (Symbol::Never, _) => return Ok(Symbol::Never),
            (_, RetType::Zst) => None,
            (Symbol::Stack(ptr), RetType::SRetParam) => {
                let sret_ptr = self.sret_ptr.u();
                let alignment = ret_ty.alignment() as u32;
                self.builder.build_memcpy(
                    sret_ptr,
                    alignment,
                    ptr,
                    alignment,
                    self.context.i64_type().const_int(ret_ty.size() as u64, false),
                )?;
                None
            },
            (Symbol::Register(val), RetType::SRetParam) => {
                let sret_ptr = self.sret_ptr.u();
                self.build_store(sret_ptr, val.basic_val(), ret_ty.alignment())?;
                None
            },
            (Symbol::Stack(ptr), RetType::Basic(ty)) => {
                Some(self.build_load(ty, ptr, "", ret_ty.alignment())?)
            },
            (Symbol::Register(val), RetType::Basic(llvm_ty)) if ret_ty.is_aggregate() => {
                // Note: llvm_ty and ret_ty might be different.
                // TODO: improve this
                let ret = self.build_alloca(llvm_ty, "ret", ret_ty)?;
                self.build_store(ret, val.basic_val(), ret_ty.alignment())?;
                Some(self.builder.build_load(llvm_ty, ret, "ret")?)
            },
            (Symbol::Register(val), RetType::Basic(_)) => Some(val.basic_val()),
            _ => panic_debug("unexpected symbol"),
        };
        match ret {
            Some(ret) => self.builder.build_return(Some(&ret))?,
            None => self.builder.build_return(None)?,
        };
        Ok(Symbol::Never)
    }

    #[inline]
    fn build_gep(
        &self,
        pointee_ty: impl BasicType<'ctx>,
        ptr: PointerValue<'ctx>,
        ordered_indexes: &[IntValue<'ctx>],
    ) -> CodegenResult<PointerValue<'ctx>> {
        Ok(unsafe { self.builder.build_in_bounds_gep(pointee_ty, ptr, ordered_indexes, "")? })
    }

    fn build_struct_access(
        &self,
        struct_ty: StructType<'ctx>,
        struct_sym: Symbol<'ctx>,
        idx: u32,
        name: &str,
    ) -> CodegenResult<Symbol<'ctx>> {
        match struct_sym {
            Symbol::Never => todo!(),
            Symbol::Stack(ptr) => {
                stack_val(self.builder.build_struct_gep(struct_ty, ptr, idx, name)?)
            },
            Symbol::Register(val) => {
                reg(self.builder.build_extract_value(val.struct_val(), idx, name)?)
            },
            _ => unreachable_debug(),
        }
    }

    fn build_slice_field_access(
        &mut self,
        slice_sym: Symbol<'ctx>,
    ) -> CodegenResult<(PointerValue<'ctx>, IntValue<'ctx>)> {
        let p = primitives();
        let slice_ty = self.slice_ty();
        let ptr = self.build_struct_access(slice_ty, slice_sym, 0, "")?;
        let ptr = self.sym_as_val(ptr, p.never_ptr_ty)?.ptr_val();
        let len = self.build_struct_access(slice_ty, slice_sym, 1, "")?;
        let len = self.sym_as_val(len, p.u64)?.int_val();
        Ok((ptr, len))
    }

    pub fn build_int_binop<I: IntMathValue<'ctx>>(
        &mut self,
        lhs: I,
        rhs: I,
        is_signed: bool,
        op: BinOpKind,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
            Ok(CodegenValue::new(val.as_value_ref()))
        }

        macro_rules! cmp {
            ($pred:ident, $var_name:expr) => {
                ret(self.builder.build_int_compare(IntPredicate::$pred, lhs, rhs, $var_name)?)
            };
        }

        match op {
            BinOpKind::Mul => ret(self.builder.build_int_mul(lhs, rhs, "mul")?),
            BinOpKind::Div => ret(if is_signed {
                self.builder.build_int_signed_div(lhs, rhs, "div")?
            } else {
                self.builder.build_int_unsigned_div(lhs, rhs, "div")?
            }),
            BinOpKind::Mod => ret(if is_signed {
                self.builder.build_int_signed_rem(lhs, rhs, "mod")?
            } else {
                self.builder.build_int_unsigned_rem(lhs, rhs, "mod")?
            }),
            BinOpKind::Add => ret(self.builder.build_int_add(lhs, rhs, "add")?),
            BinOpKind::Sub => ret(self.builder.build_int_sub(lhs, rhs, "sub")?),
            BinOpKind::ShiftL => ret(self.builder.build_left_shift(lhs, rhs, "shl")?),
            BinOpKind::ShiftR => ret(self.builder.build_right_shift(lhs, rhs, is_signed, "shr")?),
            BinOpKind::BitAnd => ret(self.builder.build_and(lhs, rhs, "bitand")?),
            BinOpKind::BitXor => ret(self.builder.build_xor(lhs, rhs, "bitxor")?),
            BinOpKind::BitOr => ret(self.builder.build_or(lhs, rhs, "bitor")?),
            BinOpKind::Eq => cmp!(EQ, "eq"),
            BinOpKind::Ne => cmp!(NE, "ne"),
            BinOpKind::Lt if is_signed => cmp!(SLT, "lt"),
            BinOpKind::Lt => cmp!(ULT, "lt"),
            BinOpKind::Le if is_signed => cmp!(SLE, "le"),
            BinOpKind::Le => cmp!(ULE, "le"),
            BinOpKind::Gt if is_signed => cmp!(SGT, "gt"),
            BinOpKind::Gt => cmp!(UGT, "gt"),
            BinOpKind::Ge if is_signed => cmp!(SGE, "ge"),
            BinOpKind::Ge => cmp!(UGE, "ge"),

            BinOpKind::And | BinOpKind::Or => unreachable!(),
        }
    }

    pub fn build_float_binop(
        &mut self,
        lhs: FloatValue<'ctx>,
        rhs: FloatValue<'ctx>,
        op: BinOpKind,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
            Ok(CodegenValue::new(val.as_value_ref()))
        }

        macro_rules! cmp {
            ($pred:ident, $var_name:expr) => {
                ret(self.builder.build_float_compare(FloatPredicate::$pred, lhs, rhs, $var_name)?)
            };
        }

        match op {
            BinOpKind::Mul => ret(self.builder.build_float_mul(lhs, rhs, "mul")?),
            BinOpKind::Div => ret(self.builder.build_float_div(lhs, rhs, "div")?),
            BinOpKind::Mod => ret(self.builder.build_float_rem(lhs, rhs, "mod")?),
            BinOpKind::Add => ret(self.builder.build_float_add(lhs, rhs, "add")?),
            BinOpKind::Sub => ret(self.builder.build_float_sub(lhs, rhs, "sub")?),
            BinOpKind::Eq => cmp!(OEQ, "eq"),
            BinOpKind::Ne => cmp!(ONE, "ne"),
            BinOpKind::Lt => cmp!(OLT, "lt"),
            BinOpKind::Le => cmp!(OLE, "le"),
            BinOpKind::Gt => cmp!(OGT, "gt"),
            BinOpKind::Ge => cmp!(OGE, "ge"),
            _ => unreachable!(),
        }
    }

    fn build_bool_binop(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
        op: BinOpKind,
    ) -> CodegenResult<Symbol<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
            Ok(Symbol::Register(CodegenValue::new(val.as_value_ref())))
        }

        match op {
            BinOpKind::Eq => {
                ret(self.builder.build_int_compare(IntPredicate::EQ, lhs, rhs, "eq")?)
            },
            BinOpKind::Ne => {
                ret(self.builder.build_int_compare(IntPredicate::NE, lhs, rhs, "ne")?)
            },

            // false = 0, true = 1
            BinOpKind::Lt => {
                ret(self.builder.build_int_compare(IntPredicate::ULT, lhs, rhs, "lt")?)
            },
            BinOpKind::Le => {
                ret(self.builder.build_int_compare(IntPredicate::ULE, lhs, rhs, "le")?)
            },
            BinOpKind::Gt => {
                ret(self.builder.build_int_compare(IntPredicate::UGT, lhs, rhs, "gt")?)
            },
            BinOpKind::Ge => {
                ret(self.builder.build_int_compare(IntPredicate::UGE, lhs, rhs, "ge")?)
            },
            BinOpKind::And | BinOpKind::Or => unreachable_debug(),
            BinOpKind::BitAnd => ret(self.builder.build_and(lhs, rhs, "and")?),
            BinOpKind::BitOr => ret(self.builder.build_or(lhs, rhs, "or")?),
            BinOpKind::BitXor => ret(self.builder.build_xor(lhs, rhs, "xor")?),
            _ => unreachable!(),
        }
    }

    fn build_bool_short_circuit_binop(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: Ptr<Ast>,
        op: BinOpKind,
    ) -> CodegenResult<Symbol<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
            Ok(Symbol::Register(CodegenValue::new(val.as_value_ref())))
        }

        match op {
            BinOpKind::And => ret({
                let func = self.cur_fn.u();
                let entry_bb = self.builder.get_insert_block().u();
                let mut rhs_bb = self.context.append_basic_block(func, "and.rhs");
                let merge_bb = self.context.append_basic_block(func, "and.merge");

                self.builder.build_conditional_branch(lhs, rhs_bb, merge_bb)?;

                self.builder.position_at_end(rhs_bb);
                debug_assert_eq!(rhs.ty, primitives().bool);
                let rhs = try_compile_expr_as_val!(self, rhs).bool_val();
                self.builder.build_unconditional_branch(merge_bb)?;
                rhs_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(lhs.get_type(), "and")?;
                let false_ = self.bool_val(false);
                phi.add_incoming(&[(&false_, entry_bb), (&rhs, rhs_bb)]);
                phi
            }),
            BinOpKind::Or => ret({
                let func = self.cur_fn.u();
                let entry_bb = self.builder.get_insert_block().u();
                let mut rhs_bb = self.context.append_basic_block(func, "or.rhs");
                let merge_bb = self.context.append_basic_block(func, "or.merge");

                self.builder.build_conditional_branch(lhs, merge_bb, rhs_bb)?;

                self.builder.position_at_end(rhs_bb);
                debug_assert_eq!(rhs.ty, primitives().bool);
                let rhs = try_compile_expr_as_val!(self, rhs).bool_val();
                self.builder.build_unconditional_branch(merge_bb)?;
                rhs_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(lhs.get_type(), "and")?;
                let true_ = self.bool_val(true);
                phi.add_incoming(&[(&true_, entry_bb), (&rhs, rhs_bb)]);
                phi
            }),
            _ => unreachable!(),
        }
    }

    fn bool_val(&self, val: bool) -> IntValue<'ctx> {
        let b_ty = self.context.bool_type();
        if val { b_ty.const_all_ones() } else { b_ty.const_zero() }
    }

    fn build_slice(
        &mut self,
        ptr: PointerValue<'ctx>,
        len: IntValue<'ctx>,
    ) -> CodegenResult<Symbol<'ctx>> {
        let slice = self.slice_ty().get_undef();
        let slice = self.builder.build_insert_value(slice, ptr, 0, "")?;
        reg(self.builder.build_insert_value(slice, len, 1, "slice")?)
    }

    /// # Usage
    /// ```rust,ignore
    /// let for_info = self.build_for(/* ... */)?;
    /// // build for body
    /// self.build_for(for_info);
    /// ```
    #[must_use]
    fn build_for(
        &mut self,
        idx_ty: IntType<'ctx>,
        idx_is_signed: bool,
        start_idx: IntValue<'ctx>,
        end_idx: IntValue<'ctx>,
        is_end_inclusive: bool,
    ) -> CodegenResult<ForInfo<'ctx>> {
        let func = self.cur_fn.u();
        let entry_bb = self.builder.get_insert_block().u();
        let cond_bb = self.context.append_basic_block(func, "for.cond");
        let body_bb = self.context.append_basic_block(func, "for.body");
        let inc_bb = self.context.append_basic_block(func, "for.inc");
        let end_bb = self.context.append_basic_block(func, "for.end");

        // entry
        self.builder.build_unconditional_branch(cond_bb)?;

        // cond
        self.builder.position_at_end(cond_bb);
        let idx = self.builder.build_phi(idx_ty, "for.idx")?;
        idx.add_incoming(&[(&start_idx, entry_bb)]);
        let idx_int = idx.as_basic_value().into_int_value();
        //self.symbols.insert("idx", reg_sym(idx_int));

        let cmp_op = match (is_end_inclusive, idx_is_signed) {
            (true, true) => IntPredicate::SLE,
            (true, false) => IntPredicate::ULE,
            (false, true) => IntPredicate::SLT,
            (false, false) => IntPredicate::ULT,
        };
        let cond = self.builder.build_int_compare(cmp_op, idx_int, end_idx, "for.idx_cmp")?;
        self.builder.build_conditional_branch(cond, body_bb, end_bb)?;

        // body
        self.builder.position_at_end(body_bb);
        let outer_loop = self.cur_loop.replace(Loop { continue_bb: inc_bb, end_bb });
        Ok(ForInfo { cond_bb, inc_bb, end_bb, idx, idx_ty, idx_int, outer_loop })
    }

    fn build_for_end(
        &mut self,
        for_info: ForInfo<'ctx>,
        body_out_sym: Symbol<'ctx>,
    ) -> CodegenResult<()> {
        let ForInfo { cond_bb, inc_bb, end_bb, idx, idx_ty, idx_int, outer_loop } = for_info;
        self.cur_loop = outer_loop;
        if !matches!(body_out_sym, Symbol::Never) {
            self.builder.build_unconditional_branch(inc_bb)?;
        }

        // inc
        self.builder.position_at_end(inc_bb);
        let next_idx =
            self.builder.build_int_add(idx_int, idx_ty.const_int(1, false), "next_idx")?;
        idx.add_incoming(&[(&next_idx, inc_bb)]);
        self.builder.build_unconditional_branch(cond_bb)?;

        // end
        self.builder.position_at_end(end_bb);
        Ok(())
    }

    fn max_int(&self, int_ty: IntType<'ctx>, is_signed: bool) -> CodegenResult<IntValue<'ctx>> {
        let all_ones = int_ty.const_all_ones();
        Ok(if is_signed {
            self.builder
                .build_right_shift(all_ones, int_ty.const_int(1, false), false, "")?
        } else {
            all_ones
        })
    }

    /// LLVM does not make a distinction between signed and unsigned integer
    /// type
    fn int_type(&self, bits: u32) -> IntType<'ctx> {
        // See <https://llvm.org/docs/Frontend/PerformanceTips.html#avoid-loads-and-stores-of-non-byte-sized-types>
        match bits {
            0 => self.context.custom_width_int_type(bits),
            ..=8 => self.context.i8_type(),
            ..=16 => self.context.i16_type(),
            ..=32 => self.context.i32_type(),
            ..=64 => self.context.i64_type(),
            ..=128 => self.context.i128_type(),
            bits => self.context.custom_width_int_type(bits),
        }
    }

    fn float_type(&self, bits: u32) -> FloatType<'ctx> {
        match bits {
            16 => self.context.f16_type(),
            32 => self.context.f32_type(),
            64 => self.context.f64_type(),
            128 => self.context.f128_type(),
            bits => todo!("{bits}-bit float"),
        }
    }

    fn ptr_type(&self) -> PointerType<'ctx> {
        self.context.ptr_type(AddressSpace::default())
    }

    fn new_struct_type(&mut self, fields: DeclList, name: Option<Ptr<str>>) -> StructType<'ctx> {
        let field_types = fields
            .iter_types()
            .filter(|ty| ty.size() > 0)
            .map(|ty| self.llvm_type(ty).basic_ty())
            .collect::<Vec<_>>();
        self.struct_type_inner(&field_types, name, false)
    }

    fn struct_type_inner<'a>(
        &self,
        fields: &[BasicTypeEnum<'ctx>],
        name: Option<Ptr<str>>,
        packed: bool,
    ) -> StructType<'ctx> {
        match name.as_ref().map(Ptr::as_ref) {
            Some(name) => {
                let ty = self.context.opaque_struct_type(name);
                if !ty.set_body(fields, packed) {
                    panic!("invalid struct type")
                }
                ty
            },
            None => self.context.struct_type(fields, packed),
        }
    }

    fn new_union_type(&mut self, fields: DeclList, name: Option<Ptr<str>>) -> StructType<'ctx> {
        for f in fields.iter() {
            // if f.var_ty is a custom type, this will compile the type.
            let _ = self.llvm_type(f.var_ty.u());
        }
        let fields = if let Some(biggest_alignment_field) = fields
            .iter_types()
            .filter(|t| t.size() > 0)
            .max_by(|a, b| a.alignment().cmp(&b.alignment()))
        {
            let remaining_size = union_size(fields) - biggest_alignment_field.size();
            let biggest_alignment_field = self.llvm_type(biggest_alignment_field).basic_ty();
            let remaining_size_field =
                self.context.i8_type().array_type((remaining_size) as u32).as_basic_type_enum();
            &[biggest_alignment_field, remaining_size_field] as &[_]
        } else {
            &[]
        };
        self.struct_type_inner(fields, name, false)
    }

    fn new_enum_type(&mut self, variants: DeclList, name: Option<Ptr<str>>) -> StructType<'ctx> {
        let mut fields = Vec::with_capacity(2);
        fields.push(self.enum_tag_type(variants).as_basic_type_enum());
        if !is_simple_enum(variants) {
            fields.push(self.new_union_type(variants, None).as_basic_type_enum());
        }
        self.struct_type_inner(&fields, name, false)
    }

    #[inline]
    fn enum_tag_type(&mut self, variants: DeclList) -> IntType<'ctx> {
        let variant_bits = util::variant_count_to_tag_size_bits(variants.len());
        self.int_type(variant_bits)
    }

    #[inline]
    fn range_type(&mut self, range_ty: Ptr<ast::RangeTy>) -> StructType<'ctx> {
        let e = self.llvm_type(range_ty.elem_ty).basic_ty();
        let fields = &[e; 2][..range_ty.rkind.get_field_count()];
        self.struct_type_inner(fields, None, false)
    }

    #[inline]
    fn slice_ty(&self) -> StructType<'ctx> {
        self.struct_type_inner(
            &[self.ptr_type().as_basic_type_enum(), self.context.i64_type().as_basic_type_enum()],
            None,
            false,
        )
    }

    fn llvm_type(&mut self, ty: Ptr<ast::Type>) -> CodegenType<'ctx> {
        macro_rules! t {
            ($t:expr) => {
                CodegenType::new(
                    $t.as_type_ref(),
                    #[cfg(debug_assertions)]
                    ty,
                )
            };
        }
        let p = primitives();
        if ty.kind == AstKind::SimpleTy {
            return if ty == p.void_ty {
                t!(self.context.void_type())
            } else if ty == p.never {
                unreachable_debug()
            } else if ty == p.bool {
                t!(self.context.bool_type())
            } else if ty == p.char {
                todo!("char ty")
            } else if ty == p.str_slice_ty {
                todo!("str_slice_ty")
            } else if ty == p.type_ty {
                todo!("type_ty")
            } else {
                unreachable_debug()
            };
        }

        if let Some(t) = self.type_table.get(&ty) {
            return *t;
        }

        let llvm_ty = match ty.matchable().as_ref() {
            TypeEnum::SimpleTy { .. } => unreachable_debug(),
            TypeEnum::IntTy { bits, .. } => t!(self.int_type(*bits)),
            TypeEnum::FloatTy { bits, .. } => t!(self.float_type(*bits)),
            TypeEnum::PtrTy { .. } => t!(self.ptr_type()),
            /*
            TypeEnum::SliceTy { .. } => panic!(
                "please pass `self.primitives.untyped_slice_struct_def` the `llvm_type` instead \
                 of the `ast::SliceTy`"
            ),
            */
            TypeEnum::SliceTy { .. } => self.llvm_type(p.untyped_slice_struct_def.upcast_to_type()),
            TypeEnum::ArrayTy { len, elem_ty, .. } => {
                t!(self.llvm_type(elem_ty.downcast_type()).basic_ty().array_type(len.int()))
            },
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } => t!(self.new_struct_type(*fields, None)),
            TypeEnum::UnionDef { fields, .. } => t!(self.new_union_type(*fields, None)),
            TypeEnum::EnumDef { variants, .. } => t!(self.new_enum_type(*variants, None)),
            TypeEnum::RangeTy { .. } => t!(self.range_type(ty.downcast())),
            TypeEnum::OptionTy { inner_ty: t, .. } if t.downcast_type().is_non_null() => {
                self.llvm_type(t.downcast_type())
            },
            TypeEnum::OptionTy { .. } => {
                todo!()
                /*
                const NONE_IDENT: Ptr<ast::Ident> =
                    Ptr::from_ref(&ast::Ident::new(Ptr::from_ref("None"), Span::ZERO));
                const SOME_IDENT: Ptr<ast::Ident> =
                    Ptr::from_ref(&ast::Ident::new(Ptr::from_ref("Some"), Span::ZERO));
                const LLVM_OPTION_VARIANTS: Ptr<[Ptr<ast::Decl>]> = Ptr::from_ref(&[
                    Ptr::from_ref(&ast::Decl::new_internal(NONE_IDENT, TypeEnum::Void)),
                    Ptr::from_ref(&ast::Decl::new_internal(SOME_IDENT, TypeEnum::PtrTy {
                        pointee_ty: TypeEnum::UNSET,
                        is_mut: false,
                    })),
                ]);
                t!(self.new_enum_type(LLVM_OPTION_VARIANTS, None))
                    */
            },
            TypeEnum::Fn { .. } => todo!(),
            TypeEnum::Unset => unreachable_debug(),
        };

        let old_entry = self.type_table.insert(ty, llvm_ty);
        debug_assert!(old_entry.is_none());

        llvm_ty
    }

    /// I gave up trying to implement the C calling convention.
    ///
    /// [`None`] means void
    ///
    /// See <https://discourse.llvm.org/t/questions-about-c-calling-conventions/72414>
    /// TODO: See <https://mcyoung.xyz/2024/04/17/calling-convention/>
    fn ret_type(&mut self, ty: Ptr<ast::Type>) -> RetType<'ctx> {
        let size = ty.size();
        match ty.matchable().as_ref() {
            _ if size == 0 => RetType::Zst,
            _ if size > 16 => RetType::SRetParam,
            //ty if ty.is_aggregate() => RetType::SRetParam,
            TypeEnum::StructDef { fields, .. } => {
                self.ret_type_for_struct(fields.iter_types(), size)
            },
            TypeEnum::UnionDef { .. } | TypeEnum::EnumDef { .. } => RetType::Basic(
                self.context
                    .struct_type(&[self.int_type(size as u32 * 8).as_basic_type_enum()], false)
                    .as_basic_type_enum(),
            ),
            /*
            TypeEnum::UnionDef { .. } => {
                RetType::Basic(self.int_type(size as u32 * 8).as_basic_type_enum())
            },
            TypeEnum::EnumDef { variants, .. } => {
                /*
                let fields = [
                    TypeEnum::IntTy {
                        bits: util::variant_count_to_tag_size_bits(variants.len()),
                        is_signed: false,
                    },
                    TypeEnum::UnionDef { fields: *variants, .. },
                ];
                self.ret_type_for_struct(fields.into_iter(), size)
                */
                RetType::Basic(self.int_type(size as u32 * 8).as_basic_type_enum())
            },
            */
            _ => RetType::Basic(self.llvm_type(ty).basic_ty()),
        }
    }

    fn ret_type_for_struct<'a>(
        &mut self,
        field_types: impl DoubleEndedIterator<Item = Ptr<ast::Type>>,
        size: usize,
    ) -> RetType<'ctx> {
        #[derive(Debug, PartialEq)]
        enum PrevState {
            None,
            Int,
            Float,
            FloatFloat,
        }

        let mut new_fields = Vec::with_capacity(size);
        let mut prev_state = PrevState::None;
        let mut prev_bytes: u32 = 0;

        macro_rules! push_prev_state_to_new_fields {
            () => {
                match prev_state {
                    PrevState::None => {},
                    PrevState::Int => new_fields.push(
                        self.context.custom_width_int_type(prev_bytes << 3).as_basic_type_enum(),
                    ),
                    PrevState::Float => {
                        new_fields.push(self.context.f32_type().as_basic_type_enum())
                    },
                    PrevState::FloatFloat => {
                        new_fields.push(self.context.f32_type().vec_type(2).as_basic_type_enum())
                    },
                }
            };
        }

        for f in field_types {
            if f.size() == 0 {
                continue;
            }

            if get_aligned_offset!(prev_bytes, f.alignment() as u32) >= 8 {
                // finished the first 8 bytes
                push_prev_state_to_new_fields!();
                prev_state = PrevState::None;
                prev_bytes = 0;
            }

            let padding = get_padding!(prev_bytes, f.alignment() as u32);
            prev_bytes += padding;

            macro_rules! handle_int {
                ($bits:expr) => {{
                    prev_state = PrevState::Int;
                    prev_bytes += $bits.div_ceil(8);
                }};
            }

            match f.matchable().as_ref() {
                TypeEnum::SimpleTy { .. } => {
                    if f == primitives().bool {
                        prev_state = PrevState::Int;
                        prev_bytes += 1;
                    } else {
                        unreachable_debug()
                    }
                },
                TypeEnum::IntTy { bits, .. } => handle_int!(bits),
                TypeEnum::FloatTy { bits: 64, .. } => {
                    debug_assert_eq!(prev_state, PrevState::None);
                    debug_assert_eq!(prev_bytes, 0);
                    new_fields.push(self.context.f64_type().as_basic_type_enum());
                },
                TypeEnum::FloatTy { bits: 32, .. } => {
                    prev_bytes += 4;
                    prev_state = match prev_state {
                        PrevState::None => PrevState::Float,
                        PrevState::Int => PrevState::Int,
                        PrevState::Float => PrevState::FloatFloat,
                        PrevState::FloatFloat => unreachable_debug(),
                    }
                },
                TypeEnum::FloatTy { .. } => todo!("float with other sizes"),
                TypeEnum::PtrTy { .. } => {
                    debug_assert_eq!(prev_state, PrevState::None);
                    debug_assert_eq!(prev_bytes, 0);
                    new_fields.push(self.llvm_type(f).basic_ty()) // `handle_int!(f.size() as u32 * 8),` // also works, but clang uses `ptr`
                },
                TypeEnum::SliceTy { .. }
                | TypeEnum::ArrayTy { .. }
                | TypeEnum::StructDef { .. }
                | TypeEnum::UnionDef { .. }
                | TypeEnum::EnumDef { .. } => handle_int!(f.size() as u32 * 8),
                _ => todo!("{f:#?}"),
                //_ => unreachable_debug(),
                //_ => new_fields.push(self.llvm_type(f).basic_ty()),
            }
        }

        // finished the last 8 bytes
        push_prev_state_to_new_fields!();

        RetType::Basic(self.struct_type_inner(&new_fields, None, false).as_basic_type_enum())
    }

    fn sym_as_val(
        &mut self,
        sym: Symbol<'ctx>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        if let Symbol::Register(val) = sym {
            Ok(val)
        } else {
            let llvm_ty = self.llvm_type(ty);
            self.sym_as_val_with_llvm_ty(sym, llvm_ty.basic_ty(), ty.alignment())
        }
    }

    fn sym_as_val_with_llvm_ty(
        &mut self,
        sym: Symbol<'ctx>,
        llvm_ty: BasicTypeEnum<'ctx>,
        alignment: usize,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        Ok(match sym {
            Symbol::Stack(ptr) => {
                CodegenValue::new(self.build_load(llvm_ty, ptr, "", alignment)?.as_value_ref())
            },
            Symbol::Register(val) => val,
            Symbol::Global { val } => {
                let val = self.build_load(llvm_ty, val.as_pointer_value(), "", alignment)?;
                CodegenValue::new(val.as_value_ref())
            },
            _ => panic_debug("unexpected symbol"),
        })
    }

    fn sym_as_val_checked(
        &mut self,
        sym: Symbol<'ctx>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<Option<CodegenValue<'ctx>>> {
        Ok(Some(match sym {
            Symbol::Stack(ptr) => {
                let llvm_ty = self.llvm_type(ty).basic_ty();
                CodegenValue::new(self.build_load(llvm_ty, ptr, "", ty.alignment())?.as_value_ref())
            },
            Symbol::Register(val) => val,
            _ => return Ok(None),
        }))
    }

    #[inline]
    fn get_symbol(&self, decl: Ptr<ast::Decl>) -> Symbol<'ctx> {
        *self.symbols.get(decl).u()
    }

    pub fn init_target_machine(target_triple: Option<&str>) -> TargetMachine {
        Target::initialize_all(&InitializationConfig::default());

        let target_triple = target_triple
            .map(TargetTriple::create)
            .unwrap_or_else(TargetMachine::get_default_triple);
        let target = Target::from_triple(&target_triple).unwrap();

        let cpu = "generic";
        let features = "";
        target
            .create_target_machine(
                &target_triple,
                cpu,
                features,
                OptimizationLevel::Aggressive,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .unwrap()
    }

    fn open_scope(&mut self) {
        self.symbols.open_scope();
        self.defer_stack.open_scope();
        self.return_depth += 1;
        self.continue_break_depth += 1;
    }

    fn close_scope(&mut self, do_compile_defer: bool) -> CodegenResult<()> {
        if do_compile_defer && !self.compile_defer_exprs()? {
            todo!()
        }
        self.symbols.close_scope();
        self.defer_stack.close_scope();
        self.return_depth -= 1;
        self.continue_break_depth -= 1;
        Ok(())
    }

    /// the [`bool`] in the return type describes whether compilation can continue or not
    #[inline]
    fn compile_defer_exprs(&mut self) -> CodegenResult<bool> {
        let exprs = unsafe { forget_lifetime(self.defer_stack.get_cur_scope()) };
        for expr in exprs.iter().rev() {
            let s = self.compile_expr(*expr)?;
            if s == Symbol::Never {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn compile_multiple_defer_scopes(&mut self, depth: usize) -> CodegenResult<bool> {
        let defer_stack = unsafe { forget_lifetime(&self.defer_stack) };
        for scope in defer_stack.iter_scopes().take(depth) {
            for expr in scope.iter().rev() {
                let s = self.compile_expr(*expr)?;
                if s == Symbol::Never {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn position_builder_at_start(&self, entry: BasicBlock<'ctx>) {
        match entry.get_first_instruction() {
            Some(first_instr) => self.builder.position_before(&first_instr),
            None => self.builder.position_at_end(entry),
        }
    }

    #[inline]
    fn module(&self) -> &Module<'ctx> {
        self.module.as_ref().u()
    }
}

pub trait CodegenModuleExt {
    fn optimize(&self, target_machine: &TargetMachine, level: u8) -> CodegenResult<()>;

    fn compile_to_obj_file(
        &self,
        target_machine: &TargetMachine,
        obj_file_path: &Path,
    ) -> Result<(), CodegenError>;

    fn jit_run_fn<Ret>(&self, fn_name: &str, opt: OptimizationLevel) -> CodegenResult<Ret>;
}

// optimizations
impl<'ctx> CodegenModuleExt for Module<'ctx> {
    fn optimize(&self, target_machine: &TargetMachine, level: u8) -> CodegenResult<()> {
        assert!((0..=3).contains(&level));
        let passes = format!("default<O{}>", level);

        // TODO: custom passes:
        //let passes = format!(
        //   "module(cgscc(inline),function({}))",
        //   ["instcombine", "reassociate", "gvn", "simplifycfg",
        //"mem2reg",].join(","), );

        Ok(self
            .run_passes(&passes, target_machine, PassBuilderOptions::create())
            .map_err(|err| CodegenError::CannotOptimizeModule(err))?)
    }

    fn compile_to_obj_file(
        &self,
        target_machine: &TargetMachine,
        obj_file_path: &Path,
    ) -> Result<(), CodegenError> {
        std::fs::create_dir_all(obj_file_path.parent().unwrap()).unwrap();
        target_machine
            .write_to_file(&self, inkwell::targets::FileType::Object, obj_file_path)
            .map_err(|err| CodegenError::CannotCompileObjFile(err))
    }

    fn jit_run_fn<Ret>(&self, fn_name: &str, opt: OptimizationLevel) -> CodegenResult<Ret> {
        match self.create_jit_execution_engine(opt) {
            Ok(jit) => {
                Ok(unsafe { jit.get_function::<unsafe extern "C" fn() -> Ret>(fn_name)?.call() })
            },
            Err(err) => Err(CodegenError::CannotCreateJit(err))?,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symbol<'ctx> {
    Void,
    /// This Symbol means that no more instructions can be added the the current BasicBlock
    Never,
    Stack(PointerValue<'ctx>),
    //Register(AnyValueEnum<'ctx>),
    Register(CodegenValue<'ctx>),
    Global {
        val: GlobalValue<'ctx>,
    },
    Function {
        val: FunctionValue<'ctx>,
    },
    Module,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CodegenValue<'ctx> {
    val: *mut LLVMValue,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> std::fmt::Debug for CodegenValue<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&unsafe { AnyValueEnum::new(self.val) }, f)
    }
}

impl<'ctx> CodegenValue<'ctx> {
    pub fn new(val: *mut LLVMValue) -> CodegenValue<'ctx> {
        #[cfg(debug_assertions)]
        unsafe {
            AnyValueEnum::new(val)
        };

        CodegenValue { val, _marker: PhantomData }
    }

    // /// for zero sized types: `self.val == null`
    // pub fn new_zst(ty: Type) -> CodegenValue<'ctx> {
    //     debug_assert!(ty.u().matches_void());
    //     CodegenValue::new(std::ptr::null_mut(), ty)
    // }

    // pub fn try_new_zst(ty: Type) -> Option<CodegenValue<'ctx>> {
    //     match ty {
    //         TypeEnum::Void | TypeEnum::Never => Some(CodegenValue::new_zst(ty)),
    //         _ => None,
    //     }
    // }

    // pub fn as_type(&self) -> CodegenType {
    //     CodegenTypeEnum::new(unsafe { LLVMTypeOf(self.val) }, self.ty)
    // }

    pub fn int_val(&self) -> IntValue<'ctx> {
        debug_assert!(unsafe { AnyValueEnum::new(self.val) }.is_int_value());
        unsafe { IntValue::new(self.val) }
    }

    pub fn bool_val(&self) -> IntValue<'ctx> {
        debug_assert!(unsafe { AnyValueEnum::new(self.val) }.is_int_value());
        unsafe { IntValue::new(self.val) }
    }

    pub fn float_val(&self) -> FloatValue<'ctx> {
        debug_assert!(unsafe { AnyValueEnum::new(self.val) }.is_float_value());
        unsafe { FloatValue::new(self.val) }
    }

    pub fn ptr_val(&self) -> PointerValue<'ctx> {
        debug_assert!(unsafe { AnyValueEnum::new(self.val) }.is_pointer_value());
        unsafe { PointerValue::new(self.val) }
    }

    pub fn struct_val(&self) -> StructValue<'ctx> {
        debug_assert!(unsafe { AnyValueEnum::new(self.val) }.is_struct_value());
        unsafe { StructValue::new(self.val) }
    }

    pub fn basic_val(&self) -> BasicValueEnum<'ctx> {
        debug_assert!(BasicValueEnum::try_from(unsafe { AnyValueEnum::new(self.val) }).is_ok());
        unsafe { BasicValueEnum::new(self.val) }
    }

    pub fn basic_metadata_val(&self) -> BasicMetadataValueEnum<'ctx> {
        // For some reason `function_value.as_global_value().as_pointer_value().as_any_value_enum().is_pointer_value()` is `false`
        match self.any_val() {
            AnyValueEnum::FunctionValue(function_value) => BasicMetadataValueEnum::PointerValue(
                function_value.as_global_value().as_pointer_value(),
            ),
            v => BasicMetadataValueEnum::try_from(v).u(),
        }
    }

    pub fn any_val(&self) -> AnyValueEnum<'ctx> {
        unsafe { AnyValueEnum::new(self.val) }
    }

    // pub fn is_never(&self) -> bool {
    //     self.ty == TypeEnum::Never
    // }
}

unsafe impl<'ctx> AsValueRef for CodegenValue<'ctx> {
    fn as_value_ref(&self) -> LLVMValueRef {
        self.val
    }
}

unsafe impl<'ctx> AnyValue<'ctx> for CodegenValue<'ctx> {}

#[derive(Debug, Clone, Copy)]
pub struct CodegenType<'ctx> {
    inner: *mut LLVMType,
    #[cfg(debug_assertions)]
    sema_ty: Ptr<ast::Type>,
    _marker: PhantomData<&'ctx ()>,
}

impl<'ctx> CodegenType<'ctx> {
    pub fn new(
        raw: *mut LLVMType,
        #[cfg(debug_assertions)] sema_ty: Ptr<ast::Type>,
    ) -> CodegenType<'ctx> {
        CodegenType {
            inner: raw,
            #[cfg(debug_assertions)]
            sema_ty,
            _marker: PhantomData,
        }
    }

    pub fn int_ty(&self) -> IntType<'ctx> {
        #[cfg(debug_assertions)]
        debug_assert!(self.sema_ty.kind == AstKind::IntTy);
        unsafe { IntType::new(self.inner) }
    }

    pub fn float_ty(&self) -> FloatType<'ctx> {
        #[cfg(debug_assertions)]
        debug_assert!(self.sema_ty.kind == AstKind::FloatTy);
        unsafe { FloatType::new(self.inner) }
    }

    pub fn ptr_ty(&self) -> PointerType<'ctx> {
        #[cfg(debug_assertions)]
        debug_assert!(self.sema_ty.kind == AstKind::PtrTy);
        unsafe { PointerType::new(self.inner) }
    }

    pub fn arr_ty(&self) -> ArrayType<'ctx> {
        #[cfg(debug_assertions)]
        debug_assert!(self.sema_ty.kind == AstKind::ArrayTy);
        unsafe { ArrayType::new(self.inner) }
    }

    pub fn struct_ty(&self) -> StructType<'ctx> {
        #[cfg(debug_assertions)]
        std::assert_matches::debug_assert_matches!(
            self.sema_ty.kind,
            AstKind::StructDef
                | AstKind::UnionDef
                | AstKind::EnumDef
                | AstKind::SliceTy
                | AstKind::RangeTy
        );
        unsafe { StructType::new(self.inner) }
    }

    pub fn basic_ty(&self) -> BasicTypeEnum<'ctx> {
        unsafe { BasicTypeEnum::new(self.inner) }
    }

    pub fn basic_metadata_ty(&self) -> BasicMetadataTypeEnum<'ctx> {
        BasicMetadataTypeEnum::try_from(self.any_ty()).u()
    }

    pub fn any_ty(&self) -> AnyTypeEnum<'ctx> {
        unsafe { AnyTypeEnum::new(self.inner) }
    }
}

fn reg<'ctx>(v: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
    Ok(Symbol::Register(CodegenValue::new(v.as_value_ref())))
}

fn reg_sym<'ctx>(v: impl AnyValue<'ctx>) -> Symbol<'ctx> {
    Symbol::Register(CodegenValue::new(v.as_value_ref()))
}

fn stack_val<'ctx>(ptr: PointerValue<'ctx>) -> CodegenResult<Symbol<'ctx>> {
    Ok(Symbol::Stack(ptr))
}

#[derive(Debug, Clone, Copy)]
pub struct Loop<'ctx> {
    continue_bb: BasicBlock<'ctx>,
    end_bb: BasicBlock<'ctx>,
}

#[allow(unused)]
mod return_strategy {
    use super::Codegen;
    use inkwell::{
        attributes::Attribute,
        context::Context,
        types::{BasicType, BasicTypeEnum},
    };

    /// x86_64 clang: for 8 byte >= size
    /// rust 1.81: for 16 byte >= size
    /// ```llvm
    /// %struct.S = type { [6 x i8] }
    ///
    /// define dso_local i48 @test() {
    /// entry:
    ///   %retval = alloca %struct.S, align 1
    ///   %coerce.dive.coerce = alloca i48, align 8
    ///   %x = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   %arrayidx = getelementptr inbounds [6 x i8], ptr %x, i64 0, i64 5
    ///   store i8 69, ptr %arrayidx, align 1
    ///   %coerce.dive = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %coerce.dive.coerce, ptr align 1 %coerce.dive, i64 6, i1 false)
    ///   %0 = load i48, ptr %coerce.dive.coerce, align 8
    ///   ret i48 %0
    /// }
    ///
    /// declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1
    /// ```
    pub fn one_int<'ctx>(c: &mut Codegen<'ctx>, bytes: usize) -> BasicTypeEnum<'ctx> {
        debug_assert!(bytes <= 8);
        c.context.custom_width_int_type(bytes as u32 * 8).as_basic_type_enum()
    }

    /// x86_64 clang: for 16 byte >= size > 8 byte
    /// ```llvm
    /// %struct.S = type { [14 x i8] }
    ///
    /// define dso_local { i64, i48 } @test() {
    /// entry:
    ///   %retval = alloca %struct.S, align 1
    ///   %coerce.dive.coerce = alloca { i64, i48 }, align 8
    ///   %x = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   %arrayidx = getelementptr inbounds [14 x i8], ptr %x, i64 0, i64 5
    ///   store i8 69, ptr %arrayidx, align 1
    ///   %coerce.dive = getelementptr inbounds nuw %struct.S, ptr %retval, i32 0, i32 0
    ///   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %coerce.dive.coerce, ptr align 1 %coerce.dive, i64 14, i1 false)
    ///   %0 = load { i64, i48 }, ptr %coerce.dive.coerce, align 8
    ///   ret { i64, i48 } %0
    /// }
    ///
    /// declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1
    /// ```
    pub fn two_int<'ctx>(c: &mut Codegen<'ctx>, bytes: usize) -> BasicTypeEnum<'ctx> {
        debug_assert!(bytes > 8 && bytes <= 16);
        let ty_i64 = c.context.i64_type().as_basic_type_enum();
        let remaining_bits = (bytes as u32 - 8) * 8;
        let ty2 = c.context.custom_width_int_type(remaining_bits).as_basic_type_enum();
        c.context.struct_type(&[ty_i64, ty2], false).as_basic_type_enum()
    }

    /// x86_64 clang: for size > 16 byte
    /// rust 1.81: for size > 8 byte
    /// ```llvm
    /// %struct.S = type { [17 x i8] }
    ///
    /// define dso_local void @test(ptr dead_on_unwind noalias writable sret(%struct.S) align 1 %agg.result) {
    /// entry:
    ///   %x = getelementptr inbounds nuw %struct.S, ptr %agg.result, i32 0, i32 0
    ///   %arrayidx = getelementptr inbounds [17 x i8], ptr %x, i64 0, i64 5
    ///   store i8 69, ptr %arrayidx, align 1
    ///   ret void
    /// }
    /// ```
    pub fn ret_param() -> ! {
        let context: Context = todo!();
        let type_ref = todo!();
        context.create_type_attribute(Attribute::get_named_enum_kind_id("sret"), type_ref);
        todo!()
    }
}

fn set_alignment(val: impl AsValueRef, alignment: usize) {
    unsafe { InstructionValue::new(val.as_value_ref()) }
        .set_alignment(alignment as u32)
        .u()
}

#[derive(Debug, PartialEq)]
enum RetType<'ctx> {
    Basic(BasicTypeEnum<'ctx>),
    Zst,
    SRetParam,
}

impl<'ctx> RetType<'ctx> {
    fn into_basic(self) -> Option<BasicTypeEnum<'ctx>> {
        match self {
            RetType::Basic(basic_type_enum) => Some(basic_type_enum),
            RetType::Zst | RetType::SRetParam => None,
        }
    }
}

struct ForInfo<'ctx> {
    cond_bb: BasicBlock<'ctx>,
    inc_bb: BasicBlock<'ctx>,
    end_bb: BasicBlock<'ctx>,
    idx: PhiValue<'ctx>,
    idx_ty: IntType<'ctx>,
    idx_int: IntValue<'ctx>,
    outer_loop: Option<Loop<'ctx>>,
}

pub fn finalize_ty(ty: &mut Ptr<ast::Type>, out_ty: Ptr<ast::Type>) {
    debug_assert!(ty_match(*ty, out_ty));
    *ty = out_ty;
}

/// `f: (value: Ptr<Ast>, param_def: Ptr<ast::Decl>, param_idx: usize) -> CodegenResult<()>`
pub fn for_each_call_arg<'ctx>(
    params: Ptr<[Ptr<ast::Decl>]>,
    args: impl IntoIterator<Item = Ptr<Ast>>,
    mut f: impl FnMut(Ptr<Ast>, Ptr<ast::Decl>, usize) -> CodegenResult<()>,
) -> CodegenResult<()> {
    let mut args = args.into_iter().peekable();

    // positional args
    let mut pos_idx = 0;
    while args.peek().is_some_and(is_pos_arg) {
        let pos_arg = args.next().u();
        let param_def = *params.get(pos_idx).u();
        f(pos_arg, param_def, pos_idx)?;
        pos_idx += 1;
    }

    // named args
    let remaining_params = &params[pos_idx..];
    let mut was_set = vec![false; remaining_params.len()];
    for named_arg in args {
        let named_arg = named_arg.downcast::<ast::Assign>();
        let arg_name = named_arg.lhs.downcast::<ast::Ident>();
        let (rem_param_idx, param_def) = remaining_params.find_field(&arg_name.text).u();
        debug_assert!(!was_set[rem_param_idx]);
        was_set[rem_param_idx] = true;
        f(named_arg.rhs, param_def, pos_idx + rem_param_idx)?
    }

    // default args
    for ((rem_idx, missing_param), was_set) in remaining_params.iter().enumerate().zip(was_set) {
        if !was_set {
            f(missing_param.init.u(), *missing_param, pos_idx + rem_idx)?
        }
    }
    Ok(())
}
