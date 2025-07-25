use crate::{
    ast::{
        self, Ast, AstEnum, AstKind, BinOpKind, ConstValEnum, DeclList, DeclListExt, DeclMarkers,
        OptionTypeExt, TypeEnum, UnaryOpKind, UpcastToAst, is_pos_arg,
    },
    context::{ctx, primitives},
    display_code::display,
    literals::replace_escape_chars,
    ptr::Ptr,
    scoped_stack::ScopedStack,
    symbol_table::CodegenSymbolTable,
    type_::{RangeKind, enum_alignment, struct_size, ty_match, union_size},
    util::{
        self, UnwrapDebug, forget_lifetime, is_simple_enum, panic_debug, round_up_to_alignment,
        unreachable_debug,
    },
};
use error::{
    CodegenError,
    CodegenResult::{self, *},
    CodegenResultAndControlFlow,
};
pub use inkwell::targets::TargetMachine;
use inkwell::{
    AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel,
    attributes::{Attribute, AttributeLoc},
    basic_block::BasicBlock,
    builder::Builder,
    context::Context,
    llvm_sys::{LLVMType, LLVMValue, prelude::LLVMValueRef},
    module::{Linkage, Module},
    passes::PassBuilderOptions,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetTriple},
    types::{
        AnyTypeEnum, ArrayType, AsTypeRef, BasicMetadataTypeEnum, BasicType, BasicTypeEnum,
        FloatType, FunctionType, IntType, PointerType, StructType,
    },
    values::{
        AggregateValue, AggregateValueEnum, AnyValue, AnyValueEnum, ArrayValue, AsValueRef,
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, CallSiteValue, FloatValue,
        FunctionValue, GlobalValue, InstructionValue, IntMathValue, IntValue, PhiValue,
        PointerValue, StructValue,
    },
};
use std::{borrow::Cow, collections::HashMap, fmt::Debug, marker::PhantomData, path::Path};

pub mod error;
pub mod jit;

/// Returns [`Symbol::Never`] if it occurs
macro_rules! try_compile_expr_as_val {
    ($codegen:ident, $expr:expr) => {{
        let expr: Ptr<Ast> = $expr;
        let sym = $codegen.compile_expr(expr)?;
        $codegen.sym_as_val(sym, expr.ty.u())?
    }};
}

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

    noundef: Attribute,
    empty_struct_ty: StructType<'ctx>,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Codegen<'ctx> {
        let mut c = Codegen {
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

            noundef: context.create_enum_attribute(Attribute::get_named_enum_kind_id("noundef"), 0),
            empty_struct_ty: context.struct_type(&[], false),
        };

        // needed for string slice literals
        c.llvm_type(primitives().untyped_slice_struct_def.upcast_to_type());

        c
    }

    pub fn compile_top_level(&mut self, stmt: Ptr<Ast>) {
        debug_assert!(stmt.ty.u().matches_void());
        self.compile_expr(stmt).unwrap();
    }

    fn compile_expr(&mut self, expr: Ptr<Ast>) -> CodegenResultAndControlFlow<Symbol<'ctx>> {
        self.compile_expr_with_write_target(expr, None)
    }

    fn compile_expr_with_write_target(
        &mut self,
        expr: Ptr<Ast>,
        mut write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResultAndControlFlow<Symbol<'ctx>> {
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
    ) -> CodegenResultAndControlFlow<Symbol<'ctx>> {
        // println!("compile {:x?}: {:?} {:?}", expr, expr.kind, ast::debug::DebugAst::to_text(&expr));

        debug_assert!(expr.ty.u().is_finalized());

        // Don't use the type of the replacement. because the init type of `_: *u8 = nil;` should
        // be `*u8` not `*never`
        let out_ty = expr.ty.u();
        expr = expr.rep();
        debug_assert!(out_ty.is_finalized());

        let p = primitives();

        macro_rules! write_target_or {
            ($alt:expr) => {
                if let Some(target) = write_target.take() { target } else { $alt }
            };
        }

        macro_rules! compile_initializer {
            ($lhs:expr, $values:expr, $compile_fn:ident) => {{
                let lhs = $lhs;
                let values = $values;
                if let Some(s_def) = out_ty.try_downcast_struct_def() {
                    let s_ty = s_def.upcast_to_type();
                    let struct_ty = self.llvm_type(s_ty).struct_ty();
                    let ptr = write_target_or!(self.build_alloca(struct_ty, "struct", s_ty)?);
                    self.$compile_fn(struct_ty, ptr, s_def.fields, values)?;
                    stack_val(ptr)
                } else if let Some(ptr) = out_ty.try_downcast::<ast::PtrTy>() {
                    let lhs = lhs.u();
                    debug_assert_eq!(lhs.ty, out_ty);
                    let s_def = ptr.pointee.downcast::<ast::StructDef>();
                    let struct_ty = self.type_table[&s_def.upcast_to_type()].struct_ty(); // TODO: test if `*struct {...}` syntax works
                    let ptr = try_compile_expr_as_val!(self, lhs).ptr_val();
                    self.$compile_fn(struct_ty, ptr, s_def.fields, values)?;
                    reg(ptr)
                } else {
                    unreachable_debug()
                }
            }};
        }

        match expr.matchable().as_mut() {
            AstEnum::Ident { decl, .. } => {
                debug_assert!(
                    !decl.u().is_const,
                    "constants should have been replaced during sema"
                );
                Ok(self.get_symbol(decl.u()))
            },
            AstEnum::Block { stmts, has_trailing_semicolon, .. } => {
                self.open_scope();
                let res: CodegenResultAndControlFlow<Symbol> = try {
                    if let Some(last) = stmts.last_mut() {
                        last.ty = Some(out_ty)
                    }
                    let mut out = Symbol::Void;
                    for s in stmts.iter() {
                        out = self.compile_expr(*s)?;
                    }
                    if !*has_trailing_semicolon {
                        out
                    } else {
                        debug_assert!(out_ty.matches_void());
                        Symbol::Void
                    }
                };
                self.close_scope(res.do_continue())?;
                res
            },
            AstEnum::PositionalInitializer { lhs, args, .. } => {
                compile_initializer!(lhs, args, compile_positional_initializer_body)
            },
            AstEnum::NamedInitializer { lhs, fields: values, .. } => {
                compile_initializer!(lhs, values, compile_named_initializer_body)
            },
            AstEnum::ArrayInitializer { lhs, elements, .. } => {
                if let Some(arr_ty) = out_ty.try_downcast::<ast::ArrayTy>() {
                    let arr_llvm_ty = self.llvm_type(arr_ty.upcast_to_type()).arr_ty();
                    let ptr = write_target_or!(self.build_alloca(
                        arr_llvm_ty,
                        "array",
                        arr_ty.upcast_to_type()
                    )?);
                    self.compile_array_initializer_body(arr_ty, arr_llvm_ty, ptr, elements)?;
                    stack_val(ptr)
                } else if let Some(ptr_ty) = out_ty.try_downcast::<ast::PtrTy>() {
                    let lhs = lhs.u();
                    debug_assert_eq!(lhs.ty, out_ty);
                    let arr_ty = ptr_ty.pointee.downcast::<ast::ArrayTy>();
                    let arr_llvm_ty = self.llvm_type(arr_ty.upcast_to_type()).arr_ty();
                    let ptr = try_compile_expr_as_val!(self, lhs).ptr_val();
                    self.compile_array_initializer_body(arr_ty, arr_llvm_ty, ptr, elements)?;
                    reg(ptr)
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::ArrayInitializerShort { lhs, val, count, .. } => {
                if let Some(arr_ty) = out_ty.try_downcast::<ast::ArrayTy>() {
                    let arr_llvm_ty = self.llvm_type(arr_ty.upcast_to_type()).arr_ty();
                    let ptr = write_target_or!(self.build_alloca(
                        arr_llvm_ty,
                        "array",
                        arr_ty.upcast_to_type()
                    )?);
                    self.compile_array_initializer_short_body(
                        arr_ty,
                        arr_llvm_ty,
                        ptr,
                        *val,
                        *count,
                    )?;
                    stack_val(ptr)
                } else if let Some(ptr_ty) = out_ty.try_downcast::<ast::PtrTy>() {
                    let lhs = lhs.u();
                    debug_assert_eq!(lhs.ty, out_ty);
                    let arr_ty = ptr_ty.pointee.downcast::<ast::ArrayTy>();
                    let arr_llvm_ty = self.llvm_type(arr_ty.upcast_to_type()).arr_ty();
                    let ptr = try_compile_expr_as_val!(self, lhs).ptr_val();
                    self.compile_array_initializer_short_body(
                        arr_ty,
                        arr_llvm_ty,
                        ptr,
                        *val,
                        *count,
                    )?;
                    reg(ptr)
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::Dot { lhs, rhs, .. } => {
                let lhs = lhs.u();
                let lhs_ty = lhs.ty.u();
                if lhs_ty == p.module {
                    let decl = rhs.decl.u();
                    debug_assert!(
                        !decl.is_const,
                        "constants should have been replaced during sema"
                    );
                    Ok(self.get_symbol(decl))
                } else if let Some(enum_def) = lhs.try_downcast::<ast::EnumDef>() {
                    debug_assert_eq!(out_ty.kind, AstKind::EnumDef);
                    debug_assert_eq!(lhs_ty, p.type_ty);
                    let _ = self.llvm_type(enum_def.upcast_to_type());
                    self.compile_enum_val(enum_def, &rhs.text, None, write_target.take())
                } else if let Some(s_def) = lhs_ty.try_downcast_struct_def() {
                    let struct_ty = self.llvm_type(s_def.upcast_to_type()).struct_ty();
                    let struct_sym = self.compile_expr(lhs)?;
                    let (field_idx, _) = s_def.fields.find_field(&rhs.text).u();
                    self.build_struct_access(struct_ty, struct_sym, field_idx as u32, &rhs.text)
                        .coerce()
                } else if let Some(u_def) = lhs_ty.try_downcast::<ast::UnionDef>() {
                    let union_ty = self.type_table[&lhs_ty].struct_ty();
                    let union_sym = self.compile_expr(lhs)?;
                    debug_assert!(!(u_def.fields.len() > 0 && union_ty.count_fields() == 0));
                    self.build_struct_access(union_ty, union_sym, 0, &rhs.text).coerce()
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::Index { lhs, idx, .. } => {
                debug_assert!(idx.ty.u().is_finalized());
                let elem_ty_out = match idx.ty.matchable().as_ref() {
                    TypeEnum::IntTy { .. } => out_ty,
                    TypeEnum::RangeTy { .. } => out_ty.get_arr_elem_ty(),
                    _ => unreachable_debug(),
                };

                let elem_ty = finalize_ty(lhs.ty.u().get_arr_elem_ty_mut(), elem_ty_out);
                let lhs_sym = self.compile_expr(*lhs)?;

                let (ptr, len) = match lhs.ty.matchable().as_mut() {
                    TypeEnum::SliceTy { .. } => self.build_slice_field_access(lhs_sym)?,
                    TypeEnum::ArrayTy { len, .. } => {
                        let arr_ptr = self.sym_as_arr_ptr_val(lhs_sym);
                        let len = self.context.i64_type().const_int(len.int(), false);
                        (arr_ptr, len)
                    },
                    _ => unreachable_debug(),
                };
                let llvm_elem_ty = self.llvm_type(elem_ty).basic_ty();

                let idx_val = try_compile_expr_as_val!(self, *idx);
                match idx.ty.matchable().as_ref() {
                    TypeEnum::IntTy { .. } => {
                        stack_val(self.build_gep(llvm_elem_ty, ptr, &[idx_val.int_val()])?)
                    },
                    TypeEnum::RangeTy { elem_ty, rkind, .. } => {
                        let i = elem_ty.try_downcast::<ast::IntTy>();
                        let range_val = idx_val.struct_val();

                        let (ptr, len) = match rkind {
                            RangeKind::Full => (ptr, len),
                            RangeKind::From => {
                                debug_assert!(i.is_some());
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
                                        end.get_type().const_int(1, i.u().is_signed),
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
                                        end.get_type().const_int(1, i.u().is_signed),
                                        "",
                                    )?;
                                }
                                let ptr = self.build_gep(llvm_elem_ty, ptr, &[start])?;
                                let len = self.builder.build_int_sub(end, start, "")?;
                                (ptr, len)
                            },
                        };
                        reg(self.build_slice(ptr, len)?)
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
                if let Some(f) = func.ty.u().try_downcast::<ast::Fn>() {
                    let sym = self.compile_expr(*func)?;
                    let fn_val = match sym {
                        Symbol::Function(val) => CallFnVal::Direct(val),
                        Symbol::Stack(_) | Symbol::Register(_) => {
                            let fn_ptr = self.sym_as_val(sym, p.any_ptr_ty)?.ptr_val();
                            CallFnVal::FnPtr(fn_ptr, self.fn_type(f).0)
                        },
                        _ => unreachable_debug(),
                    };
                    self.compile_call(f, fn_val, args.iter().copied(), write_target.take())
                } else if func.ty == p.method_stub {
                    let dot = func.downcast::<ast::Dot>();
                    let f = dot.rhs.rep().downcast::<ast::Fn>();
                    let val = *self.fn_table.get(&f).u();
                    let args = std::iter::once(dot.lhs.u()).chain(args.iter().copied());
                    self.compile_call(f, CallFnVal::Direct(val), args, write_target.take())
                } else if func.ty == p.enum_variant {
                    let dot = func.downcast::<ast::Dot>();
                    let enum_ty = out_ty.downcast::<ast::EnumDef>();
                    self.llvm_type(enum_ty.upcast_to_type()); // TODO: find a better way to compile anonymous inline types and functions.
                    debug_assert_eq!(dot.lhs.u().downcast_type(), enum_ty.upcast_to_type());
                    debug_assert!(args.len() <= 1);
                    let data = args.get(0).copied();
                    self.compile_enum_val(enum_ty, &dot.rhs.text, data, write_target.take())
                } else {
                    unreachable_debug()
                }
            },
            AstEnum::UnaryOp { op, operand, .. } => {
                op.finalize_arg_type(operand.ty.as_mut().u(), out_ty);
                let sym = self.compile_expr(*operand)?;
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
                let lhs_sym = self.compile_expr(*lhs)?;
                if matches!(op, BinOpKind::And | BinOpKind::Or) {
                    debug_assert_eq!(arg_ty, p.bool);
                    let lhs = self.sym_as_val(lhs_sym, arg_ty)?.bool_val();
                    return reg(self.build_bool_short_circuit_binop(lhs, *rhs, op)?);
                }

                let rhs_sym = self.compile_expr(*rhs)?;
                if let Some(e) = arg_ty.try_downcast::<ast::EnumDef>()
                    && e.is_simple_enum
                    && matches!(op, BinOpKind::Eq | BinOpKind::Ne)
                {
                    debug_assert!(is_simple_enum(e.variants));
                    let tag_ty = match self.enum_tag_type(e) {
                        EnumTagType::Zero => return self.build_unreachable(),
                        EnumTagType::One { .. } => return reg(self.bool_val(true)),
                        EnumTagType::IntTy(int_type) => int_type,
                    };
                    let tag_ty = tag_ty.as_basic_type_enum();
                    let tag_align = enum_alignment(e.variants);
                    let lhs = self.sym_as_val_with_llvm_ty(lhs_sym, tag_ty, tag_align)?.int_val();
                    let rhs = self.sym_as_val_with_llvm_ty(rhs_sym, tag_ty, tag_align)?.int_val();
                    return reg(self.build_int_binop(lhs, rhs, false, op)?);
                }

                let lhs_val = self.sym_as_val(lhs_sym, arg_ty)?;
                let rhs_val = self.sym_as_val(rhs_sym, arg_ty)?;
                if arg_ty == p.bool {
                    let val = self.build_bool_binop(lhs_val.bool_val(), rhs_val.bool_val(), op)?;
                    return reg(val);
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
                .coerce()
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
                debug_assert!(ty_match(lhs.ty.u(), rhs.ty.u()));
                let lhs_sym = self.compile_expr(lhs)?;
                let stack_ptr = self.build_ptr_to_sym(lhs_sym, lhs.ty.u())?;
                self.compile_expr_with_write_target(rhs, Some(stack_ptr))?;
                Ok(Symbol::Void)
            },
            &mut AstEnum::BinOpAssign { lhs, op, rhs, .. } => {
                debug_assert_eq!(lhs.ty, rhs.ty);
                let arg_ty = lhs.ty.u();
                let BasicSymbol::Ref(lhs_ptr) = self.compile_expr(lhs)?.basic() else {
                    unreachable_debug()
                };
                let lhs_llvm_ty = self.llvm_type(arg_ty).basic_ty();
                let lhs_val = self.build_load(lhs_llvm_ty, lhs_ptr, "lhs", arg_ty.alignment())?;

                let binop_res = if matches!(op, BinOpKind::And | BinOpKind::Or) {
                    debug_assert_eq!(arg_ty, p.bool);
                    self.build_bool_short_circuit_binop(lhs_val.into_int_value(), rhs, op)?
                } else {
                    debug_assert_eq!(rhs.ty, arg_ty);
                    let rhs_val = try_compile_expr_as_val!(self, rhs);
                    match arg_ty.matchable().as_ref() {
                        TypeEnum::IntTy { is_signed, .. } => self.build_int_binop(
                            lhs_val.into_int_value(),
                            rhs_val.int_val(),
                            *is_signed,
                            op,
                        )?,
                        TypeEnum::FloatTy { .. } => self.build_float_binop(
                            lhs_val.into_float_value(),
                            rhs_val.float_val(),
                            op,
                        )?,
                        t => todo!("{:?}", t),
                    }
                };
                self.build_store(lhs_ptr, binop_res.basic_val(), arg_ty.alignment())?;
                Ok(Symbol::Void)
            },
            AstEnum::Decl {
                markers, ident, on_type, var_ty, init, is_const, is_extern, ..
            } => {
                let decl = expr.downcast::<ast::Decl>();
                let var_ty = var_ty.u();
                debug_assert!(var_ty.is_finalized());
                debug_assert!(init.is_none_or(|init| init.ty == var_ty));

                if *is_const {
                    if let Some(f) = var_ty.try_downcast::<ast::Fn>() {
                        debug_assert_eq!(f.upcast(), init.u().rep());
                        if init.u().kind != AstKind::Fn {
                            // don't need to compile an alias again
                            debug_assert!(self.fn_table.contains_key(&f));
                        } else if *is_extern {
                            self.compile_prototype(f, FnKind::FnDef(decl));
                        } else {
                            self.compile_fn(f, FnKind::FnDef(decl))?;
                        }
                    } else if var_ty == p.type_ty {
                        // Ensure that the type is in `type_table`
                        self.llvm_type(init.u().downcast_type());
                    }

                    // compile time values are inlined during sema. We don't have to add those to
                    // the symbol table.
                } else {
                    debug_assert_ne!(var_ty, p.type_ty);
                    debug_assert_ne!(expr.kind, AstKind::Fn);
                    debug_assert_ne!(expr.rep().kind, AstKind::Fn);
                    debug_assert_ne!(var_ty.kind, AstKind::Fn);
                    debug_assert!(on_type.is_none());

                    const ENABLE_NON_MUT_TO_REG: bool = false;

                    let is_static = markers.get(DeclMarkers::IS_STATIC_MASK);
                    let sym = if *is_extern || is_static {
                        let ty = self.llvm_type(var_ty).basic_ty();
                        let name = self.mangle_symbol(decl);
                        let global = self.module().add_global(ty, None, name.as_ref());
                        if is_static {
                            global.set_initializer(&match init
                                .and_then(|i| i.try_downcast_const_val())
                            {
                                Some(cv) => self.compile_const_val(cv, var_ty)?.basic_val(),
                                None => ty.const_zero(),
                            });
                            if self.cur_fn.is_some() {
                                // C does this for all `static`s
                                global.set_linkage(Linkage::Internal);
                            }
                            global.set_constant(
                                ctx().do_mut_checks && !markers.get(DeclMarkers::IS_MUT_MASK),
                            );
                        }
                        global.set_alignment(var_ty.alignment() as u32);
                        Symbol::Global(global)
                    } else if ENABLE_NON_MUT_TO_REG
                        && let Some(init) = init
                        && !markers.get(DeclMarkers::IS_MUT_MASK)
                    {
                        self.compile_expr(*init)?
                    } else {
                        let stack_ptr = self.build_alloca2(var_ty, &ident.text)?;
                        if let Some(init) = init {
                            finalize_ty(init.ty.as_mut().u(), var_ty);
                            let _init =
                                self.compile_expr_with_write_target(*init, Some(stack_ptr))?;
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
                let then_sym = self
                    .compile_expr_with_write_target(*then_body, write_target)
                    .handle_unreachable()?;
                let then_val = self.sym_as_val_checked(then_sym, out_ty)?;
                if then_sym.is_some() {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                then_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(else_bb);
                let else_sym = if let Some(else_body) = else_body {
                    else_body.ty = Some(out_ty);
                    self.compile_expr_with_write_target(*else_body, write_target)
                        .handle_unreachable()?
                } else {
                    Some(Symbol::Void)
                };
                let else_val = self.sym_as_val_checked(else_sym, out_ty)?;
                if else_sym.is_some() {
                    self.builder.build_unconditional_branch(merge_bb)?;
                }
                else_bb = self.builder.get_insert_block().expect("has block");

                self.builder.position_at_end(merge_bb);

                if out_ty == p.void_ty {
                    return Ok(Symbol::Void);
                } else if out_ty == p.never {
                    return self.build_unreachable();
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
                let source_sym = self.compile_expr(*source)?;

                match source.ty.matchable().as_ref() {
                    TypeEnum::ArrayTy { len, .. } => {
                        let idx_ty = self.context.i64_type();
                        let len = idx_ty.const_int(len.int(), false);
                        let for_info =
                            self.build_for(idx_ty, false, idx_ty.const_zero(), len, false)?;

                        let arr_ptr = self.sym_as_arr_ptr_val(source_sym);
                        let iter_var_sym =
                            Symbol::Stack(self.build_gep(source_ty.arr_ty(), arr_ptr, &[
                                idx_ty.const_zero(),
                                for_info.idx_int,
                            ])?);
                        self.symbols.push((*iter_var, iter_var_sym));
                        let out = self.compile_expr(*body).handle_unreachable()?;
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
                        self.symbols.push((*iter_var, iter_var_sym));
                        let out = self.compile_expr(*body).handle_unreachable()?;
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
                        self.symbols.push((*iter_var, iter_var_sym));
                        let out = self.compile_expr(*body).handle_unreachable()?;
                        self.build_for_end(for_info, out)?
                    },
                    _ => panic_debug!("for loop over other types"),
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
                let out = self.compile_expr(*body).handle_unreachable()?;
                self.cur_loop = outer_loop;
                if out.is_some() {
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
                    if self.compile_multiple_defer_scopes(self.return_depth).as_do_continue()? {
                        self.build_return(sym, val.ty.u())?;
                    }
                } else {
                    debug_assert_eq!(f.ret_ty, p.void_ty);
                    if self.compile_multiple_defer_scopes(self.return_depth).as_do_continue()? {
                        self.builder.build_return(None)?;
                    }
                }
                Unreachable(())
            },
            AstEnum::Break { val, .. } => {
                if val.is_some() {
                    todo!("break with expr")
                }
                self.compile_multiple_defer_scopes(self.continue_break_depth)?;
                let bb = self.cur_loop.u().end_bb;
                self.builder.build_unconditional_branch(bb)?;
                Unreachable(())
            },
            AstEnum::Continue { .. } => {
                self.compile_multiple_defer_scopes(self.continue_break_depth)?;
                let bb = self.cur_loop.u().continue_bb;
                self.builder.build_unconditional_branch(bb)?;
                Unreachable(())
            },
            AstEnum::Empty { .. } => Ok(Symbol::Void),

            AstEnum::IntVal { .. }
            | AstEnum::FloatVal { .. }
            | AstEnum::BoolVal { .. }
            | AstEnum::CharVal { .. }
            | AstEnum::StrVal { .. }
            | AstEnum::PtrVal { .. }
            | AstEnum::AggregateVal { .. } => {
                reg(self.compile_const_val(expr.downcast_const_val(), out_ty)?)
            },

            AstEnum::ImportDirective { .. }
            | AstEnum::ProgramMainDirective { .. }
            | AstEnum::SimpleDirective { .. } => {
                //panic_debug!("directives should have been resolved during sema")
                Ok(Symbol::Void)
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
                let f = expr.downcast::<ast::Fn>();
                Ok(Symbol::Function(match self.fn_table.get(&f) {
                    Some(val) => *val,
                    None => self.compile_fn(f, FnKind::Lambda)?,
                }))
            },
        }
    }

    fn compile_const_val(
        &mut self,
        cv: Ptr<ast::ConstVal>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<CodegenValue<'ctx>> {
        debug_assert!(ty.is_finalized());
        use codegen_val as ret;
        let p = primitives();
        match cv.matchable().as_ref() {
            //ConstValEnum::IntVal { val, .. } => match ty.finalize().matchable().as_ref() {
            ConstValEnum::IntVal { val, .. } => match ty.matchable().as_ref() {
                TypeEnum::IntTy { bits, is_signed, .. } => {
                    ret(self.int_type(*bits).const_int(*val as u64, *is_signed))
                },
                TypeEnum::FloatTy { bits, .. } => {
                    let float = *val as f64;
                    if float as i64 != *val {
                        panic!("literal precision loss")
                    }
                    ret(self.float_type(*bits).const_float(float))
                },
                _ => unreachable_debug(),
            },
            ConstValEnum::FloatVal { val, .. } => {
                let float_ty = ty.downcast::<ast::FloatTy>();
                ret(self.float_type(float_ty.bits).const_float(*val))
            },
            ConstValEnum::BoolVal { val, .. } => {
                debug_assert!(ty == p.bool);
                ret(self.bool_val(*val))
            },
            ConstValEnum::CharVal { val, .. } => {
                //debug_assert!(expr.ty == p.char);
                debug_assert!(ty == p.u8);
                ret(self.int_type(8).const_int(*val as u8 as u64, false)) // TODO: real char type
            },
            // ConstValEnum::BCharLit { val, .. } => ret(self.int_type(8).const_int(*val as u64, false)),
            ConstValEnum::StrVal { text, .. } => {
                debug_assert!(ty.matches_str());
                let value = replace_escape_chars(&text);
                let ptr = self.builder.build_global_string_ptr(&value, "")?;
                let len = self.int_type(64).const_int(value.len() as u64, false);
                ret(self.build_slice(ptr.as_pointer_value(), len)?)
            },
            ConstValEnum::PtrVal { val, .. } => {
                debug_assert_eq!(ty.kind, AstKind::PtrTy);
                if *val == 0 {
                    ret(self.ptr_type().const_null())
                } else {
                    todo!("other const ptrs")
                }
            },
            ConstValEnum::AggregateVal { elements, .. } => {
                match ty.matchable().as_ref() {
                    TypeEnum::SimpleTy { .. } => todo!(),
                    TypeEnum::SliceTy { .. } => todo!(),
                    TypeEnum::ArrayTy { elem_ty, len, .. } => {
                        let elem_ty = elem_ty.downcast_type();
                        let elem_llvm_ty = self.llvm_type(elem_ty).basic_ty();
                        debug_assert_eq!(elements.len(), len.rep().int());
                        let mut values = Vec::with_capacity(elements.len());
                        for e in elements.iter() {
                            values.push(self.compile_const_val(*e, elem_ty)?.as_value_ref());
                        }
                        // `ArrayType::const_array` has a wrong parameter type.
                        ret(unsafe {
                            ArrayValue::new_raw_const_array(elem_llvm_ty.as_type_ref(), &values)
                        })
                    },
                    TypeEnum::StructDef { fields, .. } => {
                        let struct_ty = self.llvm_type(ty).struct_ty();
                        let mut values = Vec::with_capacity(elements.len());
                        debug_assert_eq!(fields.len(), elements.len());
                        for (e, f) in elements.iter().zip(fields.iter()) {
                            values.push(self.compile_const_val(*e, f.var_ty.u())?.as_value_ref());
                        }
                        ret(new_raw_const_struct(struct_ty, &mut values))
                    },
                    TypeEnum::UnionDef { .. } => todo!(),
                    TypeEnum::EnumDef { .. } => todo!(),
                    TypeEnum::RangeTy { .. } => todo!(),
                    TypeEnum::OptionTy { .. } => todo!(),
                    _ => unreachable_debug(),
                }
            },
            _ => todo!(),
        }
    }

    fn compile_call(
        &mut self,
        f: Ptr<ast::Fn>,
        fn_val: CallFnVal<'ctx>,
        args: impl IntoIterator<Item = Ptr<Ast>>,
        mut write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResultAndControlFlow<Symbol<'ctx>> {
        let ret_ty = f.ret_ty.u();
        let ret_ffi_ty = self.c_ffi_type(ret_ty);
        let use_sret = ret_ffi_ty.do_use_sret();
        let sret_offset = use_sret as u32;

        let ret_val_ptr = if let Some(write_target) = write_target.take() {
            Some(write_target)
        } else if use_sret || ret_ty.is_aggregate() {
            Some(self.build_alloca2(ret_ty, "call")?)
        } else {
            None
        };

        let mut arg_ffi_types = Vec::with_capacity(f.params.len());
        let mut arg_ffi_offsets = Vec::with_capacity(f.params.len());
        {
            let mut cur_arg_offset = sret_offset;
            for p in f.params.iter() {
                let c_ffi_ty = self.c_ffi_type(p.var_ty.u());
                arg_ffi_types.push(c_ffi_ty);
                arg_ffi_offsets.push(cur_arg_offset);
                cur_arg_offset += c_ffi_ty.as_param_count() as u32
            }
        }

        let args_count = match fn_val {
            CallFnVal::Direct(val) => val.count_params(),
            CallFnVal::FnPtr(_, fn_ty) => fn_ty.count_param_types(),
        } as usize;
        let mut arg_values = Vec::with_capacity(args_count);
        unsafe { arg_values.set_len(args_count) };
        #[cfg(debug_assertions)]
        let mut arg_values_was_initialized = vec![false; arg_values.len()];

        macro_rules! set_arg_val {
            ($idx:expr, $val:expr) => {{
                arg_values[$idx as usize] = $val;
                #[cfg(debug_assertions)]
                {
                    arg_values_was_initialized[$idx as usize] = true;
                }
            }};
        }

        if use_sret {
            set_arg_val!(0, ret_val_ptr.u().into());
        }

        for_each_call_arg(f.params, args, |arg, param, p_idx| {
            let sym = self.compile_expr(arg)?;
            let arg_ty = arg_ffi_types.get(p_idx).u();
            let arg_offset = *arg_ffi_offsets.get(p_idx).u();

            match arg_ty {
                CFfiType::Zst => {},
                CFfiType::Simple(simple_ty) => {
                    let val = self.sym_as_val_with_llvm_ty(
                        sym,
                        *simple_ty,
                        param.var_ty.u().alignment(),
                    )?;
                    set_arg_val!(arg_offset, val.basic_metadata_val());
                },
                CFfiType::Simple2(two_params) => {
                    let ffi_struct = self.struct_type_inner(two_params, None, false);
                    let field_align = param.var_ty.u().alignment();
                    for (f_idx, f_ty) in two_params.iter().enumerate() {
                        let f_idx = f_idx as u32;
                        let f_sym = self.build_struct_access(ffi_struct, sym, f_idx, "")?;
                        let f_val = self.sym_as_val_with_llvm_ty(f_sym, *f_ty, field_align)?;
                        set_arg_val!(arg_offset + f_idx, f_val.basic_metadata_val());
                    }
                },
                CFfiType::ByValPtr | CFfiType::Array => {
                    set_arg_val!(arg_offset, self.build_ptr_to_sym(sym, param.var_ty.u())?.into());
                },
                CFfiType::Fn => set_arg_val!(arg_offset, sym.global().as_pointer_value().into()),
                CFfiType::SmallSimpleEnum { small_int, ffi_int } => {
                    let val = self.sym_as_val_with_llvm_ty(
                        sym,
                        small_int.as_basic_type_enum(),
                        param.var_ty.u().alignment(),
                    )?;
                    set_arg_val!(
                        arg_offset,
                        self.builder.build_int_z_extend(val.int_val(), *ffi_int, "ret")?.into()
                    );
                },
            }

            Ok(())
        })?;
        debug_assert_eq!(arg_values.len(), args_count);
        #[cfg(debug_assertions)]
        debug_assert!(arg_values_was_initialized.iter().all(|b| *b));

        let ret = match fn_val {
            CallFnVal::Direct(val) => self.builder.build_direct_call(val, &arg_values, "call")?,
            CallFnVal::FnPtr(ptr, fn_ty) => {
                self.builder.build_indirect_call(fn_ty, ptr, &arg_values, "call")?
            },
        };

        ret.add_ret_attributes(self, ret_ffi_ty, f);
        let mut cur_ffi_arg_idx = sret_offset;
        for (param, ffi_ty) in f.params.iter().zip(arg_ffi_types) {
            cur_ffi_arg_idx += ret.add_param_attributes(self, *param, ffi_ty, cur_ffi_arg_idx);
        }

        let ret = if let CFfiType::SmallSimpleEnum { ffi_int, small_int } = ret_ffi_ty {
            debug_assert!(!use_sret);
            let ret = CodegenValue::new(ret.as_value_ref()).int_val();
            debug_assert_eq!(ret.get_type(), ffi_int);
            self.builder.build_int_truncate(ret, small_int, "call")?.as_any_value_enum()
        } else {
            ret.as_any_value_enum()
        };

        let p = primitives();
        if ret_ty == p.never {
            return self.build_unreachable();
        } else if ret_ty == p.void_ty {
            Ok(Symbol::Void)
        } else if let Some(ret_val_ptr) = ret_val_ptr {
            if !use_sret {
                let ret = CodegenValue::new(ret.as_value_ref()).basic_val();
                self.build_store(ret_val_ptr, ret, ret_ty.alignment())?;
            }
            stack_val(ret_val_ptr)
        } else {
            debug_assert!(!use_sret);
            reg(ret)
        }
    }

    fn compile_enum_val(
        &mut self,
        enum_def: Ptr<ast::EnumDef>,
        variant_name: &str,
        data: Option<Ptr<Ast>>,
        write_target: Option<PointerValue<'ctx>>,
    ) -> CodegenResultAndControlFlow<Symbol<'ctx>> {
        let variant_idx = enum_def.variants.find_field(variant_name).u().0;
        let variant_tag = *enum_def.variant_tags.u().get(variant_idx).u();

        let enum_ty = self.type_table[&enum_def.upcast_to_type()].struct_ty();
        let tag_ty = match self.enum_tag_type(enum_def) {
            EnumTagType::Zero => panic_debug!("Enum has no variants"),
            EnumTagType::One { .. } => None,
            EnumTagType::IntTy(int_type) => Some(int_type),
        };
        let tag_val =
            tag_ty.map(|tag_ty| tag_ty.const_int(variant_tag as u64, variant_tag.is_negative()));
        if write_target.is_some() || data.is_some() {
            let enum_ptr = if let Some(ptr) = write_target {
                ptr
            } else {
                self.build_alloca(enum_ty, "enum", enum_def.upcast_to_type())?
            };

            // set tag
            if let Some(tag) = tag_val {
                let tag_ptr = enum_ptr;
                self.build_store(tag_ptr, tag, enum_alignment(enum_def.variants))?;
            }

            // set data
            if let Some(data) = data {
                let data_idx = tag_val.is_some() as u32;
                let data_ptr =
                    self.builder.build_struct_gep(enum_ty, enum_ptr, data_idx, "enum_data")?;
                debug_assert_eq!(data.ty, enum_def.variants[variant_idx].var_ty.u());
                self.compile_expr_with_write_target(data, Some(data_ptr))?;
            }

            stack_val(enum_ptr)
        } else {
            Ok(tag_val.map(reg_sym).unwrap_or(Symbol::Void))
        }
    }

    fn compile_fn(&mut self, f: Ptr<ast::Fn>, def: FnKind) -> CodegenResult<FunctionValue<'ctx>> {
        debug_assert!(def.is_lamda_or(|d| d.init.is_some_and(|i| i.rep() == f.upcast())));
        let prev_bb = self.builder.get_insert_block();

        let (fn_val, use_sret) = self.compile_prototype(f, def);

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

    fn fn_type(&mut self, f: Ptr<ast::Fn>) -> (FunctionType<'ctx>, CFfiType<'ctx>) {
        let ptr_type = self.ptr_type().into();
        let ret_type = self.c_ffi_type(f.ret_ty.u());
        let use_sret = ret_type.do_use_sret();

        // capacity: A struct param might be flattened into at most two params
        let mut param_types = Vec::with_capacity(f.params.len() * 2 + use_sret as usize);
        if use_sret {
            param_types.push(ptr_type);
        }

        for p in f.params.iter() {
            let var_ty = p.var_ty.u();
            match self.c_ffi_type(var_ty) {
                CFfiType::Zst => {},
                CFfiType::Simple(simple) => param_types.push(simple.into()),
                CFfiType::Simple2(two_params) => {
                    for p in two_params {
                        param_types.push(p.into());
                    }
                },
                CFfiType::ByValPtr | CFfiType::Fn | CFfiType::Array => param_types.push(ptr_type),
                CFfiType::SmallSimpleEnum { ffi_int, .. } => param_types.push(ffi_int.into()),
            }
        }

        let fn_ty = match ret_type.into_basic_ret_ty(self) {
            Some(ret_type) => ret_type.fn_type(&param_types, false),
            None => self.context.void_type().fn_type(&param_types, false),
        };
        (fn_ty, ret_type)
    }

    fn compile_prototype(&mut self, f: Ptr<ast::Fn>, def: FnKind) -> (FunctionValue<'ctx>, bool) {
        let name = match def {
            FnKind::FnDef(decl) => self.mangle_symbol(decl),
            FnKind::Lambda => "lambda".into(), // TODO: also mangle lambda (e.g. `my_fn.lambda`)
        };
        debug_assert!(
            self.fn_table.get(&f).is_none(),
            "called compile_prototype multiple times on the same function ('{name}')"
        );
        let (fn_type, ret_ffi_ty) = self.fn_type(f);
        let use_sret = ret_ffi_ty.do_use_sret();

        let fn_val = self.module().add_function(name.as_ref(), fn_type, Some(Linkage::External));
        let mut params_iter = fn_val.get_param_iter();

        fn_val.add_ret_attributes(self, ret_ffi_ty, f);
        if use_sret {
            params_iter.next().u().set_name("sret");
        }

        let mut cur_ffi_arg_idx = use_sret as u32;
        for p in f.params.iter() {
            let ffi_ty = self.c_ffi_type(p.var_ty.u());
            cur_ffi_arg_idx += fn_val.add_param_attributes(self, *p, ffi_ty, cur_ffi_arg_idx);

            #[cfg(debug_assertions)]
            match ffi_ty {
                CFfiType::Zst => {},
                CFfiType::Simple(_)
                | CFfiType::ByValPtr
                | CFfiType::Fn
                | CFfiType::Array
                | CFfiType::SmallSimpleEnum { .. } => {
                    params_iter.next().u().set_name(p.ident.text.as_ref());
                },
                CFfiType::Simple2(_) => {
                    for f_idx in 0..2 {
                        params_iter
                            .next()
                            .u()
                            .set_name(&format!("{}.{f_idx}", p.ident.text.as_ref()));
                    }
                },
            }
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

            let mut param_val_iter = func.get_param_iter().skip(use_sret as usize);
            for param_def in f.params.iter() {
                let param_ty = param_def.var_ty.u();
                let s = match self.c_ffi_type(param_ty) {
                    CFfiType::Zst => {
                        debug_assert_ne!(param_def.var_ty, primitives().never);
                        continue;
                    },
                    CFfiType::Simple(simple_ty) => {
                        let param = param_val_iter.next().u();
                        debug_assert_eq!(param.get_type(), simple_ty);
                        let param = CodegenValue::new(param.as_value_ref());
                        if param_def.markers.get(DeclMarkers::IS_MUT_MASK)
                            || param_ty.is_aggregate()
                        {
                            self.position_builder_at_start(func.get_first_basic_block().u());
                            Symbol::Stack(self.build_alloca_value(param.basic_val(), param_ty)?)
                        } else {
                            debug_assert_eq!(
                                param.basic_val().get_type(),
                                self.llvm_type(param_ty).basic_ty()
                            );
                            Symbol::Register(param)
                        }
                    },
                    CFfiType::Simple2(two_params) => {
                        let ffi_struct = self.struct_type_inner(&two_params, None, false);
                        let param_align = param_ty.alignment();
                        let ptr = self.build_alloca2(param_ty, param_def.ident.text.as_ref())?;
                        for f_idx in 0..2 {
                            let f_ptr =
                                self.builder.build_struct_gep(ffi_struct, ptr, f_idx, "")?;
                            self.build_store(f_ptr, param_val_iter.next().u(), param_align)?;
                        }
                        Symbol::Stack(ptr)
                    },
                    CFfiType::ByValPtr => {
                        Symbol::Stack(param_val_iter.next().u().into_pointer_value())
                    },
                    CFfiType::Fn | CFfiType::Array => {
                        reg_sym(param_val_iter.next().u().into_pointer_value())
                    },
                    CFfiType::SmallSimpleEnum { ffi_int, small_int } => {
                        let param = param_val_iter.next().u().into_int_value();
                        debug_assert_eq!(param.get_type(), ffi_int);
                        reg_sym(self.builder.build_int_truncate(
                            param,
                            small_int,
                            &param_def.ident.text,
                        )?)
                    },
                };
                self.symbols.push((*param_def, s));
            }

            f.body.u().as_mut().ty = Some(f.ret_ty.u());
            if let Some(body) = self.compile_expr(f.body.u()).handle_unreachable()? {
                self.build_return(body, f.ret_ty.u())?;
            }

            if func.verify(true) {
                func
            } else {
                #[cfg(debug_assertions)]
                if crate::context::ctx().debug_llvm_module_on_invalid_fn() {
                    self.module().print_to_stderr();
                }
                unsafe { func.delete() };
                //panic_debug!("invalid generated function");
                Err(CodegenError::InvalidGeneratedFunction.into())?
            }
        };
        self.close_scope(true)?; // TODO: is `true` correct?
        self.return_depth = outer_return_depth;
        self.cur_fn = outer_fn;
        self.builder.clear_insertion_position();
        res
    }

    fn mangle_symbol<'n>(&self, decl: Ptr<ast::Decl>) -> Cow<'n, str> {
        if decl.ident.replacement.is_some() {
            return decl.ident.rep().flat_downcast::<ast::Ident>().text.as_ref().into();
        }
        let name = decl.ident.text.as_ref();
        if decl.is_extern {
            name.into()
        } else if name == "main" {
            "_main".into()
        } else if let Some(ty) = decl.on_type {
            format!("{}.{name}", ty.downcast_type()).into() // TODO: use correct type name
        } else if let Some(f) = self.cur_fn {
            format!("{}.{name}", f.get_name().to_str().unwrap()).into()
        } else {
            name.into()
        }
    }

    fn compile_positional_initializer_body(
        &mut self,
        struct_ty: StructType<'ctx>,
        struct_ptr: PointerValue<'ctx>,
        fields: DeclList,
        args: &[Ptr<Ast>],
    ) -> CodegenResultAndControlFlow<Symbol<'ctx>> {
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
    ) -> CodegenResultAndControlFlow<()> {
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
            self.compile_expr_with_write_target(field.init.u(), Some(field_ptr))?;
        }
        Ok(())
    }

    fn compile_array_initializer_body(
        &mut self,
        arr_ty: Ptr<ast::ArrayTy>,
        arr_llvm_ty: ArrayType<'ctx>,
        arr_ptr: PointerValue<'ctx>,
        elements: &mut [Ptr<Ast>],
    ) -> CodegenResultAndControlFlow<()> {
        let elem_ty = arr_ty.elem_ty.downcast_type();
        debug_assert_eq!(elements.len(), arr_ty.len.int());
        let idx_ty = self.context.i64_type();
        for (idx, elem) in elements.iter_mut().enumerate() {
            finalize_ty(elem.ty.as_mut().u(), elem_ty);
            let idx = idx_ty.const_int(idx as u64, false);
            let elem_ptr = self.build_gep(arr_llvm_ty, arr_ptr, &[idx_ty.const_zero(), idx])?;
            self.compile_expr_with_write_target(*elem, Some(elem_ptr))?;
        }
        Ok(())
    }

    fn compile_array_initializer_short_body(
        &mut self,
        arr_ty: Ptr<ast::ArrayTy>,
        arr_llvm_ty: ArrayType<'ctx>,
        arr_ptr: PointerValue<'ctx>,
        val: Ptr<ast::Ast>,
        count: Ptr<ast::Ast>,
    ) -> CodegenResultAndControlFlow<()> {
        let elem_ty = finalize_ty(val.as_mut().ty.as_mut().u(), arr_ty.elem_ty.downcast_type());
        let elem_val = try_compile_expr_as_val!(self, val);

        let len: u32 = count.int();
        debug_assert_eq!(len, arr_ty.len.int());

        let idx_ty = self.context.i64_type();
        let for_info = self.build_for(
            idx_ty,
            false,
            idx_ty.const_zero(),
            idx_ty.const_int(len as u64, false),
            false,
        )?;

        let elem_ptr =
            self.build_gep(arr_llvm_ty, arr_ptr, &[idx_ty.const_zero(), for_info.idx_int])?;
        self.build_store(elem_ptr, elem_val.basic_val(), elem_ty.alignment())?;

        self.build_for_end(for_info, Some(Symbol::Void))?;
        Ok(())
    }

    fn compile_cast(
        &mut self,
        expr: Ptr<Ast>,
        target_ty: Ptr<ast::Type>,
    ) -> CodegenResultAndControlFlow<Symbol<'ctx>> {
        let p = primitives();
        let sym = self.compile_expr(expr)?;
        let expr_ty = expr.ty.u();

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
            } else if target_ty.kind == AstKind::PtrTy {
                reg(self.builder.build_int_to_ptr(int, self.ptr_type(), "")?)
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
                reg(if target_i_ty.is_signed {
                    self.builder.build_float_to_signed_int(float, int_ty, "")?
                } else {
                    self.builder.build_float_to_unsigned_int(float, int_ty, "")?
                })
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

        if let Some(e) = expr_ty.try_downcast::<ast::EnumDef>()
            && e.is_simple_enum
            && let Some(target_int) = target_ty.try_downcast::<ast::IntTy>()
        {
            debug_assert!(is_simple_enum(e.variants));
            let target_ty = self.llvm_type(target_ty).int_ty();
            let tag_ty = match self.enum_tag_type(e) {
                EnumTagType::Zero => return self.build_unreachable(),
                EnumTagType::One { tag } => {
                    return reg(target_ty.const_int(tag as u64, target_int.is_signed));
                },
                EnumTagType::IntTy(int_type) => int_type.as_basic_type_enum(),
            };
            let is_tag_signed = e.tag_ty.u().is_signed;
            let tag_align = enum_alignment(e.variants);
            let val = self.sym_as_val_with_llvm_ty(sym, tag_ty, tag_align)?.int_val();
            return reg(self.builder.build_int_cast_sign_flag(
                val,
                target_ty,
                is_tag_signed,
                "",
            )?);
        }

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

    fn build_alloca2(
        &mut self,
        ty: Ptr<ast::Type>,
        name: &str,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let llvm_ty = self.llvm_type(ty).basic_ty();
        self.build_alloca(llvm_ty, name, ty)
    }

    fn build_alloca_value(
        &self,
        val: impl BasicValue<'ctx>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<PointerValue<'ctx>> {
        let val = val.as_basic_value_enum();
        let alloca = self.build_alloca(val.get_type(), "", ty)?;
        self.build_store(alloca, val, ty.alignment())?;
        Ok(alloca)
    }

    fn build_ptr_to_sym(
        &self,
        sym: Symbol<'ctx>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<PointerValue<'ctx>> {
        Ok(match sym.basic() {
            BasicSymbol::Ref(ptr_value) => ptr_value,
            BasicSymbol::Val(val) => {
                #[cfg(debug_assertions)]
                println!("INFO: doing stack allocation for register value: {val:?}");
                self.build_alloca_value(val.basic_val(), ty)?
            },
            BasicSymbol::Zst => panic_debug!("{ty} is not represented as a symbol"),
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

    fn build_return(&mut self, ret_sym: Symbol<'ctx>, ret_ty: Ptr<ast::Type>) -> CodegenResult<()> {
        let ret = match (ret_sym.basic(), self.c_ffi_type(ret_ty).flatten_simple2(self)) {
            (_, CFfiType::Zst) => None,
            (BasicSymbol::Zst, _) => unreachable_debug(),
            (BasicSymbol::Val(val), CFfiType::ByValPtr | CFfiType::Array) => {
                let sret_ptr = self.sret_ptr.u();
                self.build_store(sret_ptr, val.basic_val(), ret_ty.alignment())?;
                None
            },
            (BasicSymbol::Ref(ptr), CFfiType::ByValPtr | CFfiType::Array) => {
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
            (BasicSymbol::Val(_), CFfiType::Fn) => unreachable_debug(),
            (BasicSymbol::Ref(ptr), CFfiType::Fn) => Some(ptr.as_basic_value_enum()),
            (BasicSymbol::Val(val), CFfiType::Simple(llvm_ty)) if ret_ty.is_aggregate() => {
                let ret = self.build_alloca(llvm_ty, "ret", ret_ty)?;
                self.build_store(ret, val.basic_val(), ret_ty.alignment())?;
                Some(self.builder.build_load(llvm_ty, ret, "ret")?)
            },
            (BasicSymbol::Val(val), CFfiType::Simple(_)) => Some(val.basic_val()),
            (BasicSymbol::Ref(ptr), CFfiType::Simple(ty)) => {
                Some(self.build_load(ty, ptr, "ret", ret_ty.alignment())?)
            },
            (BasicSymbol::Val(val), CFfiType::SmallSimpleEnum { small_int, ffi_int }) => {
                let enum_ty = ret_ty.downcast::<ast::EnumDef>();
                debug_assert!(enum_ty.is_simple_enum);
                debug_assert!(enum_ty.tag_ty.u().bits < DEFAULT_C_ENUM_BITS);
                let val = val.int_val();
                debug_assert_eq!(val.get_type(), small_int);
                Some(self.builder.build_int_z_extend(val, ffi_int, "ret")?.as_basic_value_enum())
            },
            (BasicSymbol::Ref(ptr), CFfiType::SmallSimpleEnum { ffi_int, .. }) => {
                Some(self.build_load(ffi_int, ptr, "ret", ret_ty.alignment())?)
            },
            (_, CFfiType::Simple2(_)) => unreachable_debug(),
        };
        match ret {
            Some(ret) => self.builder.build_return(Some(&ret))?,
            None => self.builder.build_return(None)?,
        };
        Ok(())
    }

    fn build_unreachable<T>(&self) -> CodegenResultAndControlFlow<T> {
        let _inst: InstructionValue<'ctx> = self.builder.build_unreachable()?;
        Unreachable(())
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
        match struct_sym.basic() {
            BasicSymbol::Ref(ptr) => {
                stack_val(self.builder.build_struct_gep(struct_ty, ptr, idx, name)?)
            },
            BasicSymbol::Val(val) => {
                reg(self.builder.build_extract_value(val.struct_val(), idx, name)?)
            },
            BasicSymbol::Zst => Ok(Symbol::Void),
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
    ) -> CodegenResult<CodegenValue<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
            Ok(CodegenValue::new(val.as_value_ref()))
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
    ) -> CodegenResultAndControlFlow<CodegenValue<'ctx>> {
        fn ret<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResultAndControlFlow<CodegenValue<'ctx>> {
            Ok(CodegenValue::new(val.as_value_ref()))
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
    ) -> CodegenResult<AggregateValueEnum<'ctx>> {
        let slice = self.slice_ty().get_undef();
        let slice = self.builder.build_insert_value(slice, ptr, 0, "")?;
        Ok(self.builder.build_insert_value(slice, len, 1, "slice")?)
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
        body_out_sym: Option<Symbol<'ctx>>,
    ) -> CodegenResult<()> {
        let ForInfo { cond_bb, inc_bb, end_bb, idx, idx_ty, idx_int, outer_loop } = for_info;
        self.cur_loop = outer_loop;
        if body_out_sym.is_some() {
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
        assert!(bits > 0);
        // See <https://llvm.org/docs/Frontend/PerformanceTips.html#avoid-loads-and-stores-of-non-byte-sized-types>
        match bits {
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
            80 => self.context.x86_f80_type(),
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

    fn new_enum_type(
        &mut self,
        enum_def: Ptr<ast::EnumDef>,
        name: Option<Ptr<str>>,
    ) -> BasicTypeEnum<'ctx> {
        debug_assert_eq!(enum_def.is_simple_enum, is_simple_enum(enum_def.variants));
        let tag_ty = self
            .enum_tag_type(enum_def)
            .sized()
            .map(Into::into)
            .unwrap_or(self.empty_struct_ty.as_basic_type_enum());
        if enum_def.is_simple_enum {
            tag_ty
        } else {
            let data_ty = self.new_union_type(enum_def.variants, None).as_basic_type_enum();
            self.struct_type_inner(&[tag_ty.into(), data_ty], name, false).into()
        }
    }

    #[inline]
    fn enum_tag_type(&self, enum_def: Ptr<ast::EnumDef>) -> EnumTagType<'ctx> {
        let tag_bits = enum_def.tag_ty.u().bits;
        debug_assert_eq!(
            tag_bits,
            util::variant_count_to_tag_size_bytes(enum_def.variants.len()) * 8
        );
        match tag_bits {
            // Note: This condition will be too strict when never variants are filtered out.
            0 if enum_def.variants.len() == 0 => EnumTagType::Zero,
            0 => {
                debug_assert!(enum_def.variants.len() == 1);
                let tag = enum_def.variants[0].init.map(Ptr::<Ast>::int).unwrap_or(0);
                EnumTagType::One { tag }
            },
            b => EnumTagType::IntTy(self.int_type(b)),
        }
    }

    #[inline]
    fn range_type(&mut self, range_ty: Ptr<ast::RangeTy>) -> StructType<'ctx> {
        if range_ty.upcast_to_type() == primitives().full_range {
            return self.empty_struct_ty;
        }
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
                panic_debug!("cannot compile type: {ty}");
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
            TypeEnum::SliceTy { .. } => self.llvm_type(p.untyped_slice_struct_def.upcast_to_type()),
            TypeEnum::ArrayTy { len, elem_ty, .. } => {
                t!(self.llvm_type(elem_ty.downcast_type()).basic_ty().array_type(len.int()))
            },
            //TypeEnum::FunctionTy { .. } => todo!(),
            TypeEnum::StructDef { fields, .. } => t!(self.new_struct_type(*fields, None)),
            TypeEnum::UnionDef { fields, .. } => t!(self.new_union_type(*fields, None)),
            TypeEnum::EnumDef { .. } => t!(self.new_enum_type(ty.downcast(), None)),
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
            TypeEnum::Fn { .. } => t!(self.ptr_type()),
            TypeEnum::Unset => unreachable_debug(),
        };

        let old_entry = self.type_table.insert(ty, llvm_ty);
        debug_assert!(old_entry.is_none());

        llvm_ty
    }

    /// The C calling convention is way to complicated thus this is definitely not cross-platform.
    ///
    /// See <https://discourse.llvm.org/t/questions-about-c-calling-conventions/72414>
    /// TODO: See <https://mcyoung.xyz/2024/04/17/calling-convention/>
    fn c_ffi_type(&mut self, ty: Ptr<ast::Type>) -> CFfiType<'ctx> {
        if let Some(struct_ty) = ty.try_downcast_struct_def() {
            return self.c_ffi_struct(struct_ty.fields);
        } else if let Some(enum_def) = ty.try_downcast::<ast::EnumDef>()
            && enum_def.is_simple_enum
        {
            let Some(tag_ty) = self.enum_tag_type(enum_def).sized() else {
                return CFfiType::Zst;
            };
            return if enum_def.tag_ty.u().bits < DEFAULT_C_ENUM_BITS {
                CFfiType::SmallSimpleEnum {
                    small_int: tag_ty,
                    ffi_int: self.int_type(DEFAULT_C_ENUM_BITS),
                }
            } else {
                CFfiType::Simple(tag_ty.into())
            };
        }
        let size = ty.size();
        match ty.matchable().as_ref() {
            _ if size == 0 => CFfiType::Zst,
            TypeEnum::Fn { .. } => CFfiType::Fn,
            TypeEnum::ArrayTy { .. } => CFfiType::Array,
            _ if size > 16 => CFfiType::ByValPtr,
            TypeEnum::UnionDef { .. } | TypeEnum::EnumDef { .. } => CFfiType::Simple(
                self.context
                    .struct_type(&[self.int_type(size as u32 * 8).as_basic_type_enum()], false)
                    .as_basic_type_enum(),
            ),
            TypeEnum::StructDef { .. } => unreachable_debug(),
            _ => CFfiType::Simple(self.llvm_type(ty).basic_ty()),
        }
    }

    fn c_ffi_struct(&mut self, fields: Ptr<[Ptr<ast::Decl>]>) -> CFfiType<'ctx> {
        #[derive(Debug, PartialEq)]
        enum PrevState {
            None,
            Int,
            Float,
            FloatFloat,
        }

        let mut new_fields = [None; 2];
        macro_rules! add_field {
            ($f:expr) => {{
                if new_fields[1].is_some() {
                    return CFfiType::ByValPtr;
                }
                new_fields[new_fields[0].is_some() as usize] = Some($f);
            }};
        }

        let mut prev_state = PrevState::None;
        let mut prev_bytes: u32 = 0;

        debug_assert_eq!(
            std::mem::size_of::<[Option<BasicTypeEnum<'_>>; 2]>(),
            std::mem::size_of::<[BasicTypeEnum<'_>; 2]>()
        );

        macro_rules! push_prev_state_to_new_fields {
            () => {
                match prev_state {
                    PrevState::None => {},
                    PrevState::Int => add_field!(
                        self.context.custom_width_int_type(prev_bytes << 3).as_basic_type_enum()
                    ),
                    PrevState::Float => add_field!(self.context.f32_type().as_basic_type_enum()),
                    PrevState::FloatFloat => {
                        add_field!(self.context.f32_type().vec_type(2).as_basic_type_enum())
                    },
                }
            };
        }

        for f in fields.iter_types() {
            let f_size = f.size() as u32;
            if f_size == 0 {
                continue;
            }

            let f_align = f.alignment() as u32;
            prev_bytes = round_up_to_alignment!(prev_bytes, f_align);
            if prev_bytes >= 8 {
                // finished the first 8 bytes
                push_prev_state_to_new_fields!();
                prev_state = PrevState::None;
                prev_bytes = 0;
            }

            macro_rules! handle_normal_field {
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
                TypeEnum::IntTy { bits, .. } => handle_normal_field!(bits),
                TypeEnum::FloatTy { bits: 64, .. } => {
                    debug_assert_eq!(prev_state, PrevState::None);
                    debug_assert_eq!(prev_bytes, 0);
                    add_field!(self.context.f64_type().as_basic_type_enum());
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
                TypeEnum::PtrTy { .. } | TypeEnum::Fn { .. } => {
                    debug_assert_eq!(prev_state, PrevState::None);
                    debug_assert_eq!(prev_bytes, 0);
                    add_field!(self.ptr_type().into());
                },
                TypeEnum::SliceTy { .. }
                | TypeEnum::ArrayTy { .. }
                | TypeEnum::StructDef { .. }
                | TypeEnum::UnionDef { .. }
                | TypeEnum::EnumDef { .. }
                | TypeEnum::RangeTy { .. }
                | TypeEnum::OptionTy { .. } => {
                    // TODO: flatten struct:
                    // i32, struct {a:f32}, i64 -> i64, i64
                    // struct {a:f32}, i64 -> f32, i64
                    // []any -> ptr, i64

                    let i64_count = f_size >> 3; // f_size / 8
                    let rem_size = f_size & 0b111; // f_size % 8
                    for _ in 0..i64_count {
                        add_field!(self.context.i64_type().as_basic_type_enum())
                    }
                    if rem_size > 0 {
                        handle_normal_field!(rem_size * 8);
                    }
                },
                TypeEnum::Unset => unreachable_debug(),
            }
        }

        // finished the last 8 bytes
        push_prev_state_to_new_fields!();

        debug_assert!(struct_size(fields) <= 16);

        match new_fields {
            [None, _] => CFfiType::Zst,
            [Some(f), None] => CFfiType::Simple(f),
            [Some(a), Some(b)] => CFfiType::Simple2([a, b]),
        }
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
        Ok(match sym.basic() {
            BasicSymbol::Ref(ptr) => {
                CodegenValue::new(self.build_load(llvm_ty, ptr, "", alignment)?.as_value_ref())
            },
            BasicSymbol::Val(val) => {
                debug_assert_eq!(val.basic_val().get_type(), llvm_ty);
                val
            },
            BasicSymbol::Zst => panic_debug!("a zst is an invalid value"),
        })
    }

    fn sym_as_val_checked(
        &mut self,
        sym: Option<Symbol<'ctx>>,
        ty: Ptr<ast::Type>,
    ) -> CodegenResult<Option<CodegenValue<'ctx>>> {
        Ok(Some(match sym {
            Some(Symbol::Stack(ptr)) => {
                let llvm_ty = self.llvm_type(ty).basic_ty();
                CodegenValue::new(self.build_load(llvm_ty, ptr, "", ty.alignment())?.as_value_ref())
            },
            Some(Symbol::Register(val)) => val,
            Some(Symbol::Global(val)) => CodegenValue::new(val.as_value_ref()),
            _ => return Ok(None),
        }))
    }

    fn sym_as_arr_ptr_val(&mut self, sym: Symbol<'ctx>) -> PointerValue<'ctx> {
        match sym {
            Symbol::Stack(ptr) => ptr,
            Symbol::Register(val) => val.ptr_val(),
            Symbol::Global(_global_value) => todo!(),
            _ => unreachable_debug(),
        }
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
        if do_compile_defer {
            self.compile_defer_exprs().handle_unreachable()?;
        }
        self.symbols.close_scope();
        self.defer_stack.close_scope();
        self.return_depth -= 1;
        self.continue_break_depth -= 1;
        Ok(())
    }

    #[inline]
    fn compile_defer_exprs(&mut self) -> CodegenResultAndControlFlow<()> {
        let exprs = unsafe { forget_lifetime(self.defer_stack.get_cur_scope()) };
        for expr in exprs.iter().rev() {
            let _sym = self.compile_expr(*expr)?;
        }
        Ok(())
    }

    fn compile_multiple_defer_scopes(&mut self, depth: usize) -> CodegenResultAndControlFlow<()> {
        let defer_stack = unsafe { forget_lifetime(&self.defer_stack) };
        for scope in defer_stack.iter_scopes().take(depth) {
            for expr in scope.iter().rev() {
                let _sym = self.compile_expr(*expr)?;
            }
        }
        Ok(())
    }

    fn position_builder_at_start(&self, entry: BasicBlock<'ctx>) {
        match entry.get_first_instruction() {
            Some(first_instr) => self.builder.position_before(&first_instr),
            None => self.builder.position_at_end(entry),
        }
    }

    #[inline]
    pub fn module(&self) -> &Module<'ctx> {
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
            .map_err(CodegenError::CannotOptimizeModule)?)
    }

    fn compile_to_obj_file(
        &self,
        target_machine: &TargetMachine,
        obj_file_path: &Path,
    ) -> Result<(), CodegenError> {
        std::fs::create_dir_all(obj_file_path.parent().unwrap()).unwrap();
        target_machine
            .write_to_file(&self, inkwell::targets::FileType::Object, obj_file_path)
            .map_err(CodegenError::CannotCompileObjFile)
    }

    fn jit_run_fn<Ret>(&self, fn_name: &str, opt: OptimizationLevel) -> CodegenResult<Ret> {
        let jit = self.create_jit_execution_engine(opt).map_err(CodegenError::CannotCreateJit)?;
        Ok(unsafe { jit.get_function::<unsafe extern "C" fn() -> Ret>(fn_name)?.call() })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Symbol<'ctx> {
    Void,
    Stack(PointerValue<'ctx>),
    //Register(AnyValueEnum<'ctx>),
    Register(CodegenValue<'ctx>),
    Global(GlobalValue<'ctx>),
    Function(FunctionValue<'ctx>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasicSymbol<'ctx> {
    Zst,
    Val(CodegenValue<'ctx>),
    Ref(PointerValue<'ctx>),
}

impl<'ctx> Symbol<'ctx> {
    #[inline]
    fn basic(self) -> BasicSymbol<'ctx> {
        match self {
            Symbol::Void => BasicSymbol::Zst,
            Symbol::Stack(val) => BasicSymbol::Ref(val),
            Symbol::Register(val) => BasicSymbol::Val(val),
            Symbol::Global(val) => BasicSymbol::Ref(val.as_pointer_value()),
            Symbol::Function(val) => BasicSymbol::Ref(val.as_global_value().as_pointer_value()),
        }
    }

    fn global(self) -> GlobalValue<'ctx> {
        match self {
            Symbol::Global(val) => val,
            Symbol::Function(val) => val.as_global_value(),
            _ => unreachable_debug(),
        }
    }
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
        debug_assert!(BasicValueEnum::try_from(self.any_val()).is_ok());
        unsafe { BasicValueEnum::new(self.val) }
    }

    pub fn basic_metadata_val(&self) -> BasicMetadataValueEnum<'ctx> {
        BasicMetadataValueEnum::try_from(self.any_val()).u()
    }

    pub fn any_val(&self) -> AnyValueEnum<'ctx> {
        match unsafe { AnyValueEnum::new(self.val) } {
            AnyValueEnum::FunctionValue(f) => {
                // For some reason `f.as_global_value().as_pointer_value().as_any_value_enum().is_pointer_value()` is `false`
                AnyValueEnum::PointerValue(f.as_global_value().as_pointer_value())
            },
            v => v,
        }
    }
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

#[inline]
fn reg<'ctx, U>(v: impl AnyValue<'ctx>) -> CodegenResult<Symbol<'ctx>, U> {
    Ok(reg_sym(v))
}

#[inline]
fn reg_sym<'ctx>(v: impl AnyValue<'ctx>) -> Symbol<'ctx> {
    Symbol::Register(CodegenValue::new(v.as_value_ref()))
}

#[inline]
fn stack_val<'ctx, U>(ptr: PointerValue<'ctx>) -> CodegenResult<Symbol<'ctx>, U> {
    Ok(Symbol::Stack(ptr))
}

#[inline]
fn codegen_val<'ctx>(val: impl AnyValue<'ctx>) -> CodegenResult<CodegenValue<'ctx>> {
    Ok(CodegenValue::new(val.as_value_ref()))
}

#[derive(Debug, Clone, Copy)]
pub struct Loop<'ctx> {
    continue_bb: BasicBlock<'ctx>,
    end_bb: BasicBlock<'ctx>,
}

fn set_alignment(val: impl AsValueRef, alignment: usize) {
    unsafe { InstructionValue::new(val.as_value_ref()) }
        .set_alignment(alignment as u32)
        .u()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CFfiType<'ctx> {
    Zst,
    /// never the [`BasicTypeEnum::StructType`] variant. struct with size <= 8 Bytes will be converted to an int or a float
    Simple(BasicTypeEnum<'ctx>),
    /// never a struct type. struct with 8 < size <= 16 Bytes will be converted to ints or floats
    Simple2([BasicTypeEnum<'ctx>; 2]),
    /// passed as a `ptr` with the `byval` attribute.
    /// returned via `sret`.
    ByValPtr,
    /// functions are always passed and returned as a `ptr` but without the `byval` attribute
    Fn,
    /// arrays are always passed as a `ptr` but without the `byval` attribute.
    /// returned via `sret`.
    Array,
    /// Any simple enum smaller than [`DEFAULT_C_ENUM_BITS`]
    SmallSimpleEnum {
        ffi_int: IntType<'ctx>,
        small_int: IntType<'ctx>,
    },
}

impl<'ctx> CFfiType<'ctx> {
    fn into_basic_ret_ty(self, codegen: &Codegen<'ctx>) -> Option<BasicTypeEnum<'ctx>> {
        match self {
            CFfiType::Zst | CFfiType::ByValPtr | CFfiType::Array => None,
            CFfiType::Fn => Some(codegen.ptr_type().as_basic_type_enum()),
            CFfiType::Simple(basic_type_enum) => Some(basic_type_enum),
            CFfiType::Simple2(fields) => {
                Some(codegen.struct_type_inner(&fields, None, false).as_basic_type_enum())
            },
            CFfiType::SmallSimpleEnum { ffi_int, .. } => Some(ffi_int.as_basic_type_enum()),
        }
    }

    /// converts [`CFfiType::Simple2`] into [`CFfiType::Simple`]
    fn flatten_simple2(self, codegen: &Codegen<'ctx>) -> Self {
        match self {
            CFfiType::Zst
            | CFfiType::Simple(_)
            | CFfiType::ByValPtr
            | CFfiType::Fn
            | CFfiType::Array
            | CFfiType::SmallSimpleEnum { .. } => self,
            CFfiType::Simple2(fields) => CFfiType::Simple(
                codegen.struct_type_inner(&fields, None, false).as_basic_type_enum(),
            ),
        }
    }

    fn as_param_count(&self) -> usize {
        match self {
            CFfiType::Zst => 0,
            CFfiType::Simple(_)
            | CFfiType::ByValPtr
            | CFfiType::Fn
            | CFfiType::Array
            | CFfiType::SmallSimpleEnum { .. } => 1,
            CFfiType::Simple2(_) => 2,
        }
    }

    fn do_use_sret(&self) -> bool {
        matches!(self, CFfiType::ByValPtr | CFfiType::Array)
    }
}

fn has_ffi_noundef(ty: Ptr<ast::Type>, c_ffi_ty: CFfiType<'_>) -> bool {
    match c_ffi_ty {
        CFfiType::Zst => false,
        CFfiType::ByValPtr | CFfiType::Fn | CFfiType::Array | CFfiType::SmallSimpleEnum { .. } => {
            true
        },
        CFfiType::Simple(_) | CFfiType::Simple2(_) => ty.is_ffi_noundef(),
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

pub fn finalize_ty(ty: &mut Ptr<ast::Type>, out_ty: Ptr<ast::Type>) -> Ptr<ast::Type> {
    debug_assert!(ty_match(*ty, out_ty));
    *ty = out_ty;
    *ty
}

/// `f: (value: Ptr<Ast>, param_def: Ptr<ast::Decl>, param_idx: usize) -> CodegenResult<(), U>`
pub fn for_each_call_arg<'ctx, U>(
    params: DeclList,
    args: impl IntoIterator<Item = Ptr<Ast>>,
    mut f: impl FnMut(Ptr<Ast>, Ptr<ast::Decl>, usize) -> CodegenResult<(), U>,
) -> CodegenResult<(), U> {
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

enum CallFnVal<'ctx> {
    Direct(FunctionValue<'ctx>),
    FnPtr(PointerValue<'ctx>, FunctionType<'ctx>),
}

const DEFAULT_C_ENUM_BITS: u32 = 4 * 8;

trait AddAttribute: Copy {
    fn add_attribute(self, loc: AttributeLoc, attribute: Attribute);

    fn add_attributes<const N: usize>(self, loc: AttributeLoc, attributes: [Attribute; N]) {
        for attr in attributes {
            self.add_attribute(loc, attr);
        }
    }

    fn add_ret_attributes<'ctx>(
        self,
        codegen: &mut Codegen<'ctx>,
        ret_ffi_type: CFfiType<'ctx>,
        f: Ptr<ast::Fn>,
    ) {
        if ret_ffi_type.do_use_sret() {
            let llvm_ty = codegen.llvm_type(f.ret_ty.u()).any_ty();
            let sret = codegen
                .context
                .create_type_attribute(Attribute::get_named_enum_kind_id("sret"), llvm_ty);
            self.add_attributes(AttributeLoc::Param(0), [sret, codegen.noundef]);
        } else if has_ffi_noundef(f.ret_ty.u(), ret_ffi_type) {
            // Note: clang doesn't add the noundef attribute to return types when compiling C code.
            self.add_attribute(AttributeLoc::Return, codegen.noundef);
        }
    }

    /// Returns the result of [`CFfiType::as_param_count`] for `param_ffi_type`.
    fn add_param_attributes<'ctx>(
        self,
        codegen: &mut Codegen<'ctx>,
        param: Ptr<ast::Decl>,
        param_ffi_type: CFfiType<'ctx>,
        param_ffi_idx: u32,
    ) -> u32 {
        match param_ffi_type {
            CFfiType::ByValPtr => {
                let llvm_ty = codegen.llvm_type(param.var_ty.u()).any_ty();
                let byval = codegen
                    .context
                    .create_type_attribute(Attribute::get_named_enum_kind_id("byval"), llvm_ty);
                self.add_attribute(AttributeLoc::Param(param_ffi_idx), byval);
            },
            _ => {},
        }

        let ffi_param_count = param_ffi_type.as_param_count() as u32;
        if has_ffi_noundef(param.var_ty.u(), param_ffi_type) {
            for sub_idx in 0..ffi_param_count {
                self.add_attribute(AttributeLoc::Param(param_ffi_idx + sub_idx), codegen.noundef);
            }
        }
        ffi_param_count
    }
}

impl<'ctx> AddAttribute for FunctionValue<'ctx> {
    #[inline]
    fn add_attribute(self, loc: AttributeLoc, attribute: Attribute) {
        self.add_attribute(loc, attribute);
    }
}

impl<'ctx> AddAttribute for CallSiteValue<'ctx> {
    #[inline]
    fn add_attribute(self, loc: AttributeLoc, attribute: Attribute) {
        self.add_attribute(loc, attribute);
    }
}

#[derive(Clone, Copy)]
enum FnKind {
    FnDef(Ptr<ast::Decl>),
    Lambda,
}

impl FnKind {
    fn is_lamda_or(self, f: impl FnOnce(Ptr<ast::Decl>) -> bool) -> bool {
        match self {
            FnKind::FnDef(def) => f(def),
            FnKind::Lambda => true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EnumTagType<'ctx> {
    Zero,
    One { tag: isize },
    IntTy(IntType<'ctx>),
}

impl<'ctx> EnumTagType<'ctx> {
    fn sized(self) -> Option<IntType<'ctx>> {
        if let EnumTagType::IntTy(int_ty) = self { Some(int_ty) } else { None }
    }
}

fn new_raw_const_struct<'ctx>(
    struct_ty: StructType<'ctx>,
    args: &mut [LLVMValueRef],
) -> StructValue<'ctx> {
    unsafe {
        StructValue::new(inkwell::llvm_sys::core::LLVMConstNamedStruct(
            struct_ty.as_type_ref(),
            args.as_mut_ptr(),
            args.len() as u32,
        ))
    }
}
