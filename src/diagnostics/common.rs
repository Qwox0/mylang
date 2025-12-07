use crate::{
    ast::{self, AstKind, UpcastToAst},
    diagnostics::{HandledErr, cerror, chint},
    parser::lexer::Span,
    ptr::Ptr,
    util::{UnwrapDebug, unreachable_debug},
};
use std::fmt::{self, Display};

#[track_caller]
pub fn error_cannot_yield_from_loop_block(span: Span) -> HandledErr {
    cerror!(span, "cannot yield a value from a loop block.")
}

#[track_caller]
pub fn error_mismatched_types(
    span: Span,
    expected: Ptr<ast::Type>,
    got: Ptr<ast::Type>,
) -> HandledErr {
    // TODO: error_mismatched_types_custom(span, format_args!("`{expected}`"), got)
    error_mismatched_types_custom(span, format_args!("{expected}"), got)
}

#[track_caller]
pub fn error_mismatched_types_custom(
    span: Span,
    expected: impl fmt::Display,
    got: Ptr<ast::Type>,
) -> HandledErr {
    // TODO: cerror!(span, "mismatched types: expected {expected}; got `{got}`")
    cerror!(span, "mismatched types: expected {expected}; got {got}")
}

#[track_caller]
pub fn error_mismatched_types_binop(
    span: Span,
    lhs_ty: Ptr<ast::Type>,
    rhs_ty: Ptr<ast::Type>,
) -> HandledErr {
    // TODO: cerror!(span, "mismatched types (left: `{lhs_ty}`, right: `{rhs_ty}`)")
    cerror!(span, "mismatched types (left: {lhs_ty}, right: {rhs_ty})")
}

#[track_caller]
pub fn error_duplicate_named_arg(arg_name: Ptr<ast::Ident>) -> HandledErr {
    cerror!(arg_name.span, "Parameter '{}' specified multiple times", &arg_name.sym)
}

#[track_caller]
pub fn error_unknown_field(field: Ptr<ast::Ident>, ty: Ptr<ast::Type>) -> HandledErr {
    cerror!(field.span, "no field `{}` on type `{}`", field.sym, ty)
}

#[track_caller]
pub fn error_unknown_variant(variant: Ptr<ast::Ident>, ty: Ptr<ast::Type>) -> HandledErr {
    cerror!(variant.span, "no variant `{}` on enum type `{}`", variant.sym, ty)
}

#[track_caller]
pub fn error_cannot_apply_initializer(
    analyzed_lhs: Ptr<ast::Ast>,
    initializer_expr: Ptr<ast::Ast>,
) -> HandledErr {
    let initializer_kind = initializer_expr.kind.initializer_kind();
    let lhs_expr = match initializer_expr.matchable().as_ref() {
        ast::AstEnum::PositionalInitializer { lhs, .. }
        | ast::AstEnum::NamedInitializer { lhs, .. }
        | ast::AstEnum::ArrayInitializer { lhs, .. }
        | ast::AstEnum::ArrayInitializerShort { lhs, .. } => lhs,
        _ => unreachable_debug(),
    };
    let span = lhs_expr.unwrap_or(initializer_expr).full_span();
    if let Some(lhs_ty) = analyzed_lhs.try_downcast_type() {
        cerror!(span, "Cannot initialize a value of type `{lhs_ty}` using {initializer_kind}");
        if lhs_ty.kind == AstKind::StructDef {
            chint!(
                initializer_expr.span,
                "Consider using a positional initializer (`.(...)`) or named initializer \
                 (`.{{...}}`) instead"
            )
        } else if lhs_ty.kind == AstKind::ArrayTy {
            chint!(initializer_expr.span, "Consider using an array initializer (`.[...]`) instead")
        }
    } else {
        cerror!(
            span,
            "Cannot apply {initializer_kind} to a value of type `{}`",
            analyzed_lhs.ty.u()
        );
    }
    HandledErr
}

#[track_caller]
pub fn error_const_ptr_initializer(initializer: Ptr<ast::Ast>) -> HandledErr {
    cerror!(initializer.full_span(), "cannot initialize a struct behind a pointer at compile time")
}

#[track_caller]
pub fn error_non_const(runtimevalue: Ptr<ast::Ast>, msg: impl Display) -> HandledErr {
    cerror!(runtimevalue.full_span(), "{msg}");
    // TODO: label: not a compile time known value
    // this help doesn't make sense when `runtimevalue` is a local variable
    chint!(
        runtimevalue.full_span(),
        "help: consider using `#run` to evaluate expression at compile time"
    );
    HandledErr
}

#[track_caller]
pub fn error_non_const_initializer_field(field_init: Ptr<ast::Ast>) -> HandledErr {
    error_non_const(field_init, "fields of constant struct values must be known at compile time")
}

#[track_caller]
pub fn error_const_call(call: Ptr<ast::Call>) -> HandledErr {
    let full_span = call.upcast().full_span();
    cerror!(full_span, "Cannot directly call a function in a constant");
    chint!(
        full_span.start(),
        "Consider using the `#run` directive to evaluate the function at compile time (currently \
         not implemented): {}",
        call.func
            .try_flat_downcast::<ast::Ident>()
            .map(|i| format!(": `#run {}(...)`", i.sym))
            .unwrap_or_default()
    );
    HandledErr
}

#[track_caller]
pub fn error_unimplemented(span: Span, what: fmt::Arguments<'_>) -> HandledErr {
    cerror!(span, "{what} is currently not implemented")
}
