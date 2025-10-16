use crate::{
    ast::{self, AstKind, UpcastToAst},
    context::ctx,
    display_code::display,
    parser::lexer::Span,
    ptr::Ptr,
    util::{UnwrapDebug, unreachable_debug},
};
use core::fmt;
use std::{cmp::Ordering, fmt::Display, io::Write};

pub trait DiagnosticReporter {
    #[track_caller]
    fn report(&mut self, severity: DiagnosticSeverity, span: Span, msg: impl Display);

    #[track_caller]
    fn hint(&mut self, span: Span, msg: impl Display);

    fn max_past_severity(&self) -> Option<DiagnosticSeverity>;

    #[track_caller]
    fn error(&mut self, span: Span, msg: impl Display) {
        self.report(DiagnosticSeverity::Error, span, msg)
    }

    #[track_caller]
    fn error_without_code(&mut self, msg: impl Display) {
        self.error(Span::ZERO, msg)
    }

    #[track_caller]
    fn warn(&mut self, span: Span, msg: impl Display) {
        self.report(DiagnosticSeverity::Warn, span, msg)
    }

    #[track_caller]
    fn info(&mut self, span: Span, msg: impl Display) {
        self.report(DiagnosticSeverity::Info, span, msg)
    }

    fn do_abort_compilation(&self) -> bool {
        self.max_past_severity().is_some_and(DiagnosticSeverity::aborts_compilation)
    }

    // some common diagnostics:

    #[track_caller]
    fn error_cannot_yield_from_loop_block(&mut self, span: Span) {
        self.error(span, "cannot yield a value from a loop block.");
    }

    #[track_caller]
    fn error_mismatched_types<Expected: fmt::Display>(
        &self,
        span: Span,
        expected: Expected,
        got: Ptr<ast::Type>,
    ) -> HandledErr {
        cerror!(span, "mismatched types: expected {expected}; got {got}")
    }

    #[track_caller]
    fn error_mismatched_types_binop(
        &mut self,
        span: Span,
        lhs_ty: Ptr<ast::Type>,
        rhs_ty: Ptr<ast::Type>,
    ) -> HandledErr {
        cerror!(span, "mismatched types (left: {lhs_ty}, right: {rhs_ty})")
    }

    #[track_caller]
    fn error_duplicate_named_arg(&self, arg_name: Ptr<ast::Ident>) {
        cerror!(arg_name.span, "Parameter '{}' specified multiple times", &arg_name.sym);
    }

    #[track_caller]
    fn error_unknown_field(&self, field: Ptr<ast::Ident>, ty: Ptr<ast::Type>) -> HandledErr {
        cerror!(field.span, "no field `{}` on type `{}`", field.sym, ty)
    }

    #[track_caller]
    fn error_unknown_variant(&self, variant: Ptr<ast::Ident>, ty: Ptr<ast::Type>) -> HandledErr {
        cerror!(variant.span, "no variant `{}` on enum type `{}`", variant.sym, ty)
    }

    #[track_caller]
    fn error_cannot_apply_initializer(
        &self,
        analyzed_lhs: Ptr<ast::Ast>,
        initializer_expr: Ptr<ast::Ast>,
    ) {
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
                chint!(
                    initializer_expr.span,
                    "Consider using an array initializer (`.[...]`) instead"
                )
            }
        } else {
            cerror!(
                span,
                "Cannot apply {initializer_kind} to a value of type `{}`",
                analyzed_lhs.ty.u()
            );
        }
    }

    #[track_caller]
    fn error_non_const(&mut self, runtimevalue: Ptr<ast::Ast>, msg: impl Display) -> HandledErr {
        self.error(runtimevalue.full_span(), msg);
        // TODO: label: not a compile time known value
        // this help doesn't make sense when `runtimevalue` is a local variable
        chint!(
            runtimevalue.full_span(),
            "help: consider using `#run` to evaluate expression at compile time"
        );
        HandledErr
    }

    #[track_caller]
    fn error_non_const_initializer_field(&mut self, field_init: Ptr<ast::Ast>) -> HandledErr {
        self.error_non_const(
            field_init,
            "fields of constant struct values must be known at compile time",
        )
    }

    #[track_caller]
    fn error_const_ptr_initializer(&self, initializer: Ptr<ast::Ast>) -> HandledErr {
        cerror!(
            initializer.full_span(),
            "cannot initialize a struct behind a pointer at compile time"
        )
    }

    #[track_caller]
    fn error_const_call(&self, call: Ptr<ast::Call>) -> HandledErr {
        let full_span = call.upcast().full_span();
        cerror!(full_span, "Cannot directly call a function in a constant");
        chint!(
            full_span.start(),
            "Consider using the `#run` directive to evaluate the function at compile time \
             (currently not implemented): {}",
            call.func
                .try_flat_downcast::<ast::Ident>()
                .map(|i| format!(": `#run {}(...)`", i.sym))
                .unwrap_or_default()
        );
        HandledErr
    }

    #[track_caller]
    fn error_unimplemented(&mut self, span: Span, what: fmt::Arguments<'_>) {
        cerror!(span, "{what} is currently not implemented");
    }
}

/// default [`DiagnosticReporter`]. Prints diagnostics to the terminal
#[derive(Default)]
pub struct DiagnosticPrinter {
    max_past_severity: Option<DiagnosticSeverity>,
}

impl DiagnosticReporter for DiagnosticPrinter {
    fn report(&mut self, severity: DiagnosticSeverity, span: Span, msg: impl Display) {
        self.max_past_severity = self.max_past_severity.max(Some(severity));
        if severity < ctx().args.diagnostic_level {
            return;
        }
        let mut stderr = std::io::stderr();
        write!(&mut stderr, "{severity}: {msg}").unwrap();
        #[cfg(debug_assertions)]
        {
            write!(&mut stderr, " (reported @ '{}')", std::panic::Location::caller()).unwrap();
        }
        writeln!(&mut stderr, "{COLOR_UNSET}").unwrap();
        if span != Span::ZERO {
            display(span).color_code(severity.text_color()).finish();
        }
        writeln!(&mut stderr).unwrap();
    }

    fn hint(&mut self, span: Span, msg: impl Display) {
        display(span)
            .color_code(DiagnosticSeverity::Info.text_color())
            .label(&format!("{msg}"))
            .finish();
        println!()
    }

    fn max_past_severity(&self) -> Option<DiagnosticSeverity> {
        self.max_past_severity
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct SavedDiagnosticMessage {
    pub severity: DiagnosticSeverity,
    pub span: crate::parser::lexer::Span,
    pub msg: Box<str>,
}

/// Collects diagnostics instead of printing them.
///
/// Only used in error message tests.
#[derive(Default)]
#[cfg(test)]
pub struct DiagnosticCollector {
    pub printer: DiagnosticPrinter,
    pub diagnostics: Vec<SavedDiagnosticMessage>,
}

#[cfg(test)]
impl DiagnosticReporter for DiagnosticCollector {
    fn report(&mut self, severity: DiagnosticSeverity, span: Span, msg: impl Display) {
        self.printer.report(severity, span, &msg);
        let msg = msg.to_string().into_boxed_str();
        self.diagnostics.push(SavedDiagnosticMessage { severity, span, msg })
    }

    fn hint(&mut self, span: Span, msg: impl Display) {
        self.printer.hint(span, &msg);
        self.diagnostics.push(SavedDiagnosticMessage {
            severity: DiagnosticSeverity::Info,
            span,
            msg: msg.to_string().into_boxed_str(),
        })
    }

    fn max_past_severity(&self) -> Option<DiagnosticSeverity> {
        self.diagnostics.iter().map(|m| m.severity).max()
    }
}

pub const COLOR_UNSET: &str = "\x1b[0m";
pub const COLOR_BOLD_RED_INV: &str = "\x1b[7;1;38;5;9m";
pub const COLOR_BOLD_RED: &str = "\x1b[1;38;5;9m";
pub const COLOR_RED: &str = "\x1b[0;38;5;9m";
pub const COLOR_BOLD_YELLOW: &str = "\x1b[1;33m";
pub const COLOR_YELLOW: &str = "\x1b[0;33m";
pub const COLOR_BOLD_CYAN: &str = "\x1b[1;36m";
pub const COLOR_CYAN: &str = "\x1b[0;36m";

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DiagnosticSeverity {
    /// Internal error
    Fatal,
    /// Compiler error
    Error,
    /// Compiler warning
    Warn,
    /// Compiler hint
    Info,
}

impl PartialOrd for DiagnosticSeverity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DiagnosticSeverity {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8)).reverse()
    }
}

impl DiagnosticSeverity {
    pub fn aborts_compilation(self) -> bool {
        self >= DiagnosticSeverity::Error
    }

    pub fn label_color(self) -> &'static str {
        match self {
            DiagnosticSeverity::Fatal => COLOR_BOLD_RED_INV,
            DiagnosticSeverity::Error => COLOR_BOLD_RED,
            DiagnosticSeverity::Warn => COLOR_BOLD_YELLOW,
            DiagnosticSeverity::Info => COLOR_BOLD_CYAN,
        }
    }

    pub fn text_color(self) -> &'static str {
        match self {
            DiagnosticSeverity::Fatal => COLOR_RED,
            DiagnosticSeverity::Error => COLOR_RED,
            DiagnosticSeverity::Warn => COLOR_YELLOW,
            DiagnosticSeverity::Info => COLOR_CYAN,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DiagnosticSeverity::Fatal => "FATAL",
            DiagnosticSeverity::Error => "ERROR",
            DiagnosticSeverity::Warn => "WARN",
            DiagnosticSeverity::Info => "INFO",
        }
    }
}

impl fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.label_color(), self.label(), self.text_color())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct HandledErr;

impl<T, E: From<HandledErr>> From<HandledErr> for Result<T, E> {
    fn from(value: HandledErr) -> Self {
        Err(E::from(value))
    }
}

macro_rules! cerror {
    ($span:expr, $fmt:literal $( , $args:expr )* $(,)?) => {{
        crate::diagnostics::DiagnosticReporter::error(crate::context::ctx_mut(), $span, &format_args!($fmt, $($args),*));
        crate::diagnostics::HandledErr
    }};
}
pub(crate) use cerror;

macro_rules! cerror2 {
    ($span:expr, $fmt:literal $( , $args:expr )* $(,)?) => {{
        crate::diagnostics::DiagnosticReporter::error(crate::context::ctx_mut(), $span, &format_args!($fmt, $($args),*));
        crate::diagnostics::HandledErr.into()
    }};
}
pub(crate) use cerror2;

macro_rules! cwarn {
    ($span:expr, $fmt:literal $( , $args:expr )* $(,)?) => {
        crate::diagnostics::DiagnosticReporter::warn(crate::context::ctx_mut(), $span, &format_args!($fmt, $($args),*))
    };
}
pub(crate) use cwarn;

macro_rules! cinfo {
    ($span:expr, $fmt:literal $( , $args:expr )* $(,)?) => {
        crate::diagnostics::DiagnosticReporter::info(crate::context::ctx_mut(), $span, &format_args!($fmt, $($args),*))
    };
}
pub(crate) use cinfo;

macro_rules! chint {
    ($span:expr, $fmt:literal $( , $args:expr )* $(,)?) => {
        crate::diagnostics::DiagnosticReporter::hint(crate::context::ctx_mut(), $span, &format_args!($fmt, $($args),*))
    };
}
pub(crate) use chint;

macro_rules! cunimplemented {
    ($span:expr, $fmt:literal $( , $args:expr )* $(,)?) => {{
        crate::diagnostics::DiagnosticReporter::error_unimplemented(crate::context::ctx_mut(), $span, format_args!($fmt, $($args),*));
        return crate::diagnostics::HandledErr.into()
    }};
}
pub(crate) use cunimplemented;
