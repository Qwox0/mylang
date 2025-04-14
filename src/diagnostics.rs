use crate::{ast, display_code::display, error::SpannedError, parser::lexer::Span, ptr::Ptr};
use core::fmt;
use std::{cmp::Ordering, fmt::Display, io::Write};

pub trait DiagnosticReporter {
    #[track_caller]
    fn report<M: Display + ?Sized>(&mut self, severity: DiagnosticSeverity, span: Span, msg: &M);

    #[track_caller]
    fn hint<M: Display + ?Sized>(&mut self, span: Span, msg: &M);

    fn max_past_severity(&self) -> Option<DiagnosticSeverity>;

    #[track_caller]
    fn error<M: Display + ?Sized>(&mut self, span: Span, msg: &M) {
        self.report(DiagnosticSeverity::Error, span, msg)
    }

    #[track_caller]
    fn error_without_code<M: Display + ?Sized>(&mut self, msg: &M) {
        self.error(Span::ZERO, msg)
    }

    #[track_caller]
    fn warn<M: Display + ?Sized>(&mut self, span: Span, msg: &M) {
        self.report(DiagnosticSeverity::Warn, span, msg)
    }

    #[track_caller]
    fn info<M: Display + ?Sized>(&mut self, span: Span, msg: &M) {
        self.report(DiagnosticSeverity::Info, span, msg)
    }

    fn do_abort_compilation(&self) -> bool {
        self.max_past_severity().is_some_and(DiagnosticSeverity::aborts_compilation)
    }

    #[track_caller]
    fn report2(&mut self, severity: DiagnosticSeverity, err: &impl SpannedError) {
        if !err.was_already_handled() {
            self.report(severity, err.span(), &err.get_text());
        }
    }

    #[track_caller]
    fn error2(&mut self, err: &impl SpannedError) {
        self.report2(DiagnosticSeverity::Error, err)
    }

    #[track_caller]
    fn warn2(&mut self, err: &impl SpannedError) {
        self.report2(DiagnosticSeverity::Warn, err)
    }

    // some common diagnostics:

    #[track_caller]
    fn error_cannot_yield_from_loop_block(&mut self, span: Span) {
        self.error(span, "cannot yield a value from a loop block.");
    }

    #[track_caller]
    fn error_mismatched_types(
        &mut self,
        span: Span,
        expected: Ptr<ast::Type>,
        got: Ptr<ast::Type>,
    ) {
        self.error(span, &format_args!("expected {expected}; got {got}"));
    }

    #[track_caller]
    fn error_duplicate_named_arg(&mut self, arg_name: Ptr<ast::Ident>) {
        cerror!(arg_name.span, "Parameter '{}' specified multiple times", &arg_name.text);
    }
}

/// default [`DiagnosticReporter`]. Prints diagnostics to the terminal
#[derive(Default)]
pub struct DiagnosticPrinter {
    max_past_severity: Option<DiagnosticSeverity>,
}

impl DiagnosticReporter for DiagnosticPrinter {
    fn report<M: Display + ?Sized>(&mut self, severity: DiagnosticSeverity, span: Span, msg: &M) {
        self.max_past_severity = self.max_past_severity.max(Some(severity));
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

    fn hint<M: Display + ?Sized>(&mut self, span: Span, msg: &M) {
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
    fn report<M: Display + ?Sized>(&mut self, severity: DiagnosticSeverity, span: Span, msg: &M) {
        self.printer.report(severity, span, msg);
        let msg = msg.to_string().into_boxed_str();
        self.diagnostics.push(SavedDiagnosticMessage { severity, span, msg })
    }

    fn hint<M: Display + ?Sized>(&mut self, span: Span, msg: &M) {
        self.printer.hint(span, msg);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

macro_rules! cerror {
    ($span:expr, $fmt:literal $( , $args:expr )* $(,)?) => {
        crate::diagnostics::DiagnosticReporter::error(crate::context::ctx_mut(), $span, &format_args!($fmt, $($args),*))
    };
}
pub(crate) use cerror;

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
