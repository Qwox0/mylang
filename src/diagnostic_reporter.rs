use crate::{context::code, error::SpannedError, parser::lexer::Span, util::display_span_in_code};
use core::fmt;
use std::cell::RefCell;

const COLOR_BOLD_RED_INV: &str = "\x1b[1;7;91m";
const COLOR_BOLD_RED: &str = "\x1b[1;91m";
// const COLOR_BOLD_GREEN: &str = "\x1b[1;92m";
const COLOR_BOLD_YELLOW: &str = "\x1b[1;93m";
const COLOR_UNSET: &str = "\x1b[0m";

#[derive(Debug, Clone, Copy)]
pub enum DiagnosticSeverity {
    /// Internal error
    Fatal,
    /// Compiler error
    Error,
    /// Compiler warning
    Warn,
}

impl fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticSeverity::Fatal => write!(f, "{COLOR_BOLD_RED_INV}FATAL{COLOR_UNSET}"),
            DiagnosticSeverity::Error => write!(f, "{COLOR_BOLD_RED}ERROR{COLOR_UNSET}"),
            DiagnosticSeverity::Warn => write!(f, "{COLOR_BOLD_YELLOW}WARN{COLOR_UNSET}"),
        }
    }
}

pub trait DiagnosticReporter {
    fn report(&self, severity: DiagnosticSeverity, err: &dyn SpannedError);

    fn error(&self, err: &dyn SpannedError) {
        self.report(DiagnosticSeverity::Error, err)
    }

    fn warn(&self, err: &dyn SpannedError) {
        self.report(DiagnosticSeverity::Warn, err)
    }
}

/// default [`DiagnosticReporter`]. Prints diagnostics to the terminal
#[derive(Default)]
pub struct DiagnosticPrinter {}

impl DiagnosticReporter for DiagnosticPrinter {
    fn report(&self, severity: DiagnosticSeverity, err: &dyn SpannedError) {
        eprintln!("{severity}: {}", err.get_text());
        display_span_in_code(err.span(), &code())
    }
}

pub struct SavedDiagnosticMessage {
    pub severity: DiagnosticSeverity,
    pub span: Span,
    pub msg: Box<str>,
}

/// Collects diagnostics instead of printing them.
///
/// Only used in error message tests.
#[derive(Default)]
pub struct DiagnosticCollector {
    pub diagnostics: RefCell<Vec<SavedDiagnosticMessage>>,
}

impl DiagnosticReporter for DiagnosticCollector {
    fn report(&self, severity: DiagnosticSeverity, err: &dyn SpannedError) {
        DiagnosticPrinter {}.report(severity, err);
        self.diagnostics.borrow_mut().push(SavedDiagnosticMessage {
            severity,
            span: err.span(),
            msg: err.get_text().into_boxed_str(),
        })
    }
}
