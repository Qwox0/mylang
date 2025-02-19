use crate::{context::code, error::SpannedError, parser::lexer::Span, util::display_span_in_code};
use core::fmt;
use std::cmp::Ordering;

const COLOR_BOLD_RED_INV: &str = "\x1b[1;7;91m";
const COLOR_BOLD_RED: &str = "\x1b[1;91m";
// const COLOR_BOLD_GREEN: &str = "\x1b[1;92m";
const COLOR_BOLD_YELLOW: &str = "\x1b[1;93m";
const COLOR_UNSET: &str = "\x1b[0m";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord)]
pub enum DiagnosticSeverity {
    /// Internal error
    Fatal,
    /// Compiler error
    Error,
    /// Compiler warning
    Warn,
}

impl PartialOrd for DiagnosticSeverity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (*self as u8).partial_cmp(&(*other as u8)).map(Ordering::reverse)
    }
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
    fn report(&mut self, severity: DiagnosticSeverity, err: &impl SpannedError);

    fn max_past_severity(&self) -> Option<DiagnosticSeverity>;

    fn error(&mut self, err: &impl SpannedError) {
        self.report(DiagnosticSeverity::Error, err)
    }

    fn warn(&mut self, err: &impl SpannedError) {
        self.report(DiagnosticSeverity::Warn, err)
    }

    fn do_abort_compilation(&self) -> bool {
        self.max_past_severity().is_some_and(|sev| sev >= DiagnosticSeverity::Error)
    }
}

/// default [`DiagnosticReporter`]. Prints diagnostics to the terminal
#[derive(Default)]
pub struct DiagnosticPrinter {
    max_past_severity: Option<DiagnosticSeverity>,
}

impl DiagnosticReporter for DiagnosticPrinter {
    fn report(&mut self, severity: DiagnosticSeverity, err: &impl SpannedError) {
        self.max_past_severity = self.max_past_severity.max(Some(severity));
        eprintln!("{severity}: {}", err.get_text());
        display_span_in_code(err.span(), &code());
        println!()
    }

    fn max_past_severity(&self) -> Option<DiagnosticSeverity> {
        self.max_past_severity
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
    pub diagnostics: Vec<SavedDiagnosticMessage>,
}

impl DiagnosticReporter for DiagnosticCollector {
    fn report(&mut self, severity: DiagnosticSeverity, err: &impl SpannedError) {
        DiagnosticPrinter::default().report(severity, err);
        self.diagnostics.push(SavedDiagnosticMessage {
            severity,
            span: err.span(),
            msg: err.get_text().into_boxed_str(),
        })
    }

    fn max_past_severity(&self) -> Option<DiagnosticSeverity> {
        self.diagnostics.iter().map(|m| m.severity).max()
    }
}
