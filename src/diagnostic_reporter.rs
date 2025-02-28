use crate::{error::SpannedError, parser::lexer::Span, util::display_span_in_code};
use core::fmt;
use std::{cmp::Ordering, fmt::Display};

pub trait DiagnosticReporter {
    fn report<M: Display + ?Sized>(&mut self, severity: DiagnosticSeverity, span: Span, msg: &M);

    fn max_past_severity(&self) -> Option<DiagnosticSeverity>;

    fn do_abort_compilation(&self) -> bool {
        self.max_past_severity().is_some_and(|sev| sev >= DiagnosticSeverity::Error)
    }

    fn report2(&mut self, severity: DiagnosticSeverity, err: &impl SpannedError) {
        self.report(severity, err.span(), &err.get_text());
    }

    fn error2(&mut self, err: &impl SpannedError) {
        self.report2(DiagnosticSeverity::Error, err)
    }

    fn warn2(&mut self, err: &impl SpannedError) {
        self.report2(DiagnosticSeverity::Warn, err)
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
        eprintln!("{severity}: {msg}{COLOR_UNSET}");
        display_span_in_code(span);
        println!()
    }

    fn max_past_severity(&self) -> Option<DiagnosticSeverity> {
        self.max_past_severity
    }
}

#[cfg(test)]
#[derive(Debug)]
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
    pub diagnostics: Vec<SavedDiagnosticMessage>,
}

#[cfg(test)]
impl DiagnosticReporter for DiagnosticCollector {
    fn report<M: Display + ?Sized>(&mut self, severity: DiagnosticSeverity, span: Span, msg: &M) {
        DiagnosticPrinter::default().report(severity, span, msg);
        let msg = msg.to_string().into_boxed_str();
        self.diagnostics.push(SavedDiagnosticMessage { severity, span, msg })
    }

    fn max_past_severity(&self) -> Option<DiagnosticSeverity> {
        self.diagnostics.iter().map(|m| m.severity).max()
    }
}

const COLOR_BOLD_RED_INV: &str = "\x1b[1;7;91m";
const COLOR_BOLD_RED: &str = "\x1b[1;91m";
const COLOR_RED: &str = "\x1b[0;91m";
// const COLOR_BOLD_GREEN: &str = "\x1b[1;92m";
const COLOR_BOLD_YELLOW: &str = "\x1b[1;93m";
const COLOR_YELLOW: &str = "\x1b[0;93m";
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
            DiagnosticSeverity::Fatal => write!(f, "{COLOR_BOLD_RED_INV}FATAL{COLOR_RED}"),
            DiagnosticSeverity::Error => write!(f, "{COLOR_BOLD_RED}ERROR{COLOR_RED}"),
            DiagnosticSeverity::Warn => write!(f, "{COLOR_BOLD_YELLOW}WARN{COLOR_YELLOW}"),
        }
    }
}

macro_rules! cerror {
    ($span:expr, $msg:expr $(,)?) => {
        crate::context::ctx_mut()
            .diagnostic_reporter
            .report(crate::diagnostic_reporter::DiagnosticSeverity::Error, $span, $msg)
    };
    ($span:expr, $fmt:literal, $( $args:expr ),* $(,)?) => {
        crate::context::ctx_mut()
            .diagnostic_reporter
            .report(crate::diagnostic_reporter::DiagnosticSeverity::Error, $span, &format!($fmt, $($args),*))
    };
}
pub(crate) use cerror;
