use super::{
    lexer::{Code, Lexer, Span, Token},
    result_with_fatal::ResultWithFatal::*,
    util::TupleMap0,
    ws0, ws1, Expr, Stmt, VarDeclMarkerKind,
};
use crate::parser::result_with_fatal::ResultWithFatal;
use core::fmt;
use std::{ops::ControlFlow, rc::Rc};

/// always returns [`Ok`] variant
macro_rules! always {
    ($val:expr) => {
        f(|lex| Ok(($val, lex)))
    };
}
pub(crate) use always;

macro_rules! err {
    ($kind:ident $( ( $( $field:expr ),* $(,)? ) )? , $span:expr) => {
        Err(ParseError::new(PErrKind::$kind $( ( $($field),* ) )?, $span))
    };
}
pub(crate) use err;

#[derive(Debug)]
pub enum PErrKind {
    NoInput,
    UnexpectedToken(Token),
    NotAnIdent,
    NotAKeyword,
    /// `let mut = ...`
    MissingLetIdent,
    /// `let mut mut x;`
    /// `        ^^^`
    DoubleLetMarker(VarDeclMarkerKind),
    /// `let x x;`
    /// `      ^`
    TooManyLetIdents(Span),

    NotWasFound,

    Tmp(&'static str, Span),
    TODO,
}

pub struct ParseError {
    pub kind: PErrKind,
    pub span: Span,

    #[cfg(debug_assertions)]
    pub context: anyhow::Error,
}

impl std::fmt::Debug for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("ParseError");
        let f = f.field("kind", &self.kind).field("span", &self.span);
        #[cfg(debug_assertions)]
        let f = f.field("context", &format!("{:#}", &self.context));
        f.finish()
    }
}

impl ParseError {
    pub fn new(kind: PErrKind, span: Span) -> Self {
        Self {
            kind,
            span,
            #[cfg(debug_assertions)]
            context: anyhow::Error::msg("root"),
        }
    }

    #[cfg(debug_assertions)]
    pub fn add_context(self, context: impl fmt::Display + Send + Sync + 'static) -> Self {
        let context = self.context.context(context);
        Self { context, ..self }
    }

    #[cfg(not(debug_assertions))]
    pub fn add_context(self, _context: impl fmt::Display + Send + Sync + 'static) -> Self {
        self
    }
}

pub type PResult<'l, T> = ResultWithFatal<(T, Lexer<'l>), ParseError>;

/// This replaces the Parser trait for functions to reduce the extreme compile
/// times.
pub struct Parser<T> {
    f: Rc<dyn for<'c> Fn(Lexer<'c>) -> PResult<'c, T>>,
}

impl<T> Clone for Parser<T> {
    fn clone(&self) -> Self {
        Self { f: self.f.clone() }
    }
}

impl<T: 'static> Parser<T> {
    pub fn new(f: impl Fn(Lexer<'_>) -> PResult<'_, T> + 'static) -> Parser<T> {
        Parser { f: Rc::new(f) }
    }

    pub fn run<'c>(&self, lex: Lexer<'c>) -> PResult<'c, T> {
        (self.f)(lex)
    }

    /// ```text
    /// | self output   | other output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Ok            | /             | Ok            |
    /// | Err ?         | Ok            | Ok            |
    /// | Err ?         | Err e2        | Err e2        |
    /// | Err ?         | Fatal         | Fatal         |
    /// | Fatal         | /             | Fatal         |
    /// ```
    pub fn or(self, other: Parser<T>) -> Parser<T> {
        Parser::new(move |lex| self.run(lex.clone()).or_else(|_| other.run(lex)))
    }

    /// ```text
    /// | self output   | other output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Ok t1         | Ok t2         | Ok (t1, t2)   |
    /// | Ok ?          | Err           | Err           |
    /// | Ok ?          | Fatal         | Fatal         |
    /// | Err           | /             | Err           |
    /// | Fatal         | /             | Fatal         |
    /// ```
    pub fn and<T2: 'static>(self, other: Parser<T2>) -> Parser<(T, T2)> {
        Parser::new(move |lex| {
            self.run(lex)
                .and_then(|(t1, lex)| other.run(lex).map(|(t2, lex)| ((t1, t2), lex)))
        })
    }

    /// like [`Parser::and`] but only keeps the lhs
    pub fn and_l<T2: 'static>(self, other: Parser<T2>) -> Parser<T> {
        self.and(other).map(|(lhs, _)| lhs) // compile time: 0.97s -> 6.85s
        // Parser::new(move |lex| {
        //     self.run(lex).and_then(|(t1, lex)| other.run(lex).map(|(_, lex)|
        // (t1, lex))) })
    }

    /// like [`Parser::and`] but only keeps the rhs
    pub fn and_r<T2: 'static>(self, other: Parser<T2>) -> Parser<T2> {
        self.and(other).map(|(_, rhs)| rhs)
        // Parser::new(move |lex| {
        //     self.run(lex).and_then(|(_, lex)| other.run(lex).map(|(t2, lex)|
        // (t2, lex))) })
    }

    /// if self results in [`Ok`] then all [`Err`] from other are converted to
    /// [`Fatal`].
    ///
    /// ```text
    /// | self output   | other output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Ok t1         | Ok t2         | Ok (t1, t2)   |
    /// | Ok ?          | Err           | Fatal         |
    /// | Ok ?          | Fatal         | Fatal         |
    /// | Err           | /             | Err           |
    /// | Fatal         | /             | Fatal         |
    /// ```
    pub fn and_fatal<T2: 'static>(self, other: Parser<T2>) -> Parser<(T, T2)> {
        self.and(other.err_to_fatal())
    }

    /// ```text
    /// | self output   | other output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Ok t1         | Ok ?          | Ok t1         |
    /// | Ok ?          | Err           | Fatal         |
    /// | Ok ?          | Fatal         | Fatal         |
    /// | Err           | /             | Err           |
    /// | Fatal         | /             | Fatal         |
    /// ```
    pub fn and_l_fatal<T2: 'static>(self, other: Parser<T2>) -> Parser<T> {
        self.and_l(other.err_to_fatal())
    }

    /// ```text
    /// | self output   | other output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Ok ?          | Ok t2         | Ok t2         |
    /// | Ok ?          | Err           | Fatal         |
    /// | Ok ?          | Fatal         | Fatal         |
    /// | Err           | /             | Err           |
    /// | Fatal         | /             | Fatal         |
    /// ```
    pub fn and_r_fatal<T2: 'static>(self, other: Parser<T2>) -> Parser<T2> {
        self.and_r(other.err_to_fatal())
    }

    /// ```text
    /// | self output   | more output   | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Ok t1         | Ok t2         | Ok [t1, ...]  |
    /// | Ok t1         | Err           | Ok [t1]       |
    /// | Ok ?          | Fatal         | Fatal         |
    /// | Err           | /             | Err           |
    /// | Fatal         | /             | Fatal         |
    /// ```
    pub fn and_more(self, more: Parser<T>) -> Parser<Vec<T>> {
        Parser::new(move |lex| {
            let (first, mut lex) = self.run(lex)?;
            let mut vec = vec![first];
            loop {
                match more.run(lex) {
                    Ok((t, new_lex)) => {
                        vec.push(t);
                        lex = new_lex;
                    },
                    Err(_) => break,
                    Fatal(err) => return Fatal(err),
                }
            }
            Ok((vec, lex))
        })
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok f(t1)      |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn map<U: 'static>(self, f: impl Fn(T) -> U + 'static) -> Parser<U> {
        self.map_with_lex(move |t, _| f(t))
        //Parser::new(move |lex| self.run(lex).map(|(t, lex)| (f(t), lex)))
    }

    /// Like [`Parser::map`] and then [`Parser::prefix_of`] on `self` and the
    /// resulting parser.
    ///
    /// ```text
    /// | self output   | f(t1) output  | output        |
    /// |:-------------:|:-------------:|:-------------:|
    /// | Ok t1         | Ok t2         | Ok t2         |
    /// | Ok t1         | Err           | Err           |
    /// | Ok t1         | Fatal         | Fatal         |
    /// | Err           | /             | Err           |
    /// | Fatal         | /             | Fatal         |
    /// ```
    pub fn map_and<U: 'static>(
        self,
        f: impl Fn(T, Lexer<'_>) -> PResult<'_, U> + 'static,
    ) -> Parser<U> {
        Parser::new(move |lex| self.run(lex).and_then(|(t, lex)| f(t, lex)))
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok (t1, lex)  | Ok f(t1, lex) |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn map_with_lex<U: 'static>(self, f: impl Fn(T, &Lexer<'_>) -> U + 'static) -> Parser<U> {
        Parser::new(move |lex| self.run(lex).map(|(t, lex)| (f(t, &lex), lex)))
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | f(t1)         |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn flat_map<U: 'static>(
        self,
        f: impl Fn(T, &Lexer<'_>) -> ResultWithFatal<U, ParseError> + 'static,
    ) -> Parser<U> {
        Parser::new(move |lex| self.run(lex).and_then(|(t, lex)| Ok((f(t, &lex)?, lex))))
    }

    pub fn not(self) -> Parser<()> {
        Parser::new(move |lex| match self.run(lex) {
            Ok((_, nlex)) => err!(NotWasFound, lex.span_to(nlex)),
            Err(_) => Ok(((), lex)),
            Fatal(err) => Fatal(err),
        })
    }

    /// like [`opt`] but trys to replace the [`Some`] output variant
    /// with the output of `p`
    pub fn opt_then<U: 'static>(self, p: Parser<U>) -> Parser<Option<U>> {
        Parser::new(move |lex| {
            let (opt, lex) = opt(self.clone()).run(lex)?;
            Ok(match opt {
                Some(_) => p.run(lex)?.map0(Some),
                None => (None, lex),
            })
        })
    }

    /// `self` 0 or more times
    ///
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok [...]      |
    /// | Err           | Ok []         |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn many0(self) -> Parser<Vec<T>> {
        self.many1().or(always!(vec![]))
        /*
        Parser::new(move |mut lex| {
            let mut vec = vec![];
            loop {
                match self.run(lex) {
                    Ok((t, new_lex)) => {
                        vec.push(t);
                        lex = new_lex;
                    },
                    Err(_) => break,
                    Fatal(err) => return Fatal(err),
                }
            }
            Ok((vec, lex))
        })
            */
    }

    /// `self` 1 or more times
    ///
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok [...]      |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn many1(self) -> Parser<Vec<T>> {
        Parser::new(move |lex| {
            let (first, mut lex) = self.run(lex)?;
            let mut vec = vec![first];
            loop {
                match self.run(lex) {
                    Ok((t, new_lex)) => {
                        vec.push(t);
                        lex = new_lex;
                    },
                    Err(_) => break,
                    Fatal(err) => return Fatal(err),
                }
            }
            Ok((vec, lex))
        })
    }

    /*
    pub fn many_until<U>(self, until: ParserStruct< U>) -> ParserStruct< (Vec<T>, U)> {
        move |mut lex| {
            let mut vec = vec![];
            let u = loop {
                match until(lex) {
                    Ok((u, new_lex)) => {
                        lex = new_lex;
                        break u;
                    },
                    Err(_) => {
                        let (t, new_lex) = self(lex)?;
                        lex = new_lex;
                        vec.push(t)
                    },
                }
            };
            Ok(((vec, u), lex))
        }
    }
    */

    /// `self` 1 or more times seperated by `sep`. Trailing `sep` not allowed
    /// (see [`Parser::sep_by1_trail`])
    ///
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok [...]      |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn sep_by1<Sep: 'static>(self, sep: Parser<Sep>) -> Parser<Vec<T>> {
        Parser::new(move |lex| {
            let val = self.clone().context("sep_by value");
            let sep = sep.clone().context("seperator");
            let (first, mut lex) = val.run(lex)?;
            let mut vec = vec![first];
            loop {
                match sep.clone().and_r(val.clone()).run(lex) {
                    Ok((t, new_lex)) => {
                        vec.push(t);
                        lex = new_lex;
                    },
                    Err(_) => break,
                    Fatal(err) => return Fatal(err),
                }
            }
            Ok((vec, lex))
        })
    }

    /// `self` 1 or more times seperated by `sep`. Trailing `sep` allowed
    /// (see [`Parser::sep_by1`])
    ///
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok [...]      |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn sep_by1_trail<Sep: 'static>(self, sep: Parser<Sep>) -> Parser<Vec<T>> {
        self.sep_by1(sep.clone()).and_l(opt(sep))
    }

    /// `self` 0 or more times seperated by `sep`. Trailing `sep` not allowed
    /// (see [`Parser::sep_by0_trail`])
    ///
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok [...]      |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn sep_by0<Sep: 'static>(self, sep: Parser<Sep>) -> Parser<Vec<T>> {
        self.sep_by1(sep).or(always!(vec![]))
    }

    /// `self` 0 or more times seperated by `sep`. Trailing `sep` allowed
    /// (see [`Parser::sep_by0`])
    ///
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok [...]      |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn sep_by0_trail<Sep: 'static>(self, sep: Parser<Sep>) -> Parser<Vec<T>> {
        self.sep_by0(sep.clone()).and_l(opt(sep))
    }

    /// ```text
    /// self sep self
    /// ---vvvvvv----
    ///    reduce sep self
    ///    ----vvvvvv-----
    ///        reduce sep self
    ///              ...
    /// ```
    ///
    /// at least one `self` is required
    pub fn sep_reduce1<Sep: 'static>(
        self,
        sep: Parser<Sep>,
        reduce: impl Fn(T, Sep, T) -> T + 'static,
    ) -> Parser<T> {
        Parser::new(move |lex| {
            let (mut acc, mut lex) = self.run(lex)?;
            let tail = sep.clone().and_fatal(self.clone());
            while let Ok(((sep, rhs), new_lex)) = tail.run(lex) {
                acc = reduce(acc, sep, rhs);
                lex = new_lex;
            }
            Ok((acc, lex))
        })
    }

    /// creates a [`Span`] for the entire match
    ///
    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok t1         | Ok (t1, span) |
    /// | Err           | Err           |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn spaned(self) -> Parser<(T, Span)> {
        Parser::new(move |lex| {
            let start = lex.get_pos();
            let (t, lex) = self.run(lex)?;
            let span = Span::new(start, lex.get_pos());
            Ok(((t, span), lex))
        })
    }

    pub fn to_spaned(self) -> Parser<Span> {
        self.spaned().map(|a| a.1)
    }

    pub fn to_expr(self) -> Parser<Expr>
    where Expr: From<(T, Span)> {
        self.spaned().map(Expr::from)
    }

    pub fn to_stmt(self) -> Parser<Stmt>
    where Stmt: From<(T, Span)> {
        self.spaned().map(Stmt::from)
    }

    /// parses any amount of whitespace or comment
    pub fn ws0(self) -> Parser<T> {
        self.and_l(ws0())
    }

    /// parses at least one whitespace or comment
    pub fn ws1(self) -> Parser<T> {
        self.and_l(ws1())
    }

    pub fn inspect(self, f: impl Fn(&PResult<'_, T>) + 'static) -> Parser<T> {
        Parser::new(move |lex| {
            let res = self.run(lex);
            f(&res);
            res
        })
    }

    /// ```text
    /// | self output   | output        |
    /// |:-------------:|:-------------:|
    /// | Ok            | Ok            |
    /// | Err           | Fatal         |
    /// | Fatal         | Fatal         |
    /// ```
    pub fn err_to_fatal(self) -> Parser<T> {
        Parser::new(move |lex| match self.run(lex) {
            Err(err) => Fatal(err),
            res => res,
        })
    }

    //pub fn context(self, context: impl fmt::Display + Send + Sync + 'static) ->
    // Self {
    pub fn context(self, context: impl fmt::Display + Send + Sync + Clone + 'static) -> Self {
        Parser::new(move |lex| match self.run(lex) {
            Ok(ok) => Ok(ok),
            Err(e) => Err(e.add_context(context.clone())),
            Fatal(e) => Fatal(e.add_context(context.clone())),
        })
    }
}

/// ```text
/// | self output   | output        |
/// |:-------------:|:-------------:|
/// | Ok t1         | Ok Some(t1)   |
/// | Err           | Ok None       |
/// | Fatal         | Fatal         |
/// ```
pub fn opt<T: 'static>(p: Parser<T>) -> Parser<Option<T>> {
    p.map(Some).or(always!(None))
}

pub fn f<T: 'static>(f: fn(lex: Lexer<'_>) -> PResult<'_, T>) -> Parser<T> {
    Parser::new(f)
}

/// calls the [`Parser`] `p` but doesn't advance the [`Lexer`].
pub fn peek<T: 'static>(p: Parser<T>) -> Parser<T> {
    Parser::new(move |lex| Ok((p.run(lex.clone())?.0, lex)))
}

pub fn choice<T: 'static, const N: usize>(parsers: [Parser<T>; N]) -> Parser<T> {
    Parser::new(move |lex| {
        let mut err = ParseError::new(PErrKind::NoInput, lex.pos_span());
        for p in parsers.iter() {
            match p.run(lex) {
                Err(e) => err = e,
                res => return res,
            }
        }
        Err(err)
    })
}

pub trait ParserChoice<T> {
    fn choice(&self) -> Parser<T>;
}

/*
impl<T, P: Parser<T> + Clone> ParserChoice<T> for [P] {
    fn choice(&self) -> Parser<T> {
        choice(self.try_into().unwrap())
    }
}
*/

impl ParseError {
    pub fn display(&self, code: &Code) -> String {
        let mut buf = String::new();
        buf.push_str(&format!("ERROR: {:?}\n", self));
        /*
        match self {
            ParseError::UnexpectedToken(Token { span, .. })
            | ParseError::DoubleLetMarker(Token { span, .. })
            | ParseError::TooManyLetIdents(span)
            | ParseError::Tmp(_, span) => {
                let (line_start, line) = code
                    .lines()
                    .try_fold(0, |idx, line| {
                        let end = idx + line.len() + 1;
                        if (idx..end).contains(&span.start) {
                            ControlFlow::Break((idx, line))
                        } else {
                            ControlFlow::Continue(end)
                        }
                    })
                    .break_value()
                    .unwrap();
                let err_start = span.start - line_start;
                let err_len = span.len();
                buf.push_str(&format!("| {}\n", line));
                buf.push_str(&format!("| {}", " ".repeat(err_start) + &"^".repeat(err_len)));
            },
            ParseError::NoInput
            | ParseError::NotAnIdent
            | ParseError::NotAnKeyword
            | ParseError::MissingLetIdent
            | ParseError::NotWasFound => (),
        }
        */

        let span = self.span;
        let (line_start, line) = code
            .lines()
            .try_fold(0, |idx, line| {
                let end = idx + line.len() + 1;
                if (idx..end).contains(&span.start) {
                    ControlFlow::Break((idx, line))
                } else {
                    ControlFlow::Continue(end)
                }
            })
            .break_value()
            .unwrap();
        buf.push_str(&format!("| {}\n", line));

        let err_start = span.start - line_start;
        let err_len = span.len();
        buf.push_str(&format!("| {} err span", " ".repeat(err_start) + &"^".repeat(err_len),));

        match self.kind {
            PErrKind::UnexpectedToken(Token { span, .. })
            | PErrKind::TooManyLetIdents(span)
            | PErrKind::Tmp(_, span) => {
                let err_start = span.start - line_start;
                let err_len = span.len();
                buf.push_str(&format!(
                    "\n| {} inner span",
                    " ".repeat(err_start) + &"^".repeat(err_len),
                ));
            },
            _ => (),
        }

        #[cfg(debug_assertions)]
        buf.push_str(&format!("\nCONTEXT: {:?}", self.context));

        buf
    }
}
