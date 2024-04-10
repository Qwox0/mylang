use self::lexer::TokenKind;
use crate::parser::lexer::{Lexer, Token};

mod lexer;
mod parser;

pub trait MutChain: Sized {
    #[inline]
    fn mut_chain(mut self, f: impl FnOnce(&mut Self)) -> Self {
        f(&mut self);
        self
    }
}

impl<T> MutChain for T {}

pub struct Batching<I, F> {
    f: F,
    iter: I,
}

impl<B, F, I> Iterator for Batching<I, F>
where
    I: Iterator,
    F: FnMut(&mut I) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (self.f)(&mut self.iter)
    }
}

pub trait IteratorBatching: Iterator + Sized {
    fn batching<B, F>(self, f: F) -> Batching<Self, F>
    where F: FnMut(&mut Self) -> Option<B> {
        Batching { f, iter: self }
    }
}

impl<I: Iterator> IteratorBatching for I {}

pub fn parse(code: &str) {
    let a = Lexer::new(code).filter(|t| {
        !matches!(
            t.kind,
            TokenKind::Whitespace | TokenKind::LineComment(_) | TokenKind::BlockComment(_)
        )
    });

    let mut full = String::new();
    let mut prev_was_ident = false;

    for Token { kind, span } in a {
        let text = code.get(span.bytes).expect("correctly parsed span");
        let is_ident = kind == TokenKind::Ident;
        if prev_was_ident && is_ident {
            full.push(' ');
        }
        full.push_str(&text);
        prev_was_ident = is_ident;
        let text = format!("{:?}", text);
        println!("{:<20} -> {:?}", text, kind);
    }

    println!("+++ full code:\n{}", full);
}
