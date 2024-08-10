#![allow(unused)]

use crate::parser::result_with_fatal::ResultWithFatal;
use std::{
    convert::Infallible,
    ops::{ControlFlow, FromResidual},
};

pub trait OptionExt<T>: Sized {
    fn ok_or_nonfatal<E>(self, err: E) -> ResultWithFatal<T, E>;
    fn ok_or_fatal<E>(self, err: E) -> ResultWithFatal<T, E>;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_nonfatal<E>(self, err: E) -> ResultWithFatal<T, E> {
        self.map(ResultWithFatal::Ok).unwrap_or(ResultWithFatal::Err(err))
    }

    fn ok_or_fatal<E>(self, err: E) -> ResultWithFatal<T, E> {
        self.map(ResultWithFatal::Ok).unwrap_or(ResultWithFatal::Fatal(err))
    }
}

pub trait Join {
    type Item: ?Sized;

    fn join<I: AsRef<Self::Item>>(sep: &str, it: impl IntoIterator<Item = I>) -> Self;
}

impl Join for String {
    type Item = str;

    fn join<I: AsRef<Self::Item>>(sep: &str, it: impl IntoIterator<Item = I>) -> Self {
        let mut it = it.into_iter();
        let Some(first) = it.next() else { return String::default() };
        let mut buf = first.as_ref().to_string();
        for i in it {
            buf.push_str(sep);
            buf.push_str(i.as_ref());
        }
        buf
    }
}
