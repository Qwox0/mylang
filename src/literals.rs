use crate::util::{UnwrapDebug, unreachable_debug};
use num::BigInt;
use std::num::ParseFloatError;

pub fn parse_int_lit(text: &str) -> Option<BigInt> {
    let text = text.replace("_", "");
    match text.get(..2) {
        Some("0b") => BigInt::parse_bytes(text[2..].as_bytes(), 2),
        Some("0o") => BigInt::parse_bytes(text[2..].as_bytes(), 8),
        Some("0x") => BigInt::parse_bytes(text[2..].as_bytes(), 16),
        _ => BigInt::parse_bytes(text.as_bytes(), 10),
    }
}

pub fn parse_float_lit(text: &str) -> Result<f64, ParseFloatError> {
    text.parse()
}

pub fn replace_escape_chars(s: &str) -> String {
    let mut buf = String::with_capacity(s.len());
    let mut bytes = s.bytes();
    while let Some(byte) = bytes.next() {
        if byte != b'\\' {
            buf.push(byte as char);
            continue;
        }
        buf.push(match bytes.next().u() {
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'\\' => '\\',
            b'0' => '\0',
            b'\'' => '\'',
            b'\"' => '\"',
            _ => unreachable_debug(),
        });
    }
    buf
}
