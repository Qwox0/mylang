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
    s.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
        .replace("\\0", "\0")
        .replace("\\'", "\'")
        .replace("\\\"", "\"")
}
