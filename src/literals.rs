use core::num::ParseIntError;
use std::num::ParseFloatError;

pub fn parse_int_lit(text: &str) -> Result<i64, ParseIntError> {
    let text = text.replace("_", "");
    match text.get(..2) {
        Some("0b") => i64::from_str_radix(&text[2..], 2),
        Some("0o") => i64::from_str_radix(&text[2..], 8),
        Some("0x") => i64::from_str_radix(&text[2..], 16),
        _ => i64::from_str_radix(&text, 10),
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

pub fn parse_str_lit(text: &str) -> String {
    replace_escape_chars(&text)
}
