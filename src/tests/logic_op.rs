use crate::{sema::SemaErrorKind, tests::jit_run_test, type_::Type};
use core::fmt;

#[test]
fn test_and() {
    for a in [false, true] {
        for b in [false, true] {
            let code = format!("{a} && {b}");
            let out = jit_run_test!(&code => bool).unwrap();
            let expected = a && b;
            assert!(out == expected, "{code} == {expected}")
        }
    }
}

#[test]
fn test_or() {
    for a in [false, true] {
        for b in [false, true] {
            let code = format!("{a} || {b}");
            let out = jit_run_test!(&code => bool).unwrap();
            let expected = a || b;
            println!("{code} == {expected}",);
            assert!(out == expected, "{code} == {expected}")
        }
    }
}

#[test]
fn test_type_missmatch() {
    for op in Op::VARIANTS {
        let out = jit_run_test!(&format!("true {op} {{}}") => bool).unwrap_err().sema();
        assert!(matches!(out.kind, SemaErrorKind::MismatchedTypesBinOp {
            lhs_ty: Type::Bool,
            rhs_ty: Type::Void
        }));
        let out = jit_run_test!(&format!("{{}} {op} true") => bool).unwrap_err().sema();
        assert!(matches!(out.kind, SemaErrorKind::MismatchedTypesBinOp {
            lhs_ty: Type::Void,
            rhs_ty: Type::Bool
        }));
    }
}

#[test]
fn test_short_circuit() {
    for op in Op::VARIANTS {
        for a in [false, true] {
            let code = format!(
                "{{
    mut tmp := 0;
    {a} {op} {{
        tmp = 1;
        false
    }};
    tmp == 0
}}"
            );
            let out = jit_run_test!(&code => bool).unwrap();
            let expected = match op {
                Op::And => {
                    let mut tmp = 0;
                    let _ = a && {
                        tmp = 1;
                        false
                    };
                    tmp == 0
                },
                Op::Or => {
                    let mut tmp = 0;
                    let _ = a || {
                        tmp = 1;
                        false
                    };
                    tmp == 0
                },
            };
            assert!(out == expected, "{a} {op} -> does short circuit? {expected} (got: {out})");
        }
    }
}

enum Op {
    And,
    Or,
}

impl Op {
    pub const VARIANTS: [Op; 2] = [Op::And, Op::Or];
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::And => write!(f, "&&"),
            Op::Or => write!(f, "||"),
        }
    }
}
