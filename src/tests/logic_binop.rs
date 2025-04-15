use crate::tests::jit_run_test;
use core::fmt;

#[test]
fn test_and() {
    for op in ["&&", "and"] {
        for a in [false, true] {
            for b in [false, true] {
                let code = format!("{a} {op} {b}");
                let out = *jit_run_test::<bool>(&code).ok();
                let expected = a && b;
                assert_eq!(out, expected);
            }
        }
    }
}

#[test]
fn test_or() {
    for op in ["||", "or"] {
        for a in [false, true] {
            for b in [false, true] {
                let code = format!("{a} {op} {b}");
                let out = *jit_run_test::<bool>(&code).ok();
                let expected = a || b;
                assert_eq!(out, expected);
            }
        }
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
            let out = *jit_run_test::<bool>(code).ok();
            let expected = match op {
                Op::And | Op::AndWord => {
                    let mut tmp = 0;
                    let _ = a && {
                        tmp = 1;
                        false
                    };
                    tmp == 0
                },
                Op::Or | Op::OrWord => {
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
    AndWord,
    Or,
    OrWord,
}

impl Op {
    pub const VARIANTS: [Op; 4] = [Op::And, Op::AndWord, Op::Or, Op::OrWord];
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::And => write!(f, "&&"),
            Op::AndWord => write!(f, "and"),
            Op::Or => write!(f, "||"),
            Op::OrWord => write!(f, "or"),
        }
    }
}
