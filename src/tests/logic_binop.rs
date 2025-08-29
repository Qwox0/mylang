use crate::tests::test_body;
use core::fmt;

#[test]
fn test_and() {
    for op in ["&&", "and"] {
        for a in [false, true] {
            for b in [false, true] {
                test_body(format!("{a} {op} {b}")).ok(a && b);
            }
        }
    }
}

#[test]
fn test_or() {
    for op in ["||", "or"] {
        for a in [false, true] {
            for b in [false, true] {
                test_body(format!("{a} {op} {b}")).ok(a || b);
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
            test_body(code).ok(expected);
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
