use crate::{ast::ExprKind, parser::StmtIter, type_::Type};

macro_rules! test_helper {
    ($test_name:ident : $code:expr, $test_closure:expr) => {
        #[test]
        fn $test_name() {
            let alloc = bumpalo::Bump::new();
            let mut stmts = StmtIter::parse($code.as_ref(), &alloc);
            let expr = stmts.next().expect("could parse one expr").expect("no parse error");
            assert!(stmts.next().is_none(), "extected to parse only one expression");

            let t = $test_closure;
            (t)(expr.kind)
        }
    };
}

// Literals:

test_helper! { array_lit_empty: ".[]", |arr| {
    let ExprKind::ArrayInitializer { elements, .. } = arr else { panic!("not an array literal") };
    assert!(elements.is_empty());
}}

test_helper! { array_lit: ".[1, 2, 3, 4, 5]", |arr| {
    let ExprKind::ArrayInitializer { elements, .. } = arr else {
        panic!("not an array literal")
    };
    assert!(elements.len() == 5);
    for x in elements.iter() {
        assert!(matches!(x.kind, ExprKind::IntLit(_)));
    }
}}

test_helper! { array_lit_short: ".[0; 5]", |arr| {
    let ExprKind::ArrayInitializerShort { val, count, .. } = arr else {
        panic!("not an array literal")
    };
    assert!(matches!(val.kind, ExprKind::IntLit(_)));
    assert!(matches!(count.kind, ExprKind::IntLit(_)));
}}

test_helper! { array_lit_with_expr: ".[1 + 1, 2 + 2, 3 + 3]", |arr| {
    let ExprKind::ArrayInitializer { elements, .. } = arr else {
        panic!("not an array literal")
    };
    assert!(elements.len() == 3);
    for x in elements.iter() {
        assert!(matches!(x.kind, ExprKind::BinOp { .. }));
    }
}}

test_helper! { array_lit_short_with_expr: ".[1 + 1; 2 + 2]", |arr| {
    let ExprKind::ArrayInitializerShort { val, count, .. } = arr else {
        panic!("not an array literal")
    };
    assert!(matches!(val.kind, ExprKind::BinOp { .. }));
    assert!(matches!(count.kind, ExprKind::BinOp { .. }));
}}

// Types:

test_helper! { array_ty_f64: "[5]f64", |arr| {
    let ExprKind::ArrayTy { count, ty } = arr else {
        panic!("not an array type")
    };
    assert!(matches!(count.kind, ExprKind::IntLit(_)));
    if let Type::Unevaluated(expr) = ty && let ExprKind::Ident(ty) = expr.kind {
        assert!(*ty == *"f64");
    } else {
        panic!("parsed type incorrectly")
    }
}}

test_helper! { array_ty_f64_expr_count: "[1 + 1]f64", |arr| {
    let ExprKind::ArrayTy { count, ty } = arr else {
        panic!("not an array type")
    };
    assert!(matches!(count.kind, ExprKind::BinOp { .. }));
    if let Type::Unevaluated(expr) = ty && let ExprKind::Ident(ty) = expr.kind {
        assert!(*ty == *"f64");
    } else {
        panic!("parsed type incorrectly")
    }
}}

test_helper! { slice_ty_f64: "[]f64", |arr| {
    let ExprKind::SliceTy { ty, is_mut: false } = arr else {
        panic!("not a slice type")
    };
    if let Type::Unevaluated(expr) = ty && let ExprKind::Ident(ty) = expr.kind {
        assert!(*ty == *"f64");
    } else {
        panic!("parsed type incorrectly")
    }
}}
