use super::jit_run_test_raw;

#[test]
#[ignore = "unfinished test"]
fn parse_currying() {
    jit_run_test_raw::<()>("test :: -> -> 1;").ok();
    jit_run_test_raw::<()>("test :: -> -> i32 { 1 };").ok();
}

/*
use crate::{
    ast::{DeclMarkers, Expr, ExprKind, VarDecl},
    parser::StmtIter,
};

#[test]
#[allow(unused_variables)]
fn no_args() {
    let alloc = bumpalo::Bump::new();
    let code = "my_fn :: -> 1;";
    let mut stmts = StmtIter::parse(code.as_ref(), &alloc);
    let f = stmts.next().expect("could parse function").expect("no parse error");
    assert!(stmts.next().is_none());

    let Expr { kind, span } = *f;
    let ExprKind::VarDecl(decl) = kind else { panic!("extected to parse VarDecl") };
    let VarDecl { markers, ident, ty, default, is_const } = decl;
    assert!(markers == DeclMarkers::default());
    assert!(*ident.text == *"my_fn");
    assert!(is_const);

    let init = default.expect("could parse function");

    println!("{:#?}", f);
    panic!()
}
*/
