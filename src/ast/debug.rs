use super::{Expr, ExprKind, Fn, Ident, PreOpKind};
use crate::{ast::VarDecl, error::SpannedError, ptr::Ptr, type_::Type};

pub trait DebugAst {
    fn to_text(&self) -> String;

    fn write_tree(&self, lines: &mut TreeLines);

    fn print_tree(&self) {
        let mut lines = TreeLines::default();
        self.write_tree(&mut lines);
        println!("| {}", self.to_text());
        for l in lines.lines {
            println!("| {}", l.0);
        }
    }
}

impl<T: DebugAst> DebugAst for Ptr<T> {
    fn to_text(&self) -> String {
        self.as_ref().to_text()
    }

    fn write_tree(&self, lines: &mut TreeLines) {
        self.as_ref().write_tree(lines)
    }
}

impl DebugAst for Expr {
    fn to_text(&self) -> String {
        #[allow(unused_variables)]
        match &self.kind {
            ExprKind::Ident(text) => text.to_string(),
            ExprKind::Literal { kind: _, code } => code.to_string(),
            ExprKind::BoolLit(b) => b.to_string(),
            ExprKind::ArrayTy { count, ty } => format!("[{}]{}", count.to_text(), ty.to_text()),
            ExprKind::ArrayTy2 { ty } => format!("[]{}", ty.to_text()),
            ExprKind::ArrayLit { elements } => {
                format!("[{}]", many_expr_to_text(elements, ","))
            },
            ExprKind::ArrayLitShort { val, count } => {
                format!("[{};{}]", val.to_text(), count.to_text())
            },
            ExprKind::Tuple { elements } => {
                format!("({})", many_expr_to_text(elements, ","))
            },
            ExprKind::Fn(Fn { params, ret_type, body }) => format!(
                "({})->{}{}",
                many_to_text(&*params, |decl| var_decl_to_text(decl), ","),
                ret_type.to_text(),
                if matches!(body.kind, ExprKind::Block { .. }) {
                    body.to_text()
                } else {
                    format!("{{{}}}", body.to_text())
                }
            ),
            ExprKind::Parenthesis { expr } => format!("({})", expr.to_text()),
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                format!(
                    "{{{}{}}}",
                    many_to_text(stmts, |a| a.to_text(), ";"),
                    if *has_trailing_semicolon { ";" } else { "" }
                )
            },
            ExprKind::StructDef(fields) => {
                format!("struct {{ {} }}", many_to_text(fields, |f| var_decl_to_text(f), ","))
            },
            ExprKind::UnionDef(..) => panic!(),
            ExprKind::EnumDef {} => panic!(),
            ExprKind::OptionShort(ty) => {
                format!("?{}", ty.to_text())
            },
            ExprKind::Ptr { is_mut, ty } => {
                format!("*{}{}", mut_marker(*is_mut), ty.to_text())
            },
            ExprKind::Initializer { lhs, fields } => {
                format!(
                    "{}.{{{}}}",
                    opt_expr_to_text(lhs),
                    many_to_text(
                        fields,
                        |(f, val)| format!(
                            "{}{}",
                            f.text.as_ref(),
                            opt_to_text(val, |v| format!("={}", v.to_text())),
                        ),
                        ","
                    )
                )
            },
            ExprKind::Dot { lhs, rhs } => {
                format!("{}.{}", lhs.to_text(), rhs.text.as_ref())
            },
            //ExprKind::Colon { lhs, rhs } => { format!("{}:{}", lhs.to_text(),
            // rhs.to_text()) },
            ExprKind::PostOp { kind, expr } => panic!(),
            ExprKind::Index { lhs, idx } => format!("{}[{}]", lhs.to_text(), idx.to_text()),
            //ExprKind::CompCall { func, args } => panic!(),
            ExprKind::Call { func, args } => {
                format!("{}({})", func.to_text(), many_to_text(args, |e| e.to_text(), ","))
            },
            ExprKind::PreOp { kind, expr, .. } => format!(
                "{}{}",
                match kind {
                    PreOpKind::AddrOf => "&",
                    PreOpKind::AddrMutOf => "&mut ",
                    PreOpKind::Deref => "*",
                    PreOpKind::Not => "!",
                    PreOpKind::Neg => "- ",
                },
                expr.to_text()
            ),
            ExprKind::BinOp { lhs, op, rhs, .. } => {
                format!("{} {} {}", lhs.to_text(), op.to_binop_text(), rhs.to_text())
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                format!("{}={}", lhs.to_text(), rhs.to_text())
            },
            ExprKind::BinOpAssign { lhs, op, rhs, .. } => {
                format!("{}{}{}", lhs.to_text(), op.to_binop_assign_text(), rhs.to_text())
            },
            ExprKind::VarDecl(decl) => var_decl_to_text(decl),
            ExprKind::If { condition, then_body, else_body } => {
                format!(
                    "if {} {}{}",
                    condition.to_text(),
                    then_body.to_text(),
                    opt_to_text(else_body, |e| format!(" else {}", e.to_text()))
                )
            },
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => {
                // TODO: normal syntax
                format!("{}|for {} {}", source.to_text(), &*iter_var.text, body.to_text())
            },
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Defer(expr) => format!("defer {}", expr.to_text()),
            ExprKind::Return { expr } => {
                format!("return{}", opt_to_text(expr, |e| format!(" {}", e.to_text())))
            },
            ExprKind::Semicolon(expr) => format!("{};", opt_expr_to_text(expr)),
        }
    }

    fn write_tree(&self, lines: &mut TreeLines) {
        #[allow(unused_variables)]
        match &self.kind {
            ExprKind::Ident(_) | ExprKind::Literal { .. } | ExprKind::BoolLit(_) => {
                lines.write(&self.to_text())
            },
            ExprKind::ArrayTy { count, ty } => {
                lines.write("[");
                lines.write_tree(count);
                lines.write("]");
                lines.write_tree(ty);
            },
            ExprKind::ArrayTy2 { ty } => {
                lines.write("[]");
                lines.write_tree(ty);
            },
            ExprKind::ArrayLit { elements } => {
                lines.write("[");
                for (idx, e) in elements.iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }
                    lines.write_tree(e);
                }
                lines.write("]");
            },
            ExprKind::ArrayLitShort { val, count } => {
                lines.write("[");
                lines.write_tree(val);
                lines.write(";");
                lines.write_tree(count);
                lines.write("]");
            },
            ExprKind::Tuple { elements } => {
                lines.write("(");
                many_expr_to_text(elements, ",");
                lines.write(")");
            },
            ExprKind::Fn(Fn { params, ret_type, body }) => {
                let body = body;
                lines.write("(");

                for (idx, decl) in params.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }

                    var_decl_write_tree(decl, lines)
                }
                lines.write(")->");
                if *ret_type != Type::Unset {
                    lines.write_tree(ret_type);
                }
                if matches!(body.kind, ExprKind::Block { .. }) {
                    body.write_tree(lines);
                } else {
                    lines.write("{");
                    lines.write_tree(body);
                    lines.write("}");
                }
            },
            ExprKind::Parenthesis { expr } => {
                lines.write("(");
                lines.write_tree(expr);
                lines.write(")");
            },
            ExprKind::Block { stmts, has_trailing_semicolon } => {
                lines.write("{");
                let len = stmts.len();
                for (idx, s) in stmts.iter().enumerate() {
                    lines.write_tree(s);
                    if idx + 1 < len || *has_trailing_semicolon {
                        lines.write(";");
                    }
                }
                lines.write("}");
            },
            ExprKind::StructDef(fields) => {
                lines.write("struct { ");
                for (idx, field) in fields.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }

                    var_decl_write_tree(field, lines)
                }
                lines.write(" }");
            },
            ExprKind::UnionDef(_) => todo!(),
            ExprKind::EnumDef {} => todo!(),
            ExprKind::OptionShort(ty) => {
                lines.write("?");
                lines.write_tree(ty);
            },
            ExprKind::Ptr { is_mut, ty } => {
                lines.write("*");
                if *is_mut {
                    lines.write("mut ");
                }
                lines.write_tree(ty);
            },
            ExprKind::Initializer { lhs, fields } => {
                if let Some(lhs) = lhs {
                    lines.write_tree(lhs);
                }
                lines.write(".{");
                for (idx, (field, val)) in fields.into_iter().enumerate() {
                    if idx != 0 {
                        lines.write(",");
                    }
                    lines.write_ident(field);
                    if let Some(val) = val {
                        lines.write("=");
                        lines.write_tree(val);
                    }
                }

                lines.write("}");
            },
            ExprKind::Dot { lhs, rhs } => {
                let lhs = lhs;
                lines.write_tree(lhs);
                lines.write(".");
                lines.write_ident(rhs);
            },
            ExprKind::PostOp { expr, kind } => todo!(),
            ExprKind::Index { lhs, idx } => {
                lines.write_tree(lhs);
                lines.write("[");
                lines.write_tree(idx);
                lines.write("]");
            },
            ExprKind::Call { func, args } => {
                let func = func;
                lines.write_tree(func);
                lines.write("(");
                let len = args.len();
                for (idx, arg) in args.into_iter().enumerate() {
                    let arg = arg;
                    lines.write_tree(arg);
                    if idx + 1 != len {
                        lines.write(",");
                    }
                }
                lines.write(")");
            },
            ExprKind::PreOp { kind, expr, .. } => {
                let expr = expr;
                lines.write(match kind {
                    PreOpKind::AddrOf => "&",
                    PreOpKind::AddrMutOf => "&mut ",
                    PreOpKind::Deref => "*",
                    PreOpKind::Not => "!",
                    PreOpKind::Neg => "- ",
                });
                lines.write_tree(expr);
            },
            ExprKind::BinOp { lhs, op, rhs, .. } => {
                let lhs = lhs;
                let rhs = rhs;
                lines.write_tree(lhs);
                lines.write(" ");
                lines.write(op.to_binop_text());
                lines.write(" ");
                lines.write_tree(rhs);
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                let lhs = lhs;
                let rhs = rhs;
                lines.write_tree(lhs);
                lines.write("=");
                lines.write_tree(rhs);
            },
            ExprKind::BinOpAssign { lhs, op, rhs, .. } => {
                let lhs = lhs;
                let rhs = rhs;
                lines.write_tree(lhs);
                lines.write(op.to_binop_assign_text());
                lines.write_tree(rhs);
            },
            ExprKind::VarDecl(decl) => var_decl_write_tree(decl, lines),
            ExprKind::If { condition, then_body, else_body } => {
                let condition = condition;
                let then_body = then_body;
                lines.write("if ");
                lines.write_tree(condition);
                lines.write(" ");
                lines.write_tree(then_body);
                if let Some(else_body) = else_body {
                    let else_body = else_body;
                    lines.write(" else ");
                    lines.write_tree(else_body);
                }
            },
            ExprKind::Match { val, else_body } => todo!(),
            ExprKind::For { source, iter_var, body } => {
                // TODO: normal syntax
                lines.write_tree(source);
                lines.write("|for ");
                lines.write_ident(iter_var);
                lines.write(" ");
                lines.write_tree(body);
            },
            ExprKind::While { condition, body } => todo!(),
            ExprKind::Catch { lhs } => todo!(),
            ExprKind::Pipe { lhs } => todo!(),
            ExprKind::Defer(expr) => {
                lines.write("defer ");
                lines.write_tree(expr);
            },
            ExprKind::Return { expr } => {
                lines.write("return");
                if let Some(expr) = expr {
                    let expr = expr;
                    lines.write(" ");
                    lines.write_tree(expr);
                }
            },
            ExprKind::Semicolon(expr) => {
                if let Some(expr) = expr {
                    lines.write_tree(expr);
                }
                lines.write(";");
            },
        }
    }
}

impl DebugAst for Type {
    fn to_text(&self) -> String {
        match self {
            Type::Void => "void".to_string(),
            Type::Never => "!".to_string(),
            Type::Ptr(pointee) => format!("*{}", pointee.to_text()),
            Type::Int { bits, is_signed } => {
                format!("{}{}", if *is_signed { "i" } else { "u" }, bits)
            },
            Type::IntLiteral => "int_lit".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Float { bits } => format!("f{}", bits),
            Type::FloatLiteral => "float_lit".to_string(),
            Type::Function(_) => "fn".to_string(), // TODO: fn type as text
            Type::Array { len: count, elem_ty: ty } => format!("[{count}]{}", ty.to_text()),
            Type::Struct { fields } => format!("struct{:?}", fields.0),
            Type::Union { .. } => todo!(),
            Type::Enum { .. } => todo!(),
            Type::Type(_) => "type".to_string(),
            Type::Unset => String::default(),
            Type::Unevaluated(expr) => expr.to_text(),
        }
    }

    fn write_tree(&self, lines: &mut TreeLines) {
        if let Type::Unevaluated(expr) = self {
            expr.write_tree(lines);
        } else {
            lines.write(&self.to_text())
        }
    }
}

#[derive(Default)]
pub struct TreeLine(pub String);

impl TreeLine {
    pub fn ensure_len(&mut self, len: usize) {
        let pad = " ".repeat(len.saturating_sub(self.0.len()));
        self.0.push_str(&pad);
    }

    pub fn overwrite(&mut self, offset: usize, text: &str) {
        self.ensure_len(offset + text.len());
        self.0.replace_range(offset..offset + text.len(), text);
    }
}

#[derive(Default)]
pub struct TreeLines {
    pub lines: Vec<TreeLine>,
    cur_line: usize,
    cur_offset: usize,
}

impl TreeLines {
    pub fn ensure_lines(&mut self, idx: usize) {
        while self.lines.get(idx).is_none() {
            self.lines.push(TreeLine(String::new()))
        }
    }

    pub fn get_cur(&mut self) -> &mut TreeLine {
        self.ensure_lines(self.cur_line);
        self.lines.get_mut(self.cur_line).unwrap()
    }

    pub fn get_cur_offset(&self) -> usize {
        self.cur_offset
    }

    pub fn write(&mut self, text: &str) {
        let offset = self.cur_offset;
        self.get_cur().overwrite(offset, text);
        self.cur_offset += text.len();
    }

    pub fn write_minus(&mut self, len: usize) {
        self.write(&"-".repeat(len))
    }

    pub fn scope_next_line<O>(&mut self, f: impl FnOnce(&mut Self) -> O) -> O {
        let state = (self.cur_line, self.cur_offset);
        self.next_line();
        let out = f(self);
        self.cur_line = state.0;
        self.cur_offset = state.1;
        out
    }

    pub fn write_tree<T: DebugAst>(&mut self, expr: &T) {
        self.scope_next_line(|l| expr.write_tree(l));
        self.write_minus(expr.to_text().len());
    }

    pub fn write_ident(&mut self, ident: &Ident) {
        self.scope_next_line(|l| l.write(&ident.text));
        self.write_minus(ident.text.len());
    }

    pub fn next_line(&mut self) {
        self.cur_line += 1;
    }

    pub fn prev_line(&mut self) {
        self.cur_line -= 1;
    }

    pub fn set_offset(&mut self, offset: usize) {
        self.cur_offset = offset;
    }
}

pub fn var_decl_to_text(VarDecl { markers, ident, ty, default, is_const }: &VarDecl) -> String {
    format!(
        "{}{}{}{}{}{}{}",
        if markers.is_pub { "pub " } else { "" },
        mut_marker(markers.is_mut),
        if markers.is_rec { "rec " } else { "" },
        ident.text.as_ref(),
        if *ty != Type::Unset { ":" } else { "" },
        ty.to_text(),
        default
            .map(|default| format!(
                "{}{}{}",
                if *ty == Type::Unset { ":" } else { "" },
                if *is_const { ":" } else { "=" },
                default.to_text()
            ))
            .unwrap_or_default()
    )
}

pub fn var_decl_write_tree(
    VarDecl { markers, ident, ty, default, is_const }: &VarDecl,
    lines: &mut TreeLines,
) {
    lines.write(&format!(
        "{}{}{}",
        if markers.is_pub { "pub " } else { "" },
        mut_marker(markers.is_mut),
        if markers.is_rec { "rec " } else { "" }
    ));

    lines.write_ident(ident);

    if *ty != Type::Unset {
        lines.write(":");
        lines.write_tree(ty);
    }

    if let Some(default) = default {
        let default = default;
        lines.write(&format!(
            "{}{}",
            if *ty == Type::Unset { ":" } else { "" },
            if *is_const { ":" } else { "=" },
        ));
        lines.write_tree(default);
    }
}

pub fn many_to_text<T>(
    elements: &[T],
    single_to_text: impl FnMut(&T) -> String,
    sep: &str,
) -> String {
    elements.iter().map(single_to_text).intersperse(sep.to_string()).collect()
}

pub fn many_expr_to_text<T: DebugAst>(elements: &Ptr<[T]>, sep: &str) -> String {
    many_to_text(elements, |t| t.to_text(), sep)
}

pub fn opt_to_text<T>(opt_expr: &Option<T>, inner: impl FnOnce(&T) -> String) -> String {
    opt_expr.as_ref().map(inner).unwrap_or_default()
}

pub fn opt_expr_to_text<T: DebugAst>(opt_expr: &Option<T>) -> String {
    opt_to_text(opt_expr, |t| t.to_text())
}

#[inline]
pub fn mut_marker(is_mut: bool) -> &'static str {
    if is_mut { "mut " } else { "" }
}
