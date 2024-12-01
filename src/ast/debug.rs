use super::{Expr, ExprKind, ExprWithTy, Fn, Ident, UnaryOpKind};
use crate::{
    ast::VarDecl,
    ptr::Ptr,
    type_::Type,
    util::{self, unreachable_debug},
};
use std::fmt;

pub trait DebugAst {
    fn debug_impl(&self, buf: &mut impl DebugAstBuf);

    fn to_text(&self) -> String {
        let mut buf = DebugOneLine::default();
        self.debug_impl(&mut buf);
        buf.0
    }

    fn write_tree(&self) -> DebugTree {
        let mut lines = DebugTree::default();
        self.debug_impl(&mut lines);
        lines
    }

    fn print_tree(&self) {
        println!("| {}", self.to_text());
        for l in self.write_tree().lines {
            println!("| {}", l.0);
        }
    }
}

impl<T: DebugAst> DebugAst for Ptr<T> {
    #[inline]
    fn debug_impl(&self, buf: &mut impl DebugAstBuf) {
        self.as_ref().debug_impl(buf);
    }
}

impl DebugAst for Ident {
    #[inline]
    fn debug_impl(&self, buf: &mut impl DebugAstBuf) {
        buf.write(&self.text);
    }
}

impl DebugAst for Expr {
    fn debug_impl(&self, lines: &mut impl DebugAstBuf) {
        #[allow(unused_variables)]
        match &self.kind {
            ExprKind::Ident(text) | ExprKind::Literal { kind: _, code: text } => {
                lines.write(text.as_ref())
            },
            ExprKind::BoolLit(b) => lines.write(if *b { "true" } else { "false" }),
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
                lines.write_many(&elements, |e, _, l| l.write_tree(e), ",");
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
                lines.write_many_expr(elements, ",");
                lines.write(")");
            },
            ExprKind::Fn(Fn { params, ret_type, body }) => {
                let body = body;
                lines.write("(");
                lines.write_many_expr(&params, ",");
                lines.write(")->");
                if *ret_type != Type::Unset {
                    lines.write_tree(ret_type);
                }
                if matches!(body.kind, ExprKind::Block { .. }) {
                    body.debug_impl(lines);
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
            ExprKind::StructDef(fields) | ExprKind::UnionDef(fields) => {
                let is_struct = matches!(self.kind, ExprKind::StructDef(_));
                lines.write(if is_struct { "struct{" } else { "union{" });
                lines.write_many_expr(&fields, ",");
                lines.write("}");
            },
            ExprKind::EnumDef(variants) => {
                lines.write("enum{");
                lines.write_many(
                    &variants,
                    |v: &VarDecl, _, lines| {
                        lines.write_tree(&v.ident);
                        if v.ty != Type::Void {
                            lines.write("(");
                            lines.write_tree(&v.ty);
                            lines.write(")");
                        }
                    },
                    ",",
                );
                lines.write("}");
            },
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
            ExprKind::PositionalInitializer { lhs, args, .. } => {
                if let Some(lhs) = lhs {
                    lines.write_tree(lhs);
                }
                lines.write(".(");
                lines.write_many_expr(args, ",");
                lines.write(")");
            },
            ExprKind::NamedInitializer { lhs, fields, .. } => {
                if let Some(lhs) = lhs {
                    lines.write_tree(lhs);
                }
                lines.write(".{");
                lines.write_many(
                    &fields,
                    |(field, val): &(Ident, _), _, lines| {
                        lines.write_tree(field);
                        if let Some(val) = val {
                            lines.write("=");
                            lines.write_tree(val);
                        }
                    },
                    ",",
                );
                lines.write("}");
            },
            ExprKind::Dot { lhs, lhs_ty: _, rhs } => {
                if let Some(lhs) = lhs {
                    lines.write_tree(lhs);
                }
                lines.write(".");
                lines.write_tree(rhs);
            },
            ExprKind::Index { lhs, idx } => {
                lines.write_tree(lhs);
                lines.write("[");
                lines.write_tree(idx);
                lines.write("]");
            },
            ExprKind::Call { func, args, pipe_idx } => {
                if let Some(idx) = *pipe_idx {
                    lines.write_tree(&args[idx]);
                    lines.write("|");
                }
                lines.write_tree(func);
                lines.write("(");
                let inner = |arg, idx, lines| {
                    if pipe_idx.is_some_and(|i| i == idx) {
                        DebugAstBuf::write(lines, "^");
                    } else {
                        DebugAstBuf::write_tree(lines, arg);
                    }
                };
                lines.write_many(&args, inner, ",");
                lines.write(")");
            },
            ExprKind::UnaryOp { kind, expr, is_postfix: false } => {
                lines.write(match kind {
                    UnaryOpKind::AddrOf => "&",
                    UnaryOpKind::AddrMutOf => "&mut ",
                    UnaryOpKind::Deref => "*",
                    UnaryOpKind::Not => "!",
                    UnaryOpKind::Neg => "-",
                    UnaryOpKind::Try => unreachable_debug(),
                });
                lines.write_tree(expr);
            },
            ExprKind::UnaryOp { kind, expr, is_postfix: true } => {
                lines.write_tree(expr);
                lines.write(match kind {
                    UnaryOpKind::AddrOf => ".&",
                    UnaryOpKind::AddrMutOf => ".&mut ",
                    UnaryOpKind::Deref => ".*",
                    UnaryOpKind::Try => "?",
                    UnaryOpKind::Not | UnaryOpKind::Neg => unreachable_debug(),
                });
            },
            ExprKind::BinOp { lhs, op, rhs, .. } => {
                lines.write_tree(lhs);
                write!(lines, " {} ", op.to_binop_text()).unwrap();
                lines.write_tree(rhs);
            },
            ExprKind::Assign { lhs, rhs, .. } => {
                lines.write_tree(lhs);
                lines.write("=");
                lines.write_tree(rhs);
            },
            ExprKind::BinOpAssign { lhs, op, rhs, .. } => {
                lines.write_tree(lhs);
                lines.write(op.to_binop_assign_text());
                lines.write_tree(rhs);
            },
            ExprKind::VarDecl(decl) => decl.debug_impl(lines),
            ExprKind::If { condition, then_body, else_body, was_piped } => {
                if *was_piped {
                    lines.write_tree(condition);
                    lines.write("|if ");
                } else {
                    lines.write("if ");
                    lines.write_tree(condition);
                }
                lines.write(" ");
                lines.write_tree(then_body);
                if let Some(else_body) = else_body {
                    let else_body = else_body;
                    lines.write(" else ");
                    lines.write_tree(else_body);
                }
            },
            ExprKind::Match { val, else_body, was_piped } => todo!(),
            ExprKind::For { source, iter_var, body, was_piped } => {
                if !was_piped {
                    todo!()
                }
                // TODO: normal syntax
                lines.write_tree(source);
                lines.write("|for ");
                lines.write_tree(iter_var);
                lines.write(" ");
                lines.write_tree(body);
            },
            ExprKind::While { condition, body, was_piped } => {
                if *was_piped {
                    lines.write_tree(condition);
                    lines.write("|while");
                } else {
                    lines.write("while ");
                    lines.write_tree(condition);
                }
                lines.write(" ");
                lines.write_tree(body);
            },
            ExprKind::Catch { lhs } => todo!(),
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
            ExprKind::Break { expr } => {
                lines.write("break");
                if let Some(expr) = expr {
                    let expr = expr;
                    lines.write(" ");
                    lines.write_tree(expr);
                }
            },
            ExprKind::Continue => lines.write("continue"),
            ExprKind::Semicolon(expr) => {
                if let Some(expr) = expr {
                    lines.write_tree(expr);
                }
                lines.write(";");
            },
        }
    }
}

impl DebugAst for ExprWithTy {
    fn debug_impl(&self, buf: &mut impl DebugAstBuf) {
        self.expr.debug_impl(buf);
    }
}

impl DebugAst for VarDecl {
    fn debug_impl(&self, buf: &mut impl DebugAstBuf) {
        let VarDecl { markers, ident, ty, default, is_const } = self;
        buf.write(&format!(
            "{}{}{}",
            if markers.is_pub { "pub " } else { "" },
            mut_marker(markers.is_mut),
            if markers.is_rec { "rec " } else { "" }
        ));

        buf.write_tree(ident);

        if *ty != Type::Unset {
            buf.write(":");
            buf.write_tree(ty);
        }

        if let Some(default) = default {
            let default = default;
            buf.write(&format!(
                "{}{}",
                if *ty == Type::Unset { ":" } else { "" },
                if *is_const { ":" } else { "=" },
            ));
            buf.write_tree(default);
        }
    }
}

impl DebugAst for Type {
    fn debug_impl(&self, buf: &mut impl DebugAstBuf) {
        match self {
            Type::Void => buf.write("void"),
            Type::Never => buf.write("!"),
            Type::Ptr { pointee_ty } => {
                buf.write("*");
                pointee_ty.debug_impl(buf);
            },
            Type::Int { bits, is_signed } => {
                buf.write(if *is_signed { "i" } else { "u" });
                buf.write(&bits.to_string());
            },
            Type::IntLiteral => buf.write("int_lit"),
            Type::Bool => buf.write("bool"),
            Type::Float { bits } => {
                buf.write("f");
                buf.write(&bits.to_string());
            },
            Type::FloatLiteral => buf.write("float_lit"),
            Type::Function(_) => buf.write("fn"), // TODO: fn type as text
            Type::Array { len: count, elem_ty: ty } => {
                write!(buf, "[{count}]").unwrap();
                ty.debug_impl(buf);
            },
            Type::Struct { fields } => write!(buf, "struct{:?}", fields.0).unwrap(),
            Type::Union { fields } => write!(buf, "union{:?}", fields.0).unwrap(),
            Type::Enum { variants } => write!(buf, "enum{:?}", variants.0).unwrap(),
            Type::EnumVariant { enum_ty, idx } => {
                let Type::Enum { variants } = **enum_ty else { unreachable_debug() };
                let variant = variants[*idx];
                buf.write_tree(enum_ty);
                buf.write(".");
                buf.write_tree(&variant.ident);
            },
            Type::Type(_) => buf.write("type"),
            Type::Unset => {},
            Type::Unevaluated(expr) => expr.debug_impl(buf),
        }
    }
}

pub trait DebugAstBuf: fmt::Write {
    #[inline]
    fn write(&mut self, text: &str) {
        fmt::Write::write_str(self, text).unwrap()
    }

    fn write_tree<T: DebugAst>(&mut self, expr: &T);

    /// SAFETY: don't leak the `&mut Self` param out of the body of
    /// `single_write_tree`.
    fn write_many<'x, 'l, T>(
        &'l mut self,
        elements: &'x [T],
        mut single_write_tree: impl FnMut(&'x T, usize, &'l mut Self),
        sep: &str,
    ) {
        for (idx, x) in elements.iter().enumerate() {
            if idx != 0 {
                self.write(sep);
            }
            let lines = unsafe { util::forget_lifetime_mut(self) };
            single_write_tree(x, idx, lines);
        }
    }

    fn write_many_expr<'x, 'l, T: DebugAst>(&'l mut self, elements: &'x [T], sep: &str)
    where Self: Sized {
        self.write_many(elements, |t, _, buf| t.debug_impl(buf), sep);
    }
}

#[derive(Default)]
pub struct DebugOneLine(pub String);

impl fmt::Write for DebugOneLine {
    #[inline]
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0.write_str(text)
    }
}

impl DebugAstBuf for DebugOneLine {
    #[inline]
    fn write_tree<T: DebugAst>(&mut self, expr: &T) {
        expr.debug_impl(self);
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
pub struct DebugTree {
    pub lines: Vec<TreeLine>,
    cur_line: usize,
    cur_offset: usize,
}

impl fmt::Write for DebugTree {
    #[inline]
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let offset = self.cur_offset;
        self.get_cur_line().overwrite(offset, text);
        self.cur_offset += text.len();
        Ok(())
    }
}

impl DebugAstBuf for DebugTree {
    fn write_tree<T: DebugAst>(&mut self, expr: &T) {
        let state = (self.cur_line, self.cur_offset);

        self.cur_line += 1;
        expr.debug_impl(self);
        let len = self.cur_offset - state.1;

        self.cur_line = state.0;
        self.cur_offset = state.1;

        self.write(&"-".repeat(len))
    }
}

impl DebugTree {
    pub fn ensure_lines(&mut self, idx: usize) {
        while self.lines.get(idx).is_none() {
            self.lines.push(TreeLine(String::new()))
        }
    }

    pub fn get_cur_line(&mut self) -> &mut TreeLine {
        self.ensure_lines(self.cur_line);
        self.lines.get_mut(self.cur_line).unwrap()
    }
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
