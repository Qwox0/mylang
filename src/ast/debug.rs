use super::{DeclMarkers, HasAstKind, OptionTypeExt};
use crate::{
    ast::{self, Ast, AstEnum, AstKind, TypeEnum, UnaryOpKind, UpcastToAst},
    context::primitives,
    parser::lexer::Span,
    ptr::Ptr,
    util::{self, UnwrapDebug, unreachable_debug},
};
use std::fmt::{self, Debug, Display};

pub trait DebugAst {
    fn debug_impl(&self, buf: &mut impl DebugAstBuf);

    fn to_text(&self, is_type: bool) -> String {
        let mut buf = DebugOneLine::new(is_type);
        self.debug_impl(&mut buf);
        buf.line
    }

    fn write_tree(&self) -> DebugTree {
        let mut lines = DebugTree::default();
        self.debug_impl(&mut lines);
        lines
    }

    fn print_tree(&self) {
        eprintln!("| {}", self.to_text(false));
        for l in self.write_tree().lines {
            eprintln!("| {}", l.0);
        }
    }
}

impl<T: DebugAst> DebugAst for Ptr<T> {
    #[inline]
    fn debug_impl(&self, buf: &mut impl DebugAstBuf) {
        self.as_ref().debug_impl(buf);
    }
}

impl DebugAst for Ast {
    #[inline]
    fn debug_impl(&self, lines: &mut impl DebugAstBuf) {
        let mut ptr = Ptr::from_ref(self);
        let ty = ptr.ty;
        if ptr.replacement.is_some() {
            let rep = ptr.rep();
            if rep.span != Span::ZERO {
                ptr = rep;
            }
        }

        let parenthesis_count = ptr.parenthesis_count;
        for _ in 0..parenthesis_count {
            lines.write("(");
        }
        match ptr.matchable().as_ref() {
            AstEnum::Ident { text, .. } => lines.write(text.as_ref()),
            AstEnum::Block { stmts, has_trailing_semicolon, .. } => {
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
            AstEnum::PositionalInitializer { lhs, args, .. } => {
                lines.write_opt_tree(lhs.as_ref());
                lines.write(".(");
                lines.write_many_expr(args, ",");
                lines.write(")");
            },
            AstEnum::NamedInitializer { lhs, fields, .. } => {
                lines.write_opt_tree(lhs.as_ref());
                lines.write(".{");
                lines.write_many(
                    fields.iter(),
                    |(field, val), _, lines| {
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
            AstEnum::ArrayInitializer { lhs, elements, .. } => {
                lines.write_opt_tree(lhs.as_ref());
                lines.write(".[");
                lines.write_many(elements.iter(), |e, _, l| l.write_tree(e), ",");
                lines.write("]");
            },
            AstEnum::ArrayInitializerShort { lhs, val, count, .. } => {
                lines.write_opt_tree(lhs.as_ref());
                lines.write(".[");
                lines.write_tree(val);
                lines.write(";");
                lines.write_tree(count);
                lines.write("]");
            },
            AstEnum::Dot { lhs, rhs, .. } => {
                lines.write_opt_tree(lhs.as_ref());
                lines.write(".");
                lines.write_tree(rhs);
            },
            AstEnum::Index { mut_access, lhs, idx, .. } => {
                lines.write_tree(lhs);
                lines.write("[");
                lines.write_tree(idx);
                lines.write("]");
                if *mut_access {
                    lines.write("mut");
                }
            },
            AstEnum::Cast { operand, target_ty, .. } => {
                lines.write_tree(operand);
                lines.write(".as(");
                lines.write_tree(target_ty);
                lines.write(")");
            },
            AstEnum::Autocast { operand, .. } => {
                lines.write("xx ");
                lines.write_tree(operand);
            },
            AstEnum::Call { func, args, pipe_idx, .. } => {
                if let Some(idx) = *pipe_idx {
                    lines.write_tree(&args[idx]);
                    lines.write("|>");
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
                lines.write_many(args.iter(), inner, ",");
                lines.write(")");
            },
            AstEnum::UnaryOp { op, operand, is_postfix: false, .. } => {
                lines.write(match op {
                    UnaryOpKind::AddrOf => "&",
                    UnaryOpKind::AddrMutOf => "&mut ",
                    UnaryOpKind::Deref => "*",
                    UnaryOpKind::Not => "!",
                    UnaryOpKind::Neg => "-",
                    UnaryOpKind::Try => unreachable_debug(),
                });
                lines.write_tree(operand);
            },
            AstEnum::UnaryOp { op, operand, is_postfix: true, .. } => {
                lines.write_tree(operand);
                lines.write(match op {
                    UnaryOpKind::AddrOf => ".&",
                    UnaryOpKind::AddrMutOf => ".&mut",
                    UnaryOpKind::Deref => ".*",
                    UnaryOpKind::Try => "?",
                    UnaryOpKind::Not | UnaryOpKind::Neg => unreachable_debug(),
                });
            },
            AstEnum::BinOp { lhs, op, rhs, .. } => {
                lines.write_tree(lhs);
                write!(lines, " {} ", op.to_binop_text()).unwrap();
                lines.write_tree(rhs);
            },
            AstEnum::Range { start, end, is_inclusive, .. } => {
                lines.write_opt_tree(start.as_ref());
                lines.write(if *is_inclusive { "..=" } else { ".." });
                lines.write_opt_tree(end.as_ref());
            },
            AstEnum::Assign { lhs, rhs, .. } => {
                lines.write_tree(lhs);
                lines.write("=");
                lines.write_tree(rhs);
            },
            AstEnum::BinOpAssign { lhs, op, rhs, .. } => {
                lines.write_tree(lhs);
                lines.write(op.to_binop_assign_text());
                lines.write_tree(rhs);
            },
            AstEnum::Decl {
                is_extern: false,
                markers,
                ident,
                on_type,
                var_ty,
                var_ty_expr,
                init,
                is_const,
                ..
            } => {
                lines.write(&format!(
                    "{}{}{}{}",
                    if markers.get(DeclMarkers::IS_PUB_MASK) { "pub " } else { "" },
                    if markers.get(DeclMarkers::IS_MUT_MASK) { "mut " } else { "" },
                    if markers.get(DeclMarkers::IS_REC_MASK) { "rec " } else { "" },
                    if markers.get(DeclMarkers::IS_STATIC_MASK) { "static " } else { "" },
                ));

                if let Some(ty_expr) = on_type {
                    lines.write_tree(ty_expr);
                    lines.write(".");
                }
                lines.write_tree(ident);

                if let Some(var_ty) = var_ty_expr {
                    lines.write(":");
                    lines.write_tree(var_ty);
                } else if let Some(var_ty) = var_ty {
                    lines.write(":");
                    lines.write_tree(var_ty);
                }

                if let Some(init) = init {
                    lines.write(&format!(
                        "{}{}",
                        if var_ty_expr.is_none() { ":" } else { "" },
                        if *is_const { ":" } else { "=" },
                    ));
                    lines.write_tree(init);
                }
            },
            AstEnum::Decl { is_extern: true, ident, var_ty, var_ty_expr, .. } => {
                lines.write("extern ");
                lines.write(&ident.text);
                lines.write(":");
                if let Some(var_ty) = var_ty {
                    lines.write_tree(var_ty);
                } else {
                    lines.write_tree(&var_ty_expr.u());
                }
            },
            AstEnum::If { condition, then_body, else_body, was_piped, .. } => {
                if *was_piped {
                    lines.write_tree(condition);
                    lines.write("|>if ");
                } else {
                    lines.write("if ");
                    lines.write_tree(condition);
                }
                lines.write(" ");
                lines.write_tree(then_body);
                if let Some(else_body) = else_body {
                    lines.write(" else ");
                    lines.write_tree(else_body);
                }
            },
            AstEnum::Match { .. } => todo!(),
            AstEnum::For { source, iter_var, body, was_piped, .. } => {
                if *was_piped {
                    lines.write_tree(source);
                    lines.write("|>for ");
                    lines.write_tree(iter_var);
                } else {
                    lines.write("for ");
                    lines.write_tree(iter_var);
                    lines.write(" in ");
                    lines.write_tree(source);
                }
                lines.write(" ");
                lines.write_tree(body);
            },
            AstEnum::While { condition, body, was_piped, .. } => {
                if *was_piped {
                    lines.write_tree(condition);
                    lines.write("|>while");
                } else {
                    lines.write("while ");
                    lines.write_tree(condition);
                }
                lines.write(" ");
                lines.write_tree(body);
            },
            // AstEnum::Catch { .. } => todo!(),
            AstEnum::Defer { stmt, .. } => {
                lines.write("defer ");
                lines.write_tree(stmt);
            },
            AstEnum::Return { val, .. } => {
                lines.write("return");
                if let Some(val) = val {
                    lines.write(" ");
                    lines.write_tree(val);
                }
            },
            AstEnum::Break { val, .. } => {
                lines.write("break");
                if let Some(val) = val {
                    lines.write(" ");
                    lines.write_tree(val);
                }
            },
            AstEnum::Continue { .. } => lines.write("continue"),
            AstEnum::Empty { .. } => {},
            AstEnum::ImportDirective { path, .. } => {
                lines.write("#import ");
                lines.write_tree(path);
            },
            AstEnum::ProgramMainDirective { span, .. } | AstEnum::SimpleDirective { span, .. } => {
                lines.write(span.get_text().as_ref());
            },

            AstEnum::IntVal { val, .. } => lines.write(&val.to_string()),
            AstEnum::FloatVal { val, .. } => lines.write(&val.to_string()),
            AstEnum::BoolVal { val, .. } => lines.write(if *val { "true" } else { "false" }),
            AstEnum::CharVal { val, .. } => lines.write(&format!("'{}'", *val)),
            AstEnum::StrVal { text, .. } => lines.write(&format!("\"{}\"", text.as_ref())),
            AstEnum::PtrVal { val, .. } => lines.write(&format!("{:p}", *val as *const ())),
            AstEnum::AggregateVal { elements, .. } => match ty.matchable().as_ref() {
                TypeEnum::ArrayTy { elem_ty, .. } => {
                    lines.write_tree(elem_ty);
                    lines.write(".[");
                    lines.write_many_expr(elements, ", ");
                    lines.write("]");
                },
                TypeEnum::StructDef { fields, .. } => {
                    debug_assert_eq!(fields.len(), elements.len());
                    lines.write(".{");
                    lines.write_many(
                        fields.iter().zip(elements.iter()),
                        |(f, e), _, lines| {
                            lines.write_fmt(format_args!("{}={e}", f.ident.text.as_ref())).unwrap();
                        },
                        ", ",
                    );
                    lines.write("}");
                },
                cv => todo!("debug {cv:?}"),
            },
            AstEnum::Fn { params, ret_ty, ret_ty_expr, body, .. } => {
                lines.write("(");
                lines.write_many_expr(params, ",");
                lines.write(")->");
                if let Some(ret_type) = ret_ty_expr {
                    lines.write_tree(ret_type);
                } else if let Some(ret_ty) = ret_ty {
                    lines.write_tree(ret_ty);
                }
                if lines.write_fn_as_type() {
                    return;
                }
                let Some(body) = body else { return };
                if body.kind == AstKind::Block {
                    lines.write_tree(body);
                } else {
                    lines.write("{");
                    lines.write_tree(body);
                    lines.write("}");
                }
            },

            AstEnum::SimpleTy { decl, .. } => lines.write(&decl.ident.text),
            AstEnum::IntTy { bits, is_signed, .. } => {
                lines.write(if *is_signed { "i" } else { "u" });
                lines.write(&bits.to_string());
            },
            AstEnum::FloatTy { bits, .. } => {
                lines.write("f");
                lines.write(&bits.to_string());
            },
            AstEnum::PtrTy { pointee, is_mut, .. } => {
                lines.write(if *is_mut { "*mut " } else { "*" });
                lines.write_tree(pointee);
            },
            AstEnum::SliceTy { elem_ty, is_mut, .. } => {
                lines.write(if *is_mut { "[]mut " } else { "[]" });
                lines.write_tree(elem_ty);
            },
            AstEnum::ArrayTy { len, elem_ty, .. } => {
                lines.write("[");
                lines.write_tree(len);
                lines.write("]");
                lines.write_tree(elem_ty);
            },
            //AstEnum::FunctionTy { func, .. } => debug_fn(func.params, func.ret_type, None, lines),
            AstEnum::StructDef { fields, .. } | AstEnum::UnionDef { fields, .. } => {
                lines.write(if ptr.kind == AstKind::StructDef { "struct{" } else { "union{" });
                lines.write_many(
                    fields.iter(),
                    |f, _, b| {
                        b.write_tree(&f.ident);
                        b.write(":");
                        if let Some(t) = f.var_ty {
                            b.write_tree(&t);
                        } else {
                            b.write("?");
                        }
                    },
                    ",",
                );
                lines.write("}");
            },
            AstEnum::EnumDef { variants, .. } => {
                lines.write("enum{");
                lines.write_many(
                    variants.iter(),
                    |v, _, lines| {
                        lines.write_tree(&v.ident);
                        if let Some(ty_expr) = v.var_ty.filter(|t| *t != primitives().void_ty) {
                            lines.write("(");
                            lines.write_tree(&ty_expr);
                            lines.write(")");
                        }
                    },
                    ",",
                );
                lines.write("}");
            },
            AstEnum::RangeTy { rkind: kind, elem_ty, .. } => {
                lines.write(kind.type_name());
                lines.write("<");
                lines.write_tree(elem_ty);
                lines.write(">");
            },
            AstEnum::OptionTy { inner_ty: ty, .. } => {
                lines.write("?");
                lines.write_tree(ty);
            },
        }
        for _ in 0..parenthesis_count {
            lines.write(")");
        }
    }
}

impl<V> DebugAst for V
where Ptr<V>: UpcastToAst
{
    #[inline]
    fn debug_impl(&self, buf: &mut impl DebugAstBuf) {
        Ptr::from_ref(self).upcast().debug_impl(buf)
    }
}

pub trait DebugAstBuf: fmt::Write {
    #[inline]
    fn write(&mut self, text: &str) {
        fmt::Write::write_str(self, text).unwrap()
    }

    fn write_tree<T: DebugAst>(&mut self, expr: &T);

    fn write_fn_as_type(&self) -> bool;

    /// SAFETY: don't leak the `&mut Self` param out of the body of
    /// `single_write_tree`.
    fn write_many<'l, T>(
        &'l mut self,
        elements: impl IntoIterator<Item = T>,
        mut single_write_tree: impl FnMut(T, usize, &'l mut Self),
        sep: &str,
    ) {
        for (idx, x) in elements.into_iter().enumerate() {
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

    fn write_opt_tree<'x, 'l, T: DebugAst>(&mut self, opt_expr: Option<&T>) {
        if let Some(t) = opt_expr {
            self.write_tree(t);
        }
    }
}

pub struct DebugOneLine {
    pub line: String,
    write_fn_as_type: bool,
}

impl DebugOneLine {
    pub fn new(write_fn_as_type: bool) -> Self {
        Self { line: String::new(), write_fn_as_type }
    }
}

impl fmt::Write for DebugOneLine {
    #[inline]
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.line.write_str(text)
    }
}

impl DebugAstBuf for DebugOneLine {
    #[inline]
    fn write_tree<T: DebugAst>(&mut self, expr: &T) {
        expr.debug_impl(self);
    }

    fn write_fn_as_type(&self) -> bool {
        self.write_fn_as_type
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

    fn write_fn_as_type(&self) -> bool {
        false
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

impl Debug for Ast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr = Ptr::from_ref(self);
        if ptr.has_type_kind() {
            Debug::fmt(ptr.cast::<ast::Type>().as_ref(), f)
        } else if ptr.get_kind().is_const_val_kind() {
            Debug::fmt(ptr.cast::<ast::ConstVal>().as_ref(), f)
        } else {
            self.matchable().as_ref().fmt(f)
        }
    }
}

impl Debug for ast::ConstVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr = Ptr::from_ref(self);
        if ptr.upcast().has_type_kind() {
            Debug::fmt(ptr.cast::<ast::Type>().as_ref(), f)
        } else {
            ptr.matchable().as_ref().fmt(f)
        }
    }
}

impl Display for ast::ConstVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text(false))?;
        match self.ty {
            Some(ty) if self.kind != AstKind::AggregateVal => write!(f, "{ty}"),
            Some(_) => Ok(()),
            None => write!(f, "untyped"),
        }
    }
}

const DEBUG_TYPES: bool = true;
const DEBUG_SIMPLE_TYPES: bool = false;

impl Debug for ast::Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ptr = Ptr::from_ref(self);
        if !DEBUG_TYPES {
            write!(f, "{}", self)
        } else if let Some(sty) = ptr.try_downcast::<ast::SimpleTy>() {
            if !DEBUG_SIMPLE_TYPES {
                write!(f, "{}", self)
            } else if ptr == ptr.ty || ptr.upcast() == sty.decl.init || ptr == sty.decl.ty {
                write!(f, "{} (recursive {:x?})", self, self.matchable().as_ref())
            } else {
                debug_assert_ne!(ptr, sty.decl.ident.ty);
                Debug::fmt(self.matchable().as_ref(), f)
            }
        } else if ptr.kind == AstKind::Fn {
            let tmp = ptr.as_mut().ty.take();
            Debug::fmt(self.matchable().as_ref(), f)?;
            ptr.as_mut().ty = tmp;
            Ok(())
        } else {
            Debug::fmt(self.matchable().as_ref(), f)
        }
    }
}

impl Display for ast::Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text(true))
    }
}
