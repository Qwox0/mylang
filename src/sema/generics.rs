use crate::{ast, ptr::Ptr, sema::accumulate_type_inner, type_::ty_match, util::UnwrapDebug};

pub fn generic_match(got: Ptr<ast::ConstVal>, expected: Ptr<ast::ConstVal>) -> bool {
    if let Some(expected) = expected.try_downcast_type() {
        let Some(got) = got.try_downcast_type() else { return false };
        return ty_match(got, expected);
    }
    if got.kind != expected.kind {
        return false;
    }
    // currently no coercion allowed
    match expected.matchable2() {
        ast::ConstValMatch::IntVal(expected) => {
            let got = got.downcast::<ast::IntVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::FloatVal(expected) => {
            let got = got.downcast::<ast::FloatVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::BoolVal(expected) => {
            let got = got.downcast::<ast::BoolVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::CharVal(expected) => {
            let got = got.downcast::<ast::CharVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::StrVal(expected) => {
            let got = got.downcast::<ast::StrVal>();
            got.text.as_ref() == expected.text.as_ref()
        },
        ast::ConstValMatch::RawPtrVal(expected) => {
            let got = got.downcast::<ast::RawPtrVal>();
            got.val == expected.val
        },
        ast::ConstValMatch::StaticPtrVal(expected) => {
            let got = got.downcast::<ast::StaticPtrVal>();
            got.sym == expected.sym
        },
        ast::ConstValMatch::EnumVal(expected) => {
            debug_assert!(expected.is_valid);
            let got = got.downcast::<ast::EnumVal>();
            if !got.is_valid {
                return false;
            }

            ty_match(got.enum_ty.upcast_to_type(), expected.enum_ty.upcast_to_type())
                && got.variant_idx == expected.variant_idx
                && got.data.zip(expected.data).map(|(g, e)| generic_match(g, e)).unwrap_or(true)
        },
        ast::ConstValMatch::OptionalVal(expected) => {
            let got = got.downcast::<ast::OptionalVal>();
            generic_match(got.val.u(), expected.val.u())
        },
        ast::ConstValMatch::AggregateVal(expected) => {
            let got = got.downcast::<ast::AggregateVal>();
            if !ty_match(got.ty.u(), expected.ty.u()) {
                return false;
            }
            debug_assert_eq!(got.elements.len(), expected.elements.len());
            got.elements
                .into_iter()
                .zip(expected.elements)
                .all(|(g, e)| generic_match(g, e))
        },
        _ => todo!(),
    }
}

pub fn accumulate_generic(generic: Ptr<ast::GenericDef>, next_val: Ptr<ast::ConstVal>) -> bool {
    let expr = None; // quiet
    match &mut generic.as_mut().cur_inst {
        cur_inst @ None => {
            *cur_inst = Some(next_val);
            true
        },
        Some(generic) => {
            if let Some(ty_acc) = generic.try_downcast_type_ref() {
                let Some(next_ty) = next_val.try_downcast_type() else { return false };
                accumulate_type_inner(ty_acc, next_ty, expr).is_ok()
            } else {
                generic_match(next_val, *generic)
            }
        },
    }
}
