use crate::tests::test_body;

#[test]
fn should_not_respect_outer_precedence() {
    #[derive(Debug)]
    struct Num(i32);

    impl std::ops::Add for Num {
        type Output = Num;

        fn add(self, rhs: Self) -> Self::Output {
            Num(self.0 + rhs.0)
        }
    }

    impl std::ops::Mul<!> for Num {
        type Output = Num;

        fn mul(self, _rhs: !) -> Self::Output {
            unreachable!()
        }
    }

    #[allow(unreachable_code)]
    fn f() -> Num {
        Num(2) * return Num(3) + Num(5)
    }

    assert_eq!(f().0, 3 + 5);

    test_body("2 * return 3 + 5").ok::<i32>(3 + 5);

    // TODO: break with value
    //test_body("while true { 1 + break 2 * 3 }").ok::<i32>(2 * 3);
}
