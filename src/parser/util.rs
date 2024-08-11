pub trait Join {
    type Item: ?Sized;

    fn join<I: AsRef<Self::Item>>(sep: &str, it: impl IntoIterator<Item = I>) -> Self;
}

impl Join for String {
    type Item = str;

    fn join<I: AsRef<Self::Item>>(sep: &str, it: impl IntoIterator<Item = I>) -> Self {
        let mut it = it.into_iter();
        let Some(first) = it.next() else { return String::default() };
        let mut buf = first.as_ref().to_string();
        for i in it {
            buf.push_str(sep);
            buf.push_str(i.as_ref());
        }
        buf
    }
}
