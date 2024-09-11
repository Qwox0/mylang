pub trait ParserInterface {
    type Item;
    type PeekedItem;

    fn next(&mut self) -> Option<Self::Item>;

    fn peek(&self) -> Option<Self::PeekedItem>;

    fn is_empty(&self) -> bool {
        self.peek().is_none()
    }

    #[inline]
    fn advance(&mut self) {
        self.next();
    }

    /// Advances the inner [`Chars`] [`Iterator`] while a condition is true.
    fn advance_while(&mut self, mut f: impl FnMut(Self::PeekedItem) -> bool) {
        while self.peek().is_some_and(&mut f) {
            self.advance();
        }
    }

    #[inline]
    fn next_if(&mut self, mut f: impl FnMut(Self::PeekedItem) -> bool) -> Option<Self::Item> {
        if self.peek().is_some_and(&mut f) { self.next() } else { None }
    }

    #[inline]
    fn advance_if(&mut self, f: impl FnMut(Self::PeekedItem) -> bool) -> bool {
        self.next_if(f).is_some()
    }
}
