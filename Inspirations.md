# Inspirations List

This list contains language features/syntax and the programming languages that inspired or influenced the implementation of those features. Note that, if other languages (usually Jai) also have similar features/syntax, then I didn't know enough about those languages when I decided on implementing that feature.

* `struct`/separate data from methods: Rust
* `enum`/sum types/discriminated unions: Rust
* everything is an expression/no `;` == return value from block: Rust
* function definition syntax is anonymous: functional programming languages (`my_fn :: () -> { ... }`)
* struct definition syntax is anonymous: Zig (`A :: struct { ... }`)
* `:=`: Golang
* `::`: Jai
* `->`: Rust
* `mut`: Rust
* everything can be a postfix operator: Rust (`.await`; Note: Rust doesn't allow this to the same extend)
* `*T`, `[]T`, `[N]T`: Zig
* `.*`: Zig
* `\\ multiline string literal`: Zig
* `|>`: F#
    * First, I wanted to use `|` (Bash), which unfortunately conflicts with bitwise or (Note: Overloading `|` is not possible due to the difference in precedences).
* Named Initializer
    * Zig: `.{ ... }`
    * Rust: shorthand for `.{ abc = abc }`
* Positional Initializer
    * Jai: use field order (`.(1, 2)`)
* `xx` Autocast: Jai
* `!my_int` bitwise not: Rust

Unimplemented:

* `?`: Rust
* `?break`: Rust, Odin
* `?continue`: Rust, Odin
* `pub`, `pub(get)`, `pub(set)`, `pub(init)`: Rust, Swift
* `$T` Generics: Zig, Haskell, Jai
