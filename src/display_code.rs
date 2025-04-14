use crate::{diagnostics::COLOR_UNSET, parser::lexer::Span, util::resolve_file_loc};
use std::io::{self, stderr};

pub fn display<'l>(span: Span) -> DisplayCodeBuilder<'l> {
    DisplayCodeBuilder { span, label: "", color_code: "" }
}

#[must_use]
pub struct DisplayCodeBuilder<'l> {
    span: Span,
    label: &'l str,
    color_code: &'static str,
}

impl<'l> DisplayCodeBuilder<'l> {
    #[must_use]
    pub fn label(&mut self, label: &'l str) -> &mut DisplayCodeBuilder<'l> {
        self.label = label;
        self
    }

    #[must_use]
    pub fn color_code(&mut self, color_code: &'static str) -> &mut DisplayCodeBuilder<'l> {
        self.color_code = color_code;
        self
    }

    pub fn finish_to(&mut self, w: &mut impl io::Write) -> Result<(), io::Error> {
        let DisplayCodeBuilder { span, label, color_code } = self;
        // TODO: span behind end of code and code ends in '\n'
        let file = span.file.expect("span has file");
        let code = file.code.as_ref();
        let start_offset = code
            .get(..span.start)
            .unwrap_or("")
            .bytes()
            .rev()
            .position(|b| b == b'\n')
            .unwrap_or(span.start);
        let loc = resolve_file_loc(span.start, code);
        let line_num = loc.line.to_string();
        let line_num_padding = " ".repeat(line_num.len());
        let end_offset = code
            .get(span.end.saturating_sub(1)..)
            .and_then(|l| l.lines().next().map(str::len))
            .unwrap_or(0);
        let line = code
            .get(span.start - start_offset..(span.end + end_offset).min(code.len()))
            .unwrap()
            .lines()
            .intersperse("\\n")
            .collect::<String>();
        let linebreaks_in_span = code
            .get(span.start..span.end)
            .map(|s| s.lines().count().saturating_sub(1))
            .unwrap_or(0);
        let marker_len = span.len().saturating_add(linebreaks_in_span);
        let offset = " ".repeat(start_offset);
        // annoying problem with writeln: <https://github.com/rust-lang/rust/issues/90785>
        writeln!(
            w,
            "{line_num_padding}--> {}:{line_num}:{}", // ─┬─
            file.path.as_ref().display(),
            start_offset + 1
        )?;
        writeln!(w, "{} │", line_num_padding)?;
        writeln!(w, "{line_num} │ {}", line)?;
        writeln!(
            w,
            "{line_num_padding} │ {offset}{color_code}{} {label}{COLOR_UNSET}",
            "^".repeat(marker_len)
        )
    }

    pub fn finish(&mut self) {
        self.finish_to(&mut stderr()).unwrap();
    }

    pub fn finish_to_string(&mut self) -> String {
        let mut buf = Vec::new();
        self.finish_to(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        diagnostics::{COLOR_RED, COLOR_UNSET},
        display_code::display,
        parser::lexer::Span,
        ptr::Ptr,
        source_file::SourceFile,
        tests::test_file_mock,
    };
    use std::ops::Range;

    fn span(range: Range<usize>, file: &SourceFile) -> Span {
        Span::new(range, Some(Ptr::from_ref(file)))
    }

    fn word_span(word: &str, file: &SourceFile) -> Span {
        let word_pos = file.code.find(word).unwrap();
        span(word_pos..word_pos + word.len(), &file)
    }

    #[test]
    fn entire_line() {
        let file = test_file_mock("this is the first line\nthis is the second line".as_ref());
        let span = word_span("this is the first line", &file);
        let expected = format!(
            " --> test.mylang:1:1
  │
1 │ this is the first line
  │ ^^^^^^^^^^^^^^^^^^^^^^ {COLOR_UNSET}
"
        );
        assert_eq!(display(span).finish_to_string(), expected);
    }

    #[test]
    fn after_line_dont_show_newline() {
        let file = test_file_mock("this is the first line\nthis is the second line".as_ref());
        let span = word_span("\n", &file);
        let expected = format!(
            " --> test.mylang:1:23
  │
1 │ this is the first line
  │                       ^ {COLOR_UNSET}
"
        );
        assert_eq!(display(span).finish_to_string(), expected);
    }

    #[test]
    fn last_character_in_line() {
        let file = test_file_mock("this is the first line;\nthis is the second line".as_ref());
        let span = word_span(";", &file);
        let expected = format!(
            " --> test.mylang:1:23
  │
1 │ this is the first line;
  │                       ^ {COLOR_UNSET}
"
        );
        assert_eq!(display(span).finish_to_string(), expected);
    }

    #[test]
    fn first_character_in_line() {
        let file = test_file_mock("this is the first line\nsecond line".as_ref());
        let span = word_span("second", &file).start_pos();
        let expected = format!(
            " --> test.mylang:2:1
  │
2 │ second line
  │ ^ {COLOR_UNSET}
"
        );
        assert_eq!(display(span).finish_to_string(), expected);
    }

    #[test]
    fn colored_label() {
        let file = test_file_mock("this is the first line\nthis is the second line".as_ref());
        let span = word_span("first", &file);
        let expected = format!(
            " --> test.mylang:1:13
  │
1 │ this is the first line
  │             {COLOR_RED}^^^^^ label{COLOR_UNSET}
"
        );
        assert_eq!(display(span).color_code(COLOR_RED).label("label").finish_to_string(), expected);
    }
}
