#!/usr/bin/env bash

set -e

script_dir="$(dirname "$(readlink -f "$0")")"

info() {
    echo -e "\e[92;1m+++\e[0m $1"
}

warn() {
    echo -e "\e[93;1m+++ WARN:\e[0m $1"
}

error() {
    echo -e "\e[91;1m+++ ERROR:\e[0m $1" >&2
    if [ $2 -ne 0 ]; then exit $2; fi
}

cd "$script_dir"

#ASAN_VAR="RUSTFLAGS=-Zsanitizer=address"
ASAN_TARGET="${ASAN_VAR+"x86_64-unknown-linux-gnu"}"

tests="$@"

SHORT_BACKTRACE_LEN=9
short_backtrace="s/\
(\nstack backtrace:\n)([ ]*)0: __rustc::rust_begin_unwind\s*at [^\n]*\n\
(([ ]*\d+: [^\n]*\n[ ]*at \S*\n){,$SHORT_BACKTRACE_LEN})\
([ ]*\d+: [^\n]*\n[ ]*at \S*\n)*\
/\1\3\2?: ... (short backtrace)\n/g"

remove_newline_test_stats="s/\n(\nrunning \d tests?\n)/\1/g"
remove_0_tests_stats="s/running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured;[^\n]*\n\n//g"

highlight_individual_tests="s/test (\S*) \.\.\. (ignored[^\n]*)?\n?/$(tput setaf 6)TEST: \1$(tput sgr0) ... \2\n/g"
highlight_failed="s/(FAILED\n)\s*/$(tput setaf 9 bold)\1$(tput sgr0)\n/g"
highlight_ok="s/\s*(\sok\n)\s*/$(tput setaf 10 bold)\1$(tput sgr0)\n/g"

perl_filter="2>&1 | perl -0pe '
    ;$short_backtrace
    ;$remove_newline_test_stats
    ;$remove_0_tests_stats
    ;$highlight_individual_tests
    ;$highlight_failed
    ;$highlight_ok
'"

RUST_BACKTRACE=1 \
watchdir -c "$ASAN_VAR cargo test $ASAN_TARGET --color=always -- $tests --nocapture --test-threads 1 $perl_filter"
