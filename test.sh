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

esc=$(printf '\033')
reset="${esc}[0m"
cyan="${esc}[0;36m"
bold_red="${esc}[1;91m"
bold_green="${esc}[1;92m"

#ASAN_VAR="RUSTFLAGS=-Zsanitizer=address"
ASAN_TARGET="${ASAN_VAR+"x86_64-unknown-linux-gnu"}"

tests="$@"

watchdir -c "$ASAN_VAR cargo test $ASAN_TARGET --color=always -- $tests --nocapture --test-threads 1" 2>&1 | \
    while IFS= read -r empty_line1; do # `sed '//d'` and `grep -v` collect all lines before printing
        if ! [[ -z "$empty_line1" ]]; then
            echo "$empty_line1"
            continue
        fi

        read -r test_count_line
        if ! [[ "$test_count_line" =~ (^running ([0-9]+) test[s]?$) ]]; then
            echo "$empty_line1"
            echo "$test_count_line"
            continue
        elif [[ "${BASH_REMATCH[2]}" != "0" ]]; then
            echo "$test_count_line"
            continue
        fi

        read -r empty_line2
        read -r empty_test_results
        if ! [[ "$empty_test_results" =~ (^test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured;) ]]; then
            echo "$empty_line1"
            echo "$test_count_line"
            echo "$empty_line2"
            echo "$empty_test_results"
            continue
        fi

        read -r empty_line3
    done | \
    sed "s/^test \([^ ]*\) \.\.\. /${cyan}TEST: \1${reset} ...\n/;\
        s/\(\<FAILED\>\)/${bold_red}\1${reset}/;\
        s/\(\<ok\>\)$/${bold_green}\1${reset}\n/"
