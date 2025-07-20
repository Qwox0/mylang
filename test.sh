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

tests="$@"

(set -x
watchexec "cargo test -- $tests --nocapture --test-threads 1") | \
    sed "s/^test \([^ ]*\) \.\.\. \(FAILED\)\?\(ok\)\?/${cyan}TEST: \1${reset} ... ${bold_red}\2${bold_green}\3${reset}\n/"
