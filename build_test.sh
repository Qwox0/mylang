#!/usr/bin/env bash

script_dir="$(dirname "$0")"

info() {
    echo -e "\e[92;1m+++\e[0m $1"
}

warn() {
    echo -e "\e[93;1m+++ WARN:\e[0m $1"
}

error() {
    echo -e "\e[91;1m+++ ERROR:\e[0m $1" >&2
    exit $2
}

cd "$script_dir"

rm "target/output.o" "target/test" 2>/dev/null

cmd="cargo run"
info "$cmd"
#RUSTFLAGS=-Awarnings $cmd || error "Failed" 0
$cmd || error "Failed" 0

cmd="gcc test.c target/output.o -o target/test"
info "$cmd"
$cmd || error "Failed" 1

cmd="./target/test"
info "$cmd"
$cmd || error "Failed" 2

cargo bench
