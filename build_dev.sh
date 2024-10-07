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

rm "target/build_dev/output.o" "target/build_dev/test" 2>/dev/null

cmd="cargo run"
info "$cmd"
#RUSTFLAGS=-Awarnings $cmd || error "Failed" 0
#RUST_BACKTRACE=1 $cmd || error "Failed" 0 # slows down the Frontend (especially in non-release mode)
$cmd || error "Failed" 0

cmd="gcc test.c target/build_dev/output.o -o target/build_dev/test"
info "$cmd"
$cmd || error "Failed" 1

cmd="./target/build_dev/test"
info "$cmd"
$cmd || error "Failed" 2

cargo bench
