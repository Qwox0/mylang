#!/usr/bin/env bash

set -e

script_dir="$(dirname "$0")"

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

mkdir -p out

cmd="cargo run -q -- build ./main.mylang --out=obj"
info "$cmd"
$cmd

cmd="gcc out/main.o mylib.c -o out/main"
info "$cmd"
$cmd

info "Running ..."
err=0
./out/main || err=$?

info "out: $err"

exit $err
