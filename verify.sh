#!/usr/bin/env bash

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

cmd="cargo test -q"
info "$cmd"
$cmd

cmd="cargo test -q --release"
info "$cmd"
$cmd
