#!/usr/bin/env bash

set -e

script_dir="$(dirname "$(readlink -f "$0")")"

info() {
    echo -e "\e[92;1m+++\e[0m $1"
}

cd "$script_dir"
libpath="$script_dir/out"

mkdir -p out

info "Building mylib"

(set -x
gcc -c -o ./out/mylib.o ./mylib.c
ar rcs ./out/libmylib.a ./out/mylib.o
ld.lld -shared -o ./out/libmylib.so ./out/mylib.o
)

info "Running example"
cargo run -- run ./main.mylang
