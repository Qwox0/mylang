#!/usr/bin/env bash

set -e

script_dir="$(dirname "$(readlink -f "$0")")"

info() {
    echo -e "\e[92;1m+++\e[0m $1"
}

set -x

cd "$script_dir"

cargo run -- build ./take_arr.mylang --out=obj --lib

clang -o ./out/main ./main.c ./out/take_arr.o

./out/main
