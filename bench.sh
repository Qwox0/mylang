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

cargo build --release

sha256sum ./mylang-old
sha256sum ./target/release/mylang

cd "$script_dir"

bench() {
    sudo $(which poop) "./mylang-old $*" "./target/release/mylang $*"
}

bench check ./lib/std/bindgen/libclang.mylang --lib
#bench check ./lib/std/bindgen/libglfw3.mylang --lib
bench check ../../opengl_sphaerophoria/main.mylang

#for i in $(seq 1 9); do
#    bench check "../../aoc2024/day0${i}.mylang"
#done

# for i in $(seq 1 9); do
#     bench build "../../aoc2024/day0${i}.mylang"
# done
