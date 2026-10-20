#!/bin/sh
# Run on Linux with the ARM hard-float toolchain (or inside sim/Dockerfile).
set -eu
if [ "$#" -ne 2 ]; then
    echo "Usage: sh mnn/app/build_arm.sh /path/to/patched/MNN /path/to/tiny-vit.mnn" >&2
    exit 2
fi
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mnn_source=$(CDPATH= cd -- "$1" && pwd)
model_file=$2
test -f "$model_file"
cmake -S "$mnn_source" -B "$repo_dir/build/mnn-arm-static" \
    -DCMAKE_TOOLCHAIN_FILE="$repo_dir/platform/arm-linux-gnueabihf.cmake" \
    -DMNN_FPGA_ROOT="$repo_dir" -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_HARD=ON \
    -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_CONVERTER=OFF -DMNN_SEP_BUILD=OFF \
    -DMNN_ARM82=OFF -DMNN_USE_THREAD_POOL=OFF -DCMAKE_CXX_STANDARD=17
cmake --build "$repo_dir/build/mnn-arm-static" -j 4
cmake -S "$repo_dir/mnn/app" -B "$repo_dir/build/tinyvit-arm-static" \
    -DCMAKE_TOOLCHAIN_FILE="$repo_dir/platform/arm-linux-gnueabihf.cmake" \
    -DMNN_SOURCE="$mnn_source" -DMNN_BUILD="$repo_dir/build/mnn-arm-static" \
    -DTINYVIT_STATIC=ON
cmake --build "$repo_dir/build/tinyvit-arm-static" -j 2
deploy_dir="$repo_dir/build/de1-tinyvit"
mkdir -p "$deploy_dir"
cp "$repo_dir/build/tinyvit-arm-static/tinyvit_run" "$deploy_dir/tinyvit_run"
cp "$model_file" "$deploy_dir/tiny-vit.mnn"
cp "$repo_dir/mnn/app/DEPLOY.md" "$deploy_dir/README.md"
tar -czf "$repo_dir/build/de1-tinyvit.tar.gz" -C "$repo_dir/build" de1-tinyvit
echo "ARM executable: $deploy_dir/tinyvit_run"
echo "Board deployment archive: $repo_dir/build/de1-tinyvit.tar.gz"
