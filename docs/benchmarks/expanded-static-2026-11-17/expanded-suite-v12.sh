#!/bin/sh
set -eu
cd /home/ubuntu/tinyvit
sof_sha="$1"
trap 'status=$?; echo "$status" > expanded-suite-v12.exit; echo SUITE_EXIT="$status"' EXIT
printf 'BEGIN numeric\n'
python3 benchmark_expanded_numeric.py --runner tinyvit_native_v12 --inputs 20 --bitstream-sha256 "$sof_sha" --output-dir /home/ubuntu/tinyvit/benchmark-native-ni83-v12-numeric-20 > expanded-numeric-v12.log 2>&1
printf 'BEGIN random\n'
python3 benchmark_native_expanded_v12.py --runner tinyvit_native_v12 --configurations expanded-configurations-v12.json --modes fp32,cpu,pairs,transformer_cpu,transformer_fpga,static_cpu,static_fpga,static_unfused --inputs 20 --abi NI83 --experiment-date 2026-11-17 --bitstream-sha256 "$sof_sha" --output-dir /home/ubuntu/tinyvit/benchmark-native-ni83-v12-random-20 > expanded-random-v12.log 2>&1
printf 'BEGIN natural\n'
python3 benchmark_native_expanded_v12.py --runner tinyvit_native_v12 --configurations expanded-configurations-v12.json --modes fp32,cpu,pairs,transformer_cpu,transformer_fpga,static_cpu,static_fpga --inputs 20 --abi NI83 --experiment-date 2026-11-17 --bitstream-sha256 "$sof_sha" --natural-images /home/ubuntu/tinyvit/accuracy-real-20 --output-dir /home/ubuntu/tinyvit/benchmark-native-ni83-v12-natural-20 > expanded-natural-v12.log 2>&1
printf 'COMPLETE all experiments\n'
