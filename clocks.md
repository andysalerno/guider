nvidia-smi --query-supported-clocks=timestamp,gpu_name,gpu_uuid,memory,graphics --format=csv
watch nvidia-smi --format=csv --query-gpu clocks.mem,clocks.gr

sudo nvidia-smi -lgc 1530,1530
sudo nvidia-smi -lmc 5001,5001