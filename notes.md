sudo podman run -d --rm --gpus all -p 8000:8000 --device nvidia.com/gpu=all --security-opt=label=disable -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all localhost/embeddings-server

sudo podman run -d --rm -p 80:5173 localhost/chat-ui

sudo podman run -d --rm -v ~/mnt/drive/models:/data -p 8080:80 --gpus all --device nvidia.com/gpu=all --security-opt=label=disable -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all ghcr.io/huggingface/text-generation-inference:latest --model-id berkeley-nest/Starling-LM-7B-alpha --quantize eetq --max-best-of 1 --max-concurrent-requests 2 --validation-workers 1 --max-total-tokens 16000 --max-top-n-tokens 1 --max-input-length 8191 --max-batch-prefill-tokens 16000 --revision c5f53e9fc7978edac5a5ff74e8fe957a6e076212

podman start mongo-chatui