import subprocess
import threading
import os


def launch_process(cmd):
    os.popen(cmd)


# Load your model and tokenizer (example using GPT-2)
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Replace this with your model path or identifier

# Using 127.0.0.1 because localhost does not work properly in Colab

def run_controller():
    launch_process("python -m fastchat.serve.controller --host 127.0.0.1")

def run_model_worker():
    # launch_process(f"python -m fastchat.serve.model_worker --stream --device cpu --host 127.0.0.1 --controller-address http://127.0.0.1:21001 --model-path {model_name}")
    launch_process(f"python -m fastchat.serve.multi_model_worker --stream --device cpu --host 127.0.0.1 \
                   --controller-address http://127.0.0.1:21001 \
                   --model-path HuggingFaceTB/SmolLM2-135M-Instruct  \
                   --model-names SmolLM2-135M-Instruct  \
                   --model-path HuggingFaceTB/SmolLM2-360M-Instruct  \
                   --model-names SmolLM2-360M-Instruct  \
                   ")

def run_api_server():
    launch_process("python -m fastchat.serve.openai_api_server --host 127.0.0.1 --controller-address http://127.0.0.1:21001 --port 8000")


if __name__ == "__main__":
    run_controller()
    run_api_server()
    run_model_worker()