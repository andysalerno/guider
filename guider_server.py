import sys
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from sentence_transformers import SentenceTransformer

from llama_attn_hijack import hijack_llama_attention_xformers

if Path("./guidance").exists():
    sys.path.insert(0, str(Path("guidance")))
    print("Adding ./guidance to import tree.")
import guidance

MODEL_EXECUTOR: str = None


# EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
# EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def setup_models(model_name: str):
    """
    model_name: a Huggingface path like TheBlock/tulu-13B-GPTQ
    """

    # A slight improvement in memory usage by using xformers attention:
    if '--xformers' in sys.argv:
        hijack_llama_attention_xformers()

    model = None

    print(f"Loading model with executor: {MODEL_EXECUTOR}")

    if MODEL_EXECUTOR == "autogptq":
        from llama_autogptq import LLaMAAutoGPTQ

        model = LLaMAAutoGPTQ(model_name)
    elif MODEL_EXECUTOR == "gptq":
        from llama_gptq import LLaMAGPTQ

        model = LLaMAGPTQ(model_name)
    elif MODEL_EXECUTOR == "exllama":
        from llama_exllama import ExLLaMA

        model = ExLLaMA(model_name)
    elif MODEL_EXECUTOR == "ctransformers":
        raise Exception("Not yet implemented")

    guidance.llms.Transformers.cache.clear()
    guidance.llm = model
    print(f"Token healing enabled: {guidance.llm.token_healing}")


class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print(f"post incoming: {self.path}")

        if self.path == "/embeddings":
            self.handle_embeddings()
        elif self.path == "/chat":
            self.handle_chat_streaming()

    def handle_embeddings(self):
        print("embeddings requested")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        content_length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(content_length).decode("utf-8"))

        text = body["input"] if "input" in body else body["text"]

        if type(text) is str:
            text = [text]

        embeddings = embedding_model.encode(text).tolist()

        data = [
            {"object": "embedding", "embedding": emb, "index": n}
            for n, emb in enumerate(embeddings)
        ]

        response = json.dumps(
            {
                "object": "list",
                "data": data,
                "model": EMBEDDING_MODEL_NAME,
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

        self.wfile.write(response.encode("utf-8"))

    def handle_chat_streaming(self):
        print("streaming chat requested")
        guidance.llms.Transformers.cache.clear()

        content_length = int(self.headers["Content-Length"])  # Get the size of data
        post_data = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(post_data)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        # self.send_header('Connection', 'keep-alive')
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        template: str = data["template"]
        parameters: dict = data["parameters"]

        g = guidance(template, stream=True)

        output_skips = {}

        skip_text = 0

        print("streaming model output...")
        for output in g(**parameters):
            print("\n...streaming chunk...")

            filtered_variables = dict()

            for key, val in output.variables().items():
                if (
                    isinstance(val, str)
                    and key not in parameters.keys()
                    and key != "llm"
                    and key != "@raw_prefix"
                ):
                    skip = output_skips.get(key, 0)
                    filtered_variables[key] = val[skip:]
                    output_skips[key] = len(val)

            print(f"variables: {filtered_variables}")

            response = {
                "text": output.text[skip_text:],
                "variables": filtered_variables,
            }

            print(f"response:\n{response}")

            skip_text = len(output.text)

            response_str = f"data: {json.dumps(response).rstrip()}\n\n"
            sys.stdout.flush()
            self.wfile.write("event: streamtext\n".encode("utf-8"))
            self.wfile.write(response_str.encode("utf-8"))
            self.wfile.flush()

        print("done getting output from model.")


def run(server_class=HTTPServer, handler_class=MyHandler):
    global MODEL_EXECUTOR

    if len(sys.argv) != 3:
        raise Exception(
            "Expected to be invoked with two arguments: model_name and executor"
        )

    model_name = sys.argv[1]

    MODEL_EXECUTOR = sys.argv[2]

    setup_models(model_name)

    server_address = ("0.0.0.0", 8000)
    httpd = server_class(server_address, handler_class)
    print("Starting httpd...\n")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
