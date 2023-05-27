from http.server import BaseHTTPRequestHandler, HTTPServer
import guidance
from llama_quantized import LLaMAQuantized
from sentence_transformers import SentenceTransformer
import json

guidance.llms.Transformers.cache.clear()
guidance.llm = LLaMAQuantized(model_dir='models', model='Wizard-Vicuna-13B-Uncensored-GPTQ')
print(f'Token healing enabled: {guidance.llm.token_healing}')

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

class MyHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print(f'post incoming: {self.path}')

        if self.path == '/embeddings':
            self.handle_embeddings()
        elif self.path == '/chat':
            self.handle_chat()

    def handle_embeddings(self):
        print('embeddings requested')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        input = body['input'] if 'input' in body else body['text']

        if type(input) is str:
            input = [input]

        embeddings = embedding_model.encode(input).tolist()

        data = [{"object": "embedding", "embedding": emb, "index": n} for n, emb in enumerate(embeddings)]

        response = json.dumps({
            "object": "list",
            "data": data,
            "model": EMBEDDING_MODEL_NAME,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            }
        })

        self.wfile.write(response.encode('utf-8'))

    def handle_chat(self):
        print('chat requested')
        content_length = int(self.headers['Content-Length']) # Get the size of data
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(post_data)

        print(f"saw request body: {data}")

        template: str = data['template']
        parameters: dict = data['parameters']

        g = guidance(template, silent=False)

        print('getting model output...')
        model_output = g(**parameters)
        print('done.')

        all_variables = model_output.variables()

        filtered_variables = dict()

        for key, val in all_variables.items():
            if key not in parameters.keys() and key != 'llm':
                filtered_variables[key] = val

        print(filtered_variables)

        response = {
            'text': model_output.text,
            'variables': filtered_variables
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=MyHandler):
    server_address = ('0.0.0.0', 8000)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...\n')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
