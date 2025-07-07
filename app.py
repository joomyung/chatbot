import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Ensure torch is available for the pipeline
# Initialize the FastAPI app
app = FastAPI()

# Load a Hugging Face model for text generation (adjust model name or path as needed)
generator = pipeline("text-generation", model="gpt2")  # or your local model path

# Serve a simple HTML chat page with a text area to enter prompts and display multiple outputs
@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Chat App</title>
    </head>
    <body>
        <h1>LLM Chat App</h1>
        <textarea id="prompt" rows="4" cols="50" placeholder="Enter your question here..."></textarea><br><br>
        <button onclick="chat()">Send</button>
        <div id="output" style="margin-top: 20px;"></div>
        <script>
            async function chat() {
                const prompt = document.getElementById('prompt').value;
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const data = await response.json();
                const container = document.getElementById('output');
                container.innerHTML = '';
                data.generated_texts.forEach((text, idx) => {
                    const pre = document.createElement('pre');
                    pre.style.whiteSpace = 'pre-wrap';
                    pre.style.border = '1px solid #ccc';
                    pre.style.padding = '10px';
                    pre.style.marginBottom = '10px';
                    pre.textContent = `Example ${idx + 1}:\n${text}`;
                    container.appendChild(pre);
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Define request body for chat endpoint
class ChatRequest(BaseModel):
    prompt: str

# API endpoint to perform inference and return multiple generated examples
@app.post("/chat")
async def chat(req: ChatRequest):
    # Perform inference using the Hugging Face pipeline with 3 examples
    outputs = generator(req.prompt, max_length=100, num_return_sequences=3)

    # Extract generated texts
    texts = [out['generated_text'] for out in outputs]
    # texts = ['hello world', 'hello world 2']
    return JSONResponse({'generated_texts': texts})

# Run the app with uvicorn
def main():
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
