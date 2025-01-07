from transformers import pipeline
import gradio as gr

# Load different models
models = {
    "t5-small": pipeline("summarization", model="t5-small"),
    "t5-base": pipeline("summarization", model="t5-base")
    # "t5-large": pipeline("summarization", model="t5-large"),
}

def predict(prompt, model_name, summary_length, language):
    model = models[model_name]
    max_length = 130 if summary_length == "short" else 200 if summary_length == "medium" else 300
    summary = model(prompt, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# Create an interface for the model
with gr.Blocks() as interface:
    gr.Markdown("# Advanced Text Summarization")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Input Text", placeholder="Enter text to summarize...", lines=10)
            model_name = gr.Dropdown(label="Model", choices=["t5-small", "t5-base"], value="t5-base")
            summary_length = gr.Dropdown(label="Summary Length", choices=["short", "medium", "long"], value="medium")
            language = gr.Dropdown(label="Language", choices=["English", "French", "German"], value="English")
            submit_button = gr.Button("Summarize")
        
        with gr.Column():
            output = gr.Textbox(label="Summary", lines=10)
    
    submit_button.click(fn=predict, inputs=[prompt, model_name, summary_length, language], outputs=output)

interface.launch(share=True)