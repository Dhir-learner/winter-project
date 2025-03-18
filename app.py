from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load the tokenizer and model
MODEL_PATH = 'C://Users//Dhir//OneDrive//Desktop//winter project//t5-summarization-model'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
