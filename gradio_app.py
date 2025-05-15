import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

model_name = "mayankpuvvala/peft_lora_t5_merged_model_pytorch_issues"

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")  # adjust if needed
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

def generate_response(prompt):
    if model is None:
        return "Model not loaded properly."

    prompt = prompt.strip()
    if not prompt:
        return "Empty prompt."

    if len(prompt.split()) > 150:
        return "Prompt too long. Limit to ~150 words."

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.90,
                top_k=50,  # ADD THIS
                top_p=0.95,  # ADD THIS
                eos_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "CUDA out of memory."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Gradio interface
gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="PyTorch Issue Generator",
    description="Enter a prompt describing a PyTorch issue scenario or mainly title."
).launch(share= True)
