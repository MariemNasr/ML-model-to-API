import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "medalpaca/medalpaca-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # Will use the T4 GPU
)

# Chat function that uses GPU
def chat_with_bot(user_input):
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

    # Tokenize and move input to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate output using model on GPU
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and extract the bot response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### Response:")[-1].strip()
    return answer