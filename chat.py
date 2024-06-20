from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-11B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-11B", trust_remote_code=True)

# Function to generate response
def generate_response(prompt, model, tokenizer, max_length=100):
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response using the model
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Chat function
def chat():
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Exit the chat loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Generate response
        response = generate_response(user_input, model, tokenizer)
        
        # Print the response
        print(f"Bot: {response}")

# Start the chat
chat()
