# **TaleWeaver: AI-Powered Interactive Storytelling**  

TaleWeaver fine-tunes the **Mistral 7B** model using **LoRA (Low-Rank Adaptation)** to generate immersive, interactive stories based on prompts, genre mix, and emotions. This AI-driven system crafts unique narratives while offering dynamic story path choices.

## 🚀 **Features**
- Fine-tunes **Mistral-7B** for **custom story generation**.
- Uses **LoRA** for efficient and lightweight training.
- Dynamically adjusts **max token length** based on dataset requirements.
- Supports **multi-genre storytelling** with structured input prompts.
- Generates **interactive choices** to shape story direction.

## 📂 **Project Structure**

📦 TaleWeaver ┣ 📜 final.csv # Dataset (Prompts, Genre, Choices, Stories) ┣ 📜 TaleWeaver.ipynb # Jupyter Notebook for Training ┣ 📜 train.py # Python Script for Training (Standalone) ┣ 📜 requirements.txt # Dependencies ┣ 📂 mistral_finetuned # Saved Fine-Tuned Model ┗ 📜 README.md # Project Documentation


## 🔧 **Installation**
First, clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/TaleWeaver.git
cd TaleWeaver
pip install -r requirements.txt

📊 Dataset Format
The dataset (final.csv) consists of:

Prompt → The initial idea or seed for the story.

Genre Mix → Weighted genre distribution (e.g., 50% Horror, 30% Adventure).

Emotion → The mood of the story.

Choices → Interactive decisions for the next path.

Full Story → The generated story.

Next Story Path → Suggested continuation based on choices.

🛠 Fine-Tuning Process
1️⃣ Load Model & Tokenizer
python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Assign padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
2️⃣ Preprocess Dataset (Dynamic Max Length)
python
Copy
Edit
def calculate_max_length(dataset, tokenizer):
    max_length = 0
    for row in dataset:
        input_text = f"Prompt: {row['prompt']}\nGenre Mix: {row['genre_mix']}\nEmotion: {row['emotion']}\nChoices: {row['choices']}\nStory:"
        target_text = f"{row['full_story']}\nNext Story Path: {row['next_story_path']}"
        max_length = max(max_length, len(tokenizer(input_text)["input_ids"]), len(tokenizer(target_text)["input_ids"]))
    return max_length

MAX_LENGTH = calculate_max_length(dataset["train"], tokenizer)

def preprocess_function(examples):
    input_text = f"Prompt: {examples['prompt']}\nGenre Mix: {examples['genre_mix']}\nEmotion: {examples['emotion']}\nChoices: {examples['choices']}\nStory:"
    target_text = f"{examples['full_story']}\nNext Story Path: {examples['next_story_path']}"

    return {
        "input_ids": tokenizer(input_text, truncation=True, padding="max_length", max_length=MAX_LENGTH)["input_ids"],
        "labels": tokenizer(target_text, truncation=True, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
    }
3️⃣ Apply LoRA for Fine-Tuning
python
Copy
Edit
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16,             
    lora_alpha=32,    
    target_modules=["q_proj", "v_proj"],  # LoRA applied to attention layers
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
4️⃣ Train the Model
python
Copy
Edit
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="./taleweaver_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_dataset["train"], 
    eval_dataset=tokenized_dataset["test"]
)
trainer.train()
💾 Saving & Using the Model
python
Copy
Edit
model.save_pretrained("./taleweaver_finetuned")
tokenizer.save_pretrained("./taleweaver_finetuned")
🔮 Generate a Story
python
Copy
Edit
from transformers import pipeline
generator = pipeline("text-generation", model="./taleweaver_finetuned", tokenizer="./taleweaver_finetuned")

story_prompt = "Prompt: A haunted house in the woods\nGenre Mix: 60% Horror, 40% Mystery\nEmotion: Suspense\nChoices: Run away, Enter the house"
output = generator(story_prompt, max_length=512)
print(output)
📌 To-Do
 Enhance prompt engineering for more diverse outputs.

 Experiment with different LoRA configurations for efficiency.

 Deploy the model via Hugging Face Spaces or Flask API.

🤝 Contributing
Want to improve TaleWeaver? Feel free to open issues or submit pull requests!

📜 License
MIT License

csharp
Copy
Edit

### 📝 How to Use This:
1. Save this content in a file named `README.md` inside your project directory.
2. Add it to your GitHub repository:
   ```bash
   git add README.md
   git commit -m "Added README for TaleWeaver"
   git push origin main
