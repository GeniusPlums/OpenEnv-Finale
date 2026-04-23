import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset


def train_sft(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    data_path: str = "data/sft/warmstart_conversations.jsonl",
    output_dir: str = "checkpoints/sft",
    num_train_epochs: float = 1.0,
    learning_rate: float = 2e-5,
):
    """SFT warm-start on agent-only turns from high-reward conversations."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"SFT data not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parse conversations into agent-turn training samples
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for turn in obj.get("history", []):
                agent_text = turn.get("agent_text", "").strip()
                if agent_text:
                    samples.append({"text": agent_text})

    dataset = Dataset.from_list(samples)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT checkpoint saved to {output_dir}")


if __name__ == "__main__":
    train_sft()
