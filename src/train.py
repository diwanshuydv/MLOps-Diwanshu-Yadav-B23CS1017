import os
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data import prepare_datasets
from utils import compute_metrics
import argparse

def train_model(model_name='distilbert-base-cased', output_dir='./results', epochs=1):
    print("Preparing datasets...")
    train_dataset, test_dataset, label2id, id2label, tokenizer = prepare_datasets(model_name=model_name)
    
    device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device_name}")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    ).to(device_name)
    
    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        warmup_steps=50,
        weight_decay=0.01,
        output_dir=output_dir,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to=[] # disable wandb
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    trainer.train()
    print("Training finished. Evaluating...")
    
    eval_results = trainer.evaluate()
    print("Eval Results:", eval_results)
    
    # Save the model
    os.makedirs(f"{output_dir}/final_model", exist_ok=True)
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    print(f"Model saved to {output_dir}/final_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='distilbert-base-cased')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    
    os.environ["WANDB_DISABLED"] = "true"
    train_model(model_name=args.model_name, epochs=args.epochs)
