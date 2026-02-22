import os
import argparse
import torch
from transformers import DistilBertForSequenceClassification, Trainer
from data import prepare_datasets
from utils import compute_metrics
from sklearn.metrics import classification_report

def run_evaluation(model_path, model_name='distilbert-base-cased'):
    print(f"Loading data and model for evaluation: {model_path}")
    _, test_dataset, label2id, id2label, tokenizer = prepare_datasets(model_name=model_name)
    
    device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device_name}")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    ).to(device_name)
    
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    results = trainer.evaluate()
    print("Evaluation Results:", results)

    # Save results to a file
    os.makedirs('results', exist_ok=True)
    with open('results/eval_metrics.txt', 'w') as f:
        f.write(str(results))
    
    predicted_results = trainer.predict(test_dataset)
    predicted_labels = predicted_results.predictions.argmax(-1)
    predicted_labels = [id2label[l] for l in predicted_labels]
    test_labels = [id2label[l.item()] for l in test_dataset.labels]
    
    report = classification_report(test_labels, predicted_labels)
    print(report)
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
        
    print("Evaluation complete and results saved to 'results' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to local or HF hub model")
    parser.add_argument('--model_name', type=str, default='distilbert-base-cased', help="Base model name")
    args = parser.parse_args()
    
    os.environ["WANDB_DISABLED"] = "true"
    run_evaluation(args.model_path, args.model_name)
