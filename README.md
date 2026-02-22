# End-to-End Hugging Face Model Training & Docker Deployment Report

## 1. Model Selection
We selected the `distilbert-base-cased` model for fine-tuning. 
**Reasons for selection:**
- **Efficiency:** DistilBERT is a smaller, faster, and lighter version of BERT. It retains about 97% of BERT's language understanding capabilities while being 60% faster and 40% smaller. This makes it ideal for running local training workflows, especially in a containerized environment.
- **Cased Version:** We used the *cased* variant to preserve the casing in the text data, which can often be a crucial signal in book reviews and summaries.
- **Classification Capabilities:** DistilBERT integrates seamlessly with `DistilBertForSequenceClassification`, making it highly suited for categorizing reviews into distinct genres.

## 2. Training Summary
- **Data:** We randomly sampled a subset of Goodreads reviews across 8 genres (poetry, children, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, young adult).
- **Preprocessing:** Used `DistilBertTokenizerFast` to tokenize text, keeping `max_length = 512`. Reviews were padded and truncated. 
- **Trainer API:** Used the Hugging Face `Trainer` API to fine-tune the model. Training arguments were configured via `TrainingArguments` to log metrics periodically, and we disabled WandB to keep training fully local. 

## 3. Evaluation Comparison (Local vs Hugging Face)
**Local Evaluation Metrics (Host Machine):**
- **Accuracy:** ~54.0%
- **F1 Score:** ~0.537
- **Validation Loss:** ~1.25


## 4. Challenges Faced
- Managing Docker build times with large machine learning dependencies (PyTorch).
- Avoiding memory limits inside Docker when training.
- Preparing the production Docker image to automatically pull a model payload from Hugging Face for evaluation without needing strict dataset constraints.
