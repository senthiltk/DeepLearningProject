import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    TrainingArguments, 
    Trainer, 
    DistilBertTokenizerFast, 
    DistilBertForTokenClassification, 
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification
)
from datasets import Dataset, ClassLabel, Sequence
import torch
import os

def load_ner_data(csv_path):
    """Load and preprocess NER data from a CSV file."""
    print(f"Loading data from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    
    # Try to read the file with different encodings
    try:
        # First attempt with default encoding
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV with default encoding: {e}")
        try:
            # Second attempt with explicit utf-8 encoding
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"Error reading CSV with utf-8 encoding: {e}")
            try:
                # Third attempt with latin-1 encoding
                df = pd.read_csv(csv_path, encoding='latin-1')
            except Exception as e:
                print(f"Error reading CSV with latin-1 encoding: {e}")
                # Try a more manual approach
                print("Attempting to read file line by line...")
                
                data = {'text': [], 'labels': []}
                with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                    header = f.readline().strip().split(',')
                    for line in f:
                        parts = line.strip().split(',', 1)  # Split only at first comma
                        if len(parts) == 2:
                            data['text'].append(parts[0])
                            data['labels'].append(parts[1])
                
                if not data['text']:
                    raise ValueError(f"Could not read any data from {csv_path}")
                
                df = pd.DataFrame(data)
    
    # Print DataFrame info for debugging
    print(f"Successfully read CSV file. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Process the data into a format suitable for token classification
    texts = []
    tags_list = []
    
    for _, row in df.iterrows():
        # Skip rows with missing or invalid data
        if pd.isna(row['text']) or pd.isna(row['labels']):
            print(f"Warning: Skipping row with missing data: {row}")
            continue
            
        text = str(row['text'])
        labels_str = str(row['labels'])
        
        words = text.split()
        
        # Split the labels string and remove the intent label (first element)
        label_parts = labels_str.split()
        if not label_parts:
            print(f"Warning: No labels found for text: '{text}'")
            continue
        
        intent = label_parts[0]  # First element is the intent (e.g., metro_booking)
        bio_tags = label_parts[1:]  # Rest are BIO tags
        
        # Ensure words and BIO tags have the same length
        if len(words) != len(bio_tags):
            print(f"Warning: Mismatch for '{text}'. Words: {len(words)}, Tags: {len(bio_tags)}")
            continue
        
        # Convert BIO tags to our simplified tag scheme
        tags = []
        for bio_tag in bio_tags:
            if bio_tag.startswith('B-from_station'):
                tags.append('B-FROM')
            elif bio_tag.startswith('I-from_station'):
                tags.append('I-FROM')
            elif bio_tag.startswith('B-to_station'):
                tags.append('B-TO')
            elif bio_tag.startswith('I-to_station'):
                tags.append('I-TO')
            elif bio_tag.startswith('B-num_tickets'):
                tags.append('B-NUM')
            else:
                tags.append('O')
        
        texts.append(words)
        tags_list.append(tags)
    
    # Check if we have any valid data
    if not texts or not tags_list:
        raise ValueError("No valid data found in the CSV file after processing")
    
    print(f"Successfully processed {len(texts)} examples")
    return texts, tags_list

def tokenize_and_align_labels(texts, tags, tokenizer, label_to_id):
    """
    Tokenize the texts and align the labels with the tokens.
    
    Args:
        texts (list): List of tokenized texts (words)
        tags (list): List of token tags
        tokenizer: The tokenizer to use
        label_to_id (dict): Mapping from label to ID
    
    Returns:
        dict: Tokenized texts with aligned labels
    """
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i, (words, word_tags) in enumerate(zip(texts, tags)):
        # Convert words to token IDs
        tokens = []
        word_ids = []
        for word_idx, word in enumerate(words):
            # Tokenize the word and get token IDs
            tokenized_word = tokenizer(word, add_special_tokens=False)["input_ids"]
            tokens.extend(tokenized_word)
            # Map each token to its original word
            word_ids.extend([word_idx] * len(tokenized_word))
        
        # Add special tokens and their mappings
        tokens = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
        word_ids = [None] + word_ids + [None]
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Align labels with tokens
        labels = [-100]  # CLS token gets label -100 (ignored in loss calculation)
        for word_id in word_ids[1:-1]:  # Skip special tokens
            if word_id is not None:
                labels.append(label_to_id[word_tags[word_id]])
            else:
                labels.append(-100)
        labels.append(-100)  # SEP token gets label -100
        
        # Add to tokenized inputs
        tokenized_inputs["input_ids"].append(tokens)
        tokenized_inputs["attention_mask"].append(attention_mask)
        tokenized_inputs["labels"].append(labels)
    
    return tokenized_inputs

def create_ner_datasets(csv_path, tokenizer, max_length=128):
    """
    Create NER datasets for training and evaluation.
    
    Args:
        csv_path (str): Path to the CSV file
        tokenizer: Tokenizer to use
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    # Define the labels - now including I- tags for multi-token entities
    labels = ["O", "B-FROM", "I-FROM", "B-TO", "I-TO", "B-NUM"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    
    # Load and process data
    texts, tags = load_ner_data(csv_path)
    
    # Split data
    train_texts, eval_texts, train_tags, eval_tags = train_test_split(
        texts, tags, test_size=0.2, random_state=42
    )
    
    # Tokenize and align labels
    train_encodings = tokenize_and_align_labels(train_texts, train_tags, tokenizer, label_to_id)
    eval_encodings = tokenize_and_align_labels(eval_texts, eval_tags, tokenizer, label_to_id)
    
    # Create datasets
    train_dataset = Dataset.from_dict(train_encodings)
    eval_dataset = Dataset.from_dict(eval_encodings)
    
    return train_dataset, eval_dataset, labels

def fine_tune_ner_model(csv_path, output_dir='./models/metro_ner'):
    """
    Fine-tune DistilBERT for NER to extract source, destination, and passenger count.
    
    Args:
        csv_path (str): Path to the CSV file with training data
        output_dir (str): Directory to save the fine-tuned model
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Initialize tokenizer and model
    model_id = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
    
    # Create datasets
    train_dataset, eval_dataset, labels = create_ner_datasets(csv_path, tokenizer)
    
    # Initialize the model and ensure it's on CPU
    model = DistilBertForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(labels)
    ).to("cpu")
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Define minimal training arguments that should work with any version
    print("Using basic training arguments compatible with all versions")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=500,
        logging_steps=100,
        # Explicitly use CPU
        no_cuda=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train the model
    print("Fine-tuning the NER model...")
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping
    with open(os.path.join(output_dir, "labels.txt"), "w") as f:
        f.write("\n".join(labels))
    
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer

def extract_entities(text, model, tokenizer):
    """Extract source station, destination station, and number of passengers from text."""
    # Load label mapping
    if os.path.exists(os.path.join(model.config._name_or_path, "labels.txt")):
        with open(os.path.join(model.config._name_or_path, "labels.txt"), "r") as f:
            id_to_label = {i: label for i, label in enumerate(f.read().splitlines())}
    else:
        # Default label mapping for B- and I- tags
        id_to_label = {0: "O", 1: "B-FROM", 2: "I-FROM", 3: "B-TO", 4: "I-TO", 5: "B-NUM"}
    
    # Explicitly move model to CPU
    model = model.to("cpu")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Ensure inputs are on CPU
    inputs = {key: tensor.to("cpu") for key, tensor in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Extract entities with improved logic for multi-token entities (B-/I- tags)
    current_entity_type = None
    current_tokens = []
    from_station = []
    to_station = []
    num_tickets = []
    
    for token, prediction in zip(tokens, predictions[0]):
        # Skip special tokens
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        # Get the label safely
        pred_id = prediction.item()
        if pred_id >= len(id_to_label):
            continue  # Skip if prediction ID is out of range
            
        label = id_to_label[pred_id]
        
        # Handle subword tokens (starting with ##)
        if token.startswith("##"):
            if current_tokens:  # if we're building a token
                current_tokens.append(token[2:])
            continue
        
        # Determine entity type from label (stripping B- or I- prefix)
        if label.startswith("B-") or label.startswith("I-"):
            entity_type = label[2:]  # Remove "B-" or "I-"
        else:
            entity_type = None
                
        # Complete the current entity if we have one and:
        # 1. We hit an O tag, or
        # 2. We hit a B- tag of any type (starting new entity), or
        # 3. We hit an I- tag of a different entity type
        if current_tokens and (
            label == "O" or 
            label.startswith("B-") or 
            (label.startswith("I-") and entity_type != current_entity_type)
        ):
            # Join the tokens, considering we might have collected multiple tokens
            entity_text = " ".join(current_tokens).replace(" ##", "")
            if current_entity_type == "FROM":
                from_station.append(entity_text)
            elif current_entity_type == "TO":
                to_station.append(entity_text)
            elif current_entity_type == "NUM":
                num_tickets.append(entity_text)
            # Reset for the next entity
            current_tokens = []
            current_entity_type = None
        
        # Start a new entity with B- tag
        if label.startswith("B-"):
            current_entity_type = entity_type
            current_tokens = [token]
        # Continue an existing entity with I- tag of matching type
        elif label.startswith("I-") and (
            current_entity_type == entity_type or not current_entity_type
        ):
            # If we get an I- tag without a preceding B-, we'll start collecting anyway
            current_entity_type = entity_type
            current_tokens.append(token)
        
    # Handle the last entity if there is one
    if current_tokens and current_entity_type:
        entity_text = " ".join(current_tokens).replace(" ##", "")
        if current_entity_type == "FROM":
            from_station.append(entity_text)
        elif current_entity_type == "TO":
            to_station.append(entity_text)
        elif current_entity_type == "NUM":
            num_tickets.append(entity_text)
    
    # Post-process the extracted entities
    # Join multiple tokens for each entity type if necessary
    from_station_text = "".join(from_station) if from_station else None
    to_station_text = "".join(to_station) if to_station else None
    num_tickets_text = " ".join(num_tickets) if num_tickets else None
    
    # Try to convert numeric words to numbers
    if num_tickets_text:
        num_map = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "single": "1", "couple": "2", "few": "3","a":"1"
        }
        if num_tickets_text.lower() in num_map:
            num_tickets_text = num_map[num_tickets_text.lower()]
    
    return {
        "from_station": from_station_text,
        "to_station": to_station_text,
        "num_tickets": num_tickets_text
    }

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "~/IISc/DeepLearning/Project/data/metro_en_fine_tune_new.csv"
    
    try:
        # Fine-tune the model
        model, tokenizer = fine_tune_ner_model(csv_path)
        
        # Test the model on some examples
        examples = [
            "Book 2 tickets from MG Road to Indiranagar",
            "I need a single ticket from Baiyappanahalli to Cubbon Park",
            "Get me 3 tickets from Nagasandra to Jayanagar"
        ]
        
        print("\nTesting model on examples:")
        for example in examples:
            try:
                entities = extract_entities(example, model, tokenizer)
                print(f"Text: {example}")
                print(f"Extracted: {entities}")
                print()
            except Exception as e:
                print(f"Error processing example '{example}': {str(e)}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()

