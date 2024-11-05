import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler
from transformers import DataCollatorWithPadding
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def fine_tune_and_evaluate2(model_name, data, labels, paths, sampling_method='none'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        print(f"Training fold {fold + 1}/{skf.n_splits}")

        # Sampling
        if sampling_method == 'oversample':
            ros = RandomOverSampler(random_state=42)
            train_idx, _ = ros.fit_resample(np.array(train_idx).reshape(-1, 1), labels[train_idx])
        elif sampling_method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            train_idx, _ = rus.fit_resample(np.array(train_idx).reshape(-1, 1), labels[train_idx])

        train_texts, train_labels = data[train_idx], labels[train_idx]
        test_texts, test_labels = data[test_idx], labels[test_idx]

        # Tokenization
        train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512,
                                    return_tensors="pt")
        test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512, return_tensors="pt")

        # Dataset creation
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'],
                                      torch.tensor(train_labels.values))
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'],
                                     torch.tensor(test_labels.values))

        # DataLoader setup
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        # Optimizer and learning rate scheduler setup
        optimizer = AdamW(model.parameters(), lr=5e-5)
        num_training_steps = len(train_loader) * 3  # number of epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Training loop
        model.train()
        for epoch in range(3):  # number of epochs
            for batch in train_loader:
                batch = tuple(item.to(device) for item in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                print(f"Training loss: {loss.item()}")

            # Evaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = tuple(item.to(device) for item in batch)
                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch[2].cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
            results.append({'accuracy': acc, 'precision': precision, 'recall': recall, 'f1_score': f1})
            print(f"Results for fold {fold + 1}: Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    # Saving results
    results_df = pd.DataFrame(results)
    results_df.to_csv(paths + "/results.csv", index=False)
    print("Training complete.")

    return results_df


class PlottingCallback(TrainerCallback):
    """
    A callback that plots the training loss and evaluation metric at the end of each epoch.
    """
    def __init__(self, output_dir):
        self.training_losses = []
        self.eval_metrics = []
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        # Check if there are any logs and append the last one if it exists
        if state.log_history:
            self.training_losses.append(state.log_history[-1].get('loss', None))
            eval_log = [log for log in state.log_history if 'eval_loss' in log]
            if eval_log:
                self.eval_metrics.append(eval_log[-1]['eval_loss'])

        if self.eval_metrics:
            plt.subplot(1, 1, 1)
            plt.plot(self.eval_metrics, label='Evaluation Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Evaluation Loss Over Epochs')
            plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig(f"{self.output_dir}/training_curve_epoch_{state.epoch}.png")
        plt.close()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, np.squeeze(predictions)),
        'recall': recall_score(labels, np.squeeze(predictions)),
        'f1_score': f1_score(labels, np.squeeze(predictions))
    }


def fine_tune_and_evaluate(model_name, data, labels, paths, sampling_method='none'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The device is {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_model = None
    best_metric = 0

    for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):
        print(f"Training fold {fold + 1}/{skf.n_splits}")

        if sampling_method == 'oversample':
            ros = RandomOverSampler(random_state=42)
            train_idx, _ = ros.fit_resample(np.array(train_index).reshape(-1, 1), labels[train_index])
        elif sampling_method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            train_idx, _ = rus.fit_resample(np.array(train_index).reshape(-1, 1), labels[train_index])
        else:
            train_idx = train_index

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        #model.to(device)

        train_texts, test_texts = data[np.squeeze(train_idx)], data[test_index]
        train_labels, test_labels = labels[np.squeeze(train_idx)], labels[test_index]

        train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': test_labels
        })

        training_args = TrainingArguments(
            output_dir=os.path.join(paths.path_result, f'results_fold_{fold}'),
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='f1_score',
            report_to=[]
        )

        plotting_callback = PlottingCallback(output_dir=paths.path_result)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[plotting_callback]
        )

        trainer.train()
        eval_results = trainer.evaluate()

        if eval_results['eval_f1_score'] > best_metric:
            best_metric = eval_results['eval_f1_score']
            best_model = model
            model.save_pretrained(os.path.join(paths.path_result, f'best_model_{model_name}'))  # Save the best model

        results.append(eval_results)
        print(f"Results for fold {fold + 1}: {eval_results}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(paths.path_result, f"evaluation_results_{model_name}.csv"), index=False)

    # Calculate mean and std for the metrics
    mean_results = results_df.mean()
    std_results = results_df.std()
    summary_df = pd.DataFrame({'mean': mean_results, 'std': std_results})
    summary_df.to_csv(paths.path_result + f"results_summary_{model_name}.csv", index=True)

    return best_model


# Load and predict using the best model
def load_model_and_predict(model_path, tokenizer, texts):
    model = AutoModelForSequenceClassification.from_pretrained(model_path + 'best_model_bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, axis=-1)
    output = []
    for text, pred_label in zip(texts, predictions):
        output.append(
            {'text': text,
             'label': int(pred_label)}
        )
    return output, predictions.tolist()
