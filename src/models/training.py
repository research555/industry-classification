from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.model_selection import train_test_split
import pandas as pd



class FineTuneSentenceTransfomer:
    def __init__(self, startups, industries, label_count_threshold=2, sentence_transformer='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(sentence_transformer)
        self.data = self.merge_features(startups, industries, label_count_threshold)
        self.loss = losses.MultipleNegativesRankingLoss(self.model)

    def merge_features(self, startups, industries, label_count_threshold=3):

        merged_df = pd.merge(startups, industries, left_on='industry1', right_on='industry', how='left')
        merged_df = merged_df[['cb_description', 'industry1', 'keywords', 'id_y']].dropna()
        merged_df = merged_df.groupby('industry1').filter(lambda x : len(x)>label_count_threshold)
        merged_df['id_y'] = merged_df['id_y'].astype(int)
        self.merged_df = merged_df.rename(columns={'industry1': 'industry', 'cb_description': 'description', 'id_y': 'industry_id'})
        return self.merged_df


    def split_data(self):

        descriptions = self.merged_df['description']
        keywords = self.merged_df['keywords']

        descriptions_train, self.descriptions_test, keywords_train, self.keywords_test = train_test_split(descriptions, keywords, test_size=0.15, random_state=42, stratify=keywords)
        self.descriptions_train, self.descriptions_val, self.keywords_train, self.keywords_val = train_test_split(descriptions_train, keywords_train, test_size=0.1765, random_state=42, stratify=keywords_train)

        return self.keywords_train, self.keywords_val, self.keywords_test, self.descriptions_train, self.descriptions_val, self.descriptions_test

    def prepare_examples(self, descriptions, keywords):
        examples = []
        for i in range(len(descriptions)):
            examples.append(InputExample(texts=[descriptions[i]], label=keywords[i]))
        return examples

    def prepare_dataloader(self, examples, batch_size=16):
        return DataLoader(examples, shuffle=True, batch_size=batch_size)

    def prepare_evaluator(self):

        val_sentences1 = [description for description, keyword in zip(self.descriptions_val, self.keywords_val)]
        val_sentences2 = [keyword for description, keyword in zip(self.descriptions_val, self.keywords_val)]
        val_scores = [1] * len(val_sentences1)
        self.evaluator = EmbeddingSimilarityEvaluator(val_sentences1, val_sentences2, val_scores)

        return self.evaluator

    def prepare_loss(self):
        self.loss = losses.CosineSimilarityLoss(self.model)
        return self.loss

    def train(self,
              train_dataloader,
              train_loss,
              epochs=5,
              output_path=r'C:\Users\imran\DataspellProjects\WalidCase\models\finetuned_sentence_transformer_1',
              warmup_steps=500,
              evaluation_steps=500,
              weight_decay=0.01,
              max_grad_norm=1.0,
              save_best_model=True,
              checkpoint_save_steps=500,
              checkpoint_path=r'C:\Users\imran\DataspellProjects\WalidCase\models\finetuned_sentence_transformer_1\checkpoint'
              ):

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=self.evaluator,
                       epochs=epochs,
                       evaluation_steps=evaluation_steps,
                       warmup_steps=warmup_steps,
                       output_path=output_path,
                       weight_decay=weight_decay,
                       checkpoint_path=checkpoint_path,
                       checkpoint_save_steps=checkpoint_save_steps,
                       max_grad_norm=max_grad_norm,
                       save_best_model=save_best_model,
                       )
        return self.model


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.utils.data DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class FineTuneTransfomer:

    def __init__(self, startups, industries, label_count_threshold=2, sentence_transformer=''):
        self.model = (sentence_transformer)
        self.data = self.merge_features(startups, industries, label_count_threshold)
        self.loss = losses.MultipleNegativesRankingLoss(self.model)

    def merge_features(self, startups, industries, label_count_threshold=3):

        merged_df = pd.merge(startups, industries, left_on='industry1', right_on='industry', how='left')
        merged_df = merged_df[['cb_description', 'industry1', 'keywords', 'id_y']].dropna()
        merged_df = merged_df.groupby('industry1').filter(lambda x : len(x)>label_count_threshold)
        merged_df['id_y'] = merged_df['id_y'].astype(int)
        self.merged_df = merged_df.rename(columns={'industry1': 'industry', 'cb_description': 'description', 'id_y': 'industry_id'})
        return self.merged_df


    def split_data(self):

        descriptions = self.merged_df['description']
        keywords = self.merged_df['keywords']

        descriptions_train, self.descriptions_test, keywords_train, self.keywords_test = train_test_split(descriptions, keywords, test_size=0.15, random_state=42, stratify=keywords)
        self.descriptions_train, self.descriptions_val, self.keywords_train, self.keywords_val = train_test_split(descriptions_train, keywords_train, test_size=0.1765, random_state=42, stratify=keywords_train)

        return self.keywords_train, self.keywords_val, self.keywords_test, self.descriptions_train, self.descriptions_val, self.descriptions_test



from sklearn.preprocessing import LabelEncoder



class FineTuneTransformer:
    def __init__(self, num_labels,  startups, industries, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.data = self.merge_features(startups, industries)



    def merge_features(self, startups, industries, label_count_threshold=3):

        merged_df = pd.merge(startups, industries, left_on='industry1', right_on='industry', how='left')
        merged_df = merged_df[['cb_description', 'industry1', 'keywords', 'id_y']].dropna()
        merged_df = merged_df.groupby('industry1').filter(lambda x : len(x)>label_count_threshold)
        merged_df['id_y'] = merged_df['id_y'].astype(int)
        self.data = merged_df.rename(columns={'industry1': 'industry', 'cb_description': 'description', 'id_y': 'industry_id'})

        return self.data

    def __encode_labels(self):
        encoder = LabelEncoder()
        self.data['industry_id'] = encoder.fit_transform(self.data['keywords'])
        return self.data

    def __split_data(self):
        features = self.data['description']
        labels = self.data['industry_id']

        features_train, self.features_test, labels_train, self.labels_test = train_test_split(features, labels, test_size=0.15, random_state=42, stratify=labels)
        self.features_train, self.features_val, self.labels_train, self.labels_val = train_test_split(features_train, labels_train, test_size=0.1765, random_state=42, stratify=labels_train)

        return self.features_train, self.features_val, self.features_test, self.labels_train, self.labels_val, self.labels_test



    def tokenize_data(self, texts, labels):
        encoded_data = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            max_length=60,
            return_tensors='pt'
        )
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels

    def create_dataloader(self, input_ids, attention_masks, labels, batch_size=16):
        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader

    def train(self, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5, warmup_steps=500):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * epochs)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_dataloader:
                input_ids, attention_masks, labels = tuple(b.to(self.device) for b in batch)
                self.model.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

            self.evaluate(val_dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        predictions, true_labels = [], []

        for batch in dataloader:
            input_ids, attention_masks, labels = tuple(b.to(self.device) for b in batch)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_masks)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            predictions.extend(list(np.argmax(logits, axis=1)))
            true_labels.extend(list(label_ids))

        acc = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
