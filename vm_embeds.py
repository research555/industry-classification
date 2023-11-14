import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from tqdm import tqdm_notebook as tqdm
from sentence_transformers import SentenceTransformer
import torch


class Embedding:
    """
    A class to generate embeddings for startups and industries using specified language models and pooling methods.
    """

    def __init__(self, startups, industries, llm='bert', pool='max', sentence_transformer=False, sent='all-MiniLM-L6-v2'):

        """
        Initializes the Embedding class with specified language models and pooling methods.

        :param startups: DataFrame containing startup data with 'id' and 'cb_description' columns
        :param industries: DataFrame containing industry data with 'id' and 'keywords' columns
        :param llm: string, the language model to use for generating embeddings, default is 'bert'
        :param pool: string, the pooling method to use for generating embeddings, default is 'max'
        :param sentence_transformer: bool, whether to use a sentence transformer model, default is False
        """

        self.startups = startups
        self.industries = industries
        self.sentence_transformer = sentence_transformer
        self.pool = pool
        self.llm = {
            'bert': 'bert-base-uncased',
            'gpt2': 'gpt2',
            'gpt': 'openai-gpt',
            'roberta': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'xlnet': 'xlnet-base-uncased',
            'electra': 'google/electra-base-discriminator',
            'industry_classifier': 'sampathkethineedi/industry-classification'
        }
        if not sentence_transformer:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.llm[llm])
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm[llm])
        else:
            self.model = SentenceTransformer(sent)


    def generate_embeddings(self, startup=True):
        """
        Generates embeddings for startups or industries using the specified language model and pooling method.

        :param startup: bool, if True, generates embeddings for startups, if False, generates embeddings for industries
        :return: DataFrame with generated embeddings merged with the original input DataFrame
        """
        texts = self.startups if startup else self.industries
        embeddings_list = []

        for i, row in tqdm(texts.iterrows()):
            id = row['id']
            if startup:
                description = row['cb_description']
            else:
                description = row['keywords']
            if self.sentence_transformer:
                embeddings = self.model.encode(description)
            else:
                inputs = self.tokenizer.encode_plus(description, return_tensors="pt", truncation=True, padding="max_length", max_length=60)
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                embeddings = self.pooling(last_hidden_states)

            embeddings_list.append({'id': id, 'embeddings': embeddings.tolist()})

        embeddings_df = pd.DataFrame(embeddings_list)
        merged_df = pd.merge(texts, embeddings_df, on='id', how='left')

        if startup:
            self.startups = merged_df
        else:
            self.industries = merged_df

        return merged_df


    def assign_industry(self, num_labels=3):
        """
        Assigns top industries to startups based on their cosine similarity to the industry embeddings.

        :param num_labels: int, the number of top industries to assign to each startup, default is 3
        :return: list of lists containing dictionaries with assigned industries and their similarity scores
        """
        self.assigned_industries = []
        for startup_embedding in self.startups['embeddings']:
            startup_embedding = np.array(startup_embedding).flatten()
            industry_embeddings = np.array([np.array(x).flatten() for x in self.industries['embeddings']])

            similarities = cosine_similarity([startup_embedding], industry_embeddings)[0]
            top_industry_indices = np.argsort(similarities)[-num_labels:][::-1]
            top_industries = [{'industry': self.industries.iloc[index]['industry'], 'score': similarities[index]} for index in top_industry_indices]

            self.assigned_industries.append(top_industries)

        return self.assigned_industries

    def pooling(self, last_hidden_states):
        """
        Applies the specified pooling method to the given last hidden states tensor.

        :param last_hidden_states: tensor, the last hidden states from the language model
        :return: NumPy array of pooled embeddings
        """
        if self.pool == 'max':
            self.pooled_embeds = torch.max(last_hidden_states, dim=1).values
        elif self.pool == 'avg':
            self.pooled_embeds = torch.mean(last_hidden_states, dim=1)
        elif self.pool == 'concat':
            max_pooling = torch.max(last_hidden_states, dim=1).values
            average_pooling = torch.mean(last_hidden_states, dim=1)
            self.pooled_embeds = torch.cat((max_pooling, average_pooling), dim=1)
        else:
            raise ValueError('pool must be either max, avg or concat')
        return self.pooled_embeds.detach().numpy()

    def update_dataframe(self):
        """
        Updates the startup and industry DataFrames with assigned industries and their similarity scores.

        :return: DataFrame with updated startups data
        """
        max_industries = max([len(x) for x in self.assigned_industries])

        for i in range(max_industries):
            self.startups[f'industry{i + 1}'] = [x[i]['industry'] if i < len(x) else None for x in self.assigned_industries]
            self.startups[f'score{i + 1}'] = [x[i]['score'].round(3) if i < len(x) else None for x in self.assigned_industries]

        self.startups.drop(columns=['embeddings'], inplace=True)
        self.industries.drop(columns=['embeddings'], inplace=True)

        return self.startups



industry_data = pd.read_csv(r'C:\Users\imran\DataspellProjects\WalidCase\data\processed\GPT4_generated_keywords.csv')
industry_data.insert(0, 'id', industry_data.index)
startups = pd.read_csv(r'C:\Users\imran\DataspellProjects\WalidCase\data\processed\full_startups.csv')

mask = startups['cb_description'].apply(lambda x: len(x.split()) > 20)
startups = startups[mask]

embeddings = Embedding(startups, industry_data, sentence_transformer=True)
embeddings.generate_embeddings(startup=True)
embeddings.generate_embeddings(startup=False)
embeddings.assign_industry(num_labels=3)
df = embeddings.update_dataframe()
df.to_csv(r'C:\Users\imran\DataspellProjects\WalidCase\data\tagged\full_startups_industries.csv', index=False)
