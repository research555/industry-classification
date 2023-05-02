import unittest
import pandas as pd
from src.data.tagging import Embedding

# Sample data for testing
startups_data = {
    'id': [1, 2, 3],
    'cb_description': [
        "Startup 1 is focused on AI and machine learning.",
        "Startup 2 is working on blockchain technology.",
        "Startup 3 is developing a new social media platform."
    ]
}

industries_data = {
    'id': [1, 2, 3],
    'keywords': [
        "Artificial Intelligence, Machine Learning, AI",
        "Blockchain, Cryptocurrency",
        "Social Media, Platform"
    ],
    'industry': ['AI', 'Blockchain', 'Social Media']
}

startups = pd.DataFrame(startups_data)
industries = pd.DataFrame(industries_data)


class TestEmbedding(unittest.TestCase):

    def test_init(self):
        emb = Embedding(startups, industries)
        self.assertIsNotNone(emb)
        self.assertEqual(emb.llm['bert'], 'bert-base-uncased')

    def test_generate_embeddings(self):
        emb = Embedding(startups, industries)
        emb.generate_embeddings(startup=True)
        emb.generate_embeddings(startup=False)
        self.assertTrue('embeddings' in emb.startups.columns)
        self.assertTrue('embeddings' in emb.industries.columns)

    def test_assign_industry(self):
        emb = Embedding(startups, industries)
        emb.generate_embeddings(startup=True)
        emb.generate_embeddings(startup=False)
        assigned_industries = emb.assign_industry(num_labels=3)
        self.assertIsNotNone(assigned_industries)
        self.assertEqual(len(assigned_industries), len(startups))

    def test_pooling(self):
        emb = Embedding(startups, industries)
        emb.generate_embeddings(startup=True)
        emb.generate_embeddings(startup=False)
        self.assertIsNotNone(emb.pooled_embeds)

    def test_update_dataframe(self):
        emb = Embedding(startups, industries)
        emb.generate_embeddings(startup=True)
        emb.generate_embeddings(startup=False)
        emb.assign_industry(num_labels=3)
        updated_df = emb.update_dataframe()
        self.assertIsNotNone(updated_df)
        self.assertTrue('industry1' in updated_df.columns)
        self.assertTrue('score1' in updated_df.columns)


if __name__ == '__main__':
    unittest.main()