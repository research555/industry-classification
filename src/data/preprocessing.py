from tqdm import tqdm_notebook as tqdm
import spacy
import pandas as pd



class TextProcessing:
    def __init__(self, df: pd.DataFrame = None, industry=False, startup=False):
        self.nlp = spacy.load("en_core_web_sm")
        self.startups = pd.DataFrame([])
        self.industries = pd.DataFrame([])
        if startup:
            self.startups = df.copy()
        elif industry:
            self.industries = df.copy()
        else:
            raise ValueError("Please specify if the data is for startups or industries")

    def __iterate_rows(self):
        df = self.startups if not self.startups.empty else self.industries
        for index, row in tqdm(df.iterrows()):
            self.index = index
            if not self.industries.empty:
                self.about_us = row["keywords"]
            else:
                self.about_us = row["cb_description"]
            yield self

    def length_range(self, data, length_range=(30, 150)):

        self.startups = data.copy()
        self.startups.dropna(inplace=True)
        for i, row in self.startups.iterrows():
            length = len(row['cb_description'].split())
            if length < 15 or length > 60:
                self.startups.drop(i, inplace=True)

        return self.startups

    def remove_non_english_tokens(self, data=None):
        if data is not None:
            if not self.industries.empty:
                self.industries = data.copy()
            else:
                self.startups = data.copy()
        english_tokens = []
        for description in self.__iterate_rows():
            doc = self.nlp(self.about_us)
            tokens = [token.text for token in doc if token.lang_ == 'en' and token.is_alpha]
            self.about_us = " ".join(tokens)
            english_tokens.append(self.about_us)

        if not self.startups.empty:
            self.startups['cb_description'].replace(to_replace=self.startups['cb_description'].unique(),
                                                    value=english_tokens, inplace=True)
            return self.startups

        else:
            self.industries['keywords'].replace(to_replace=self.industries['keywords'].unique(), value=english_tokens,
                                                inplace=True)
            return self.industries

    def remove_noisy_tokens(self, data=None):
        if data is not None:
            if not self.industries.empty:
                self.industries = data.copy()
            else:
                self.startups = data.copy()

        cleaned_about_us = []
        for item in self.__iterate_rows():
            doc = self.nlp(self.about_us)
            for ent in doc.ents:
                if ent.label_:
                    self.about_us = self.about_us.replace(ent.text, "")
            cleaned_doc = self.nlp(self.about_us)

            tokens = [token.text.lower() for token in cleaned_doc if
                      not token.is_stop
                      and not token.is_punct
                      and not token.is_space
                      and not token.like_num
                      and not token.is_digit
                      and not token.is_currency
                      and not token.is_bracket
                      and not token.is_quote
                      and not token.is_left_punct
                      and not token.is_right_punct
                      and not token.like_url
                      and not token.like_email]

            self.about_us = " ".join(tokens)
            cleaned_about_us.append(self.about_us)
        if not self.industries.empty:
            self.industries['keywords'].replace(to_replace=self.industries['keywords'].unique(), value=cleaned_about_us,
                                                inplace=True)
            return self.industries
        else:
            self.startups['cb_description'].replace(to_replace=self.startups['cb_description'].unique(),
                                                    value=cleaned_about_us, inplace=True)
            return self.startups

    def lemma(self, data=None):
        if data is not None:
            if not self.industries.empty:
                self.industries = data.copy()
            else:
                self.startups = data.copy()
        lemmatized_about_us = []
        for description in self.__iterate_rows():
            doc = self.nlp(self.about_us)
            tokens = [token.lemma_ for token in doc]
            self.about_us = " ".join(tokens)
            lemmatized_about_us.append(" ".join(tokens))
        if not self.industries.empty:
            self.industries['keywords'].replace(to_replace=self.industries['keywords'].unique(),
                                                value=lemmatized_about_us, inplace=True)
            return self.industries
        else:
            self.startups['cb_description'].replace(to_replace=self.startups['cb_description'].unique(),
                                                    value=lemmatized_about_us, inplace=True)
            return self.startups

    @staticmethod
    def make_keywords_unique(df): # very ugly function but its okay
        unique_keywords = set()
        for index, row in df.iterrows():
            keywords = row['keywords'].split()
            unique_keywords.update(set(keywords))
        appended_keywords = []
        for index, row in df.iterrows():
            keywords = [keyword for keyword in row['keywords'].split() if keyword in unique_keywords and keyword not in appended_keywords]
            appended_keywords.extend(keywords)
            new_keys = ' '.join(keywords)
            df.at[index, 'keywords'] = new_keys
        return df



