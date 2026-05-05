import re
import string
from config import Config
import pandas as pd 

config = Config()

class TextPreprocessor:
    dataset = pd.read_excel(config.DATASET_PATH)
    SPACE_RE = re.compile(r"\s+")
    NONWORD_RE= re.compile(r"[^a-z0-9\s]")
    
    @classmethod
    def handle_dataset(cls):
        print("\nLoading & preprocessing data …")
        print(f"  Dataset size : {len(cls.dataset):,}")
        print(f"  Label distribution:\n{cls.dataset['Toxic Category'].value_counts().to_string()}")

        query_df = cls.dataset[['query', 'Toxic Category']].copy().dropna()
        query_df.columns = ['text_content', 'label'] 

        desc_df = cls.dataset[['image descriptions', 'Toxic Category']].copy().dropna()
        desc_df.columns = ['text_content', 'label']

        df = pd.concat([query_df, desc_df], ignore_index=True)

        print(df.head())
        print(f"Total training examples: {len(df)}")
        return df

    def get_data_info(self,df):

        return {
            "dataset name": config.DATASET_PATH,
            "dataset size": df.size,
            "columns": df.columns,
            "num of empty rows ": df.isnull().sum()
        }
    @classmethod
    def clean_text(cls, text:str)->str:

        if not isinstance(text,str):
            return ""

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = cls.SPACE_RE.sub(" ", text)
        text = cls.NONWORD_RE.sub(" ", text)
        return text.strip()


    @classmethod
    def clean_batch(cls, texts) -> list:
        return [cls.clean_text(t) for t in texts]