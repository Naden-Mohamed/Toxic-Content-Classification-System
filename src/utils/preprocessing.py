import re
import pandas as pd 

class TextPreprocessor:
    CLEAN_RE = re.compile(r"[^a-z0-9]+")

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._dataset = None
        self._processed_df = None

    def load_dataset(self):
        if self._dataset is None:
            try:
                self._dataset = pd.read_excel(self.dataset_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset: {e}")
        return self._dataset

    def handle_dataset(self):
        if self._processed_df is not None:
            return self._processed_df

        df = self.load_dataset()

        query_df = df[['query', 'Toxic Category']].dropna().copy()
        query_df.columns = ['text_content', 'labels']

        desc_df = df[['image descriptions', 'Toxic Category']].dropna().copy()
        desc_df = (
              desc_df.groupby('image descriptions')['Toxic Category']
              .agg(lambda x: x.mode()[0])
              .reset_index()
          )
        desc_df.columns = ['text_content', 'labels']

        combined_df = pd.concat([query_df, desc_df], ignore_index=True)
        self._processed_df = combined_df
        return combined_df

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = self.CLEAN_RE.sub(" ", text)
        return text.strip()

    def clean_batch(self):
        df = self.handle_dataset()
        df.drop_duplicates(inplace=True)

        df = df.copy()
        df['text_content'] = df['text_content'].apply(self.clean_text)
        return df

    def get_data_info(self):
        df = self.handle_dataset()

        info = {
            "dataset_size": len(df),
            "label_distribution": df['labels'].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }

        return info