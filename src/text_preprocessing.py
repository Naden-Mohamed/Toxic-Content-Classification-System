import re
import string


class TextPreprocessor:
    SPACE_RE = re.compile(r"\s+")
    NONWORD_RE= re.compile(r"[^a-z0-9\s]")

    def get_data_info(self,df):
        print(df.columns)
        print(df.isnull().sum())
        df.drop(columns=['image description'], inplace=True)
        print(df.duplicated().sum())
        print(df.describe())
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