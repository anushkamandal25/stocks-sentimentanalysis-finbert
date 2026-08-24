import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FinBertSentimentAnalyzer:
    """
    A class for sentiment analysis of financial news using FinBERT.

    Attributes:
    - tokenizer: Hugging Face tokenizer for the FinBERT model.
    - model: Hugging Face sequence classification model (FinBERT).
    """

    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initializes the FinBertSentimentAnalyzer object and loads FinBERT.

        Args:
        - model_name (str): Hugging Face model identifier.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def preprocess(self, text):
        """
        Strips hashtags, usernames, and URLs from text before scoring.

        Args:
        - text (str): Raw text.

        Returns:
        - str: Cleaned text.
        """
        if text is None:
            return ""
        new_text = []
        for t in text.split(" "):
            t = '' if t.startswith('#') and len(t) > 1 else t
            t = '' if t.startswith('@') and len(t) > 1 else t
            t = '' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given piece of text.

        Args:
        - text (str): News description/summary text.

        Returns:
        - dict: {'positive': float, 'negative': float, 'neutral': float}
        """
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {'positive': scores[0], 'negative': scores[1], 'neutral': scores[2]}

    def add_sentiment_columns(self, df, text_column='Description'):
        """
        Runs sentiment analysis over every row of a DataFrame and appends score columns.

        Args:
        - df (pd.DataFrame): News DataFrame.
        - text_column (str): Column containing text to score.

        Returns:
        - pd.DataFrame: Same DataFrame with Sent_positive, Sent_negative, Sent_neutral columns added.
        """
        tqdm.pandas()
        sentiments = df[text_column].progress_apply(self.analyze_sentiment)
        df['Sent_positive'] = sentiments.apply(lambda x: x['positive'])
        df['Sent_negative'] = sentiments.apply(lambda x: x['negative'])
        df['Sent_neutral'] = sentiments.apply(lambda x: x['neutral'])
        return df


if __name__ == '__main__':
    # Example Usage:
    analyzer = FinBertSentimentAnalyzer()
    result = analyzer.analyze_sentiment("Apple beats earnings expectations this quarter.")
    print(result)