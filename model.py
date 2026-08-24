from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ['sentiment_lag1', 'sentiment_ma3', 'sentiment_momentum', 'news_count']


class DirectionModel:
    """
    A class for training and evaluating a logistic regression model
    that predicts next-day price direction from sentiment features.

    Attributes:
    - clf (LogisticRegression): The trained classifier (None until fit).
    """

    def __init__(self):
        """
        Initializes the DirectionModel object.
        """
        self.clf = None

    def train_test_split_by_time(self, merged_df, train_frac=0.8):
        """
        Splits data chronologically into train/test sets (no shuffling for time-series data).

        Args:
        - merged_df (pd.DataFrame): Feature-engineered DataFrame.
        - train_frac (float): Fraction of rows used for training.

        Returns:
        - tuple: (X_train, X_test, y_train, y_test, split_index)
        """
        split = int(len(merged_df) * train_frac)
        X, y = merged_df[FEATURES], merged_df['direction']
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], split

    def train(self, X_train, y_train):
        """
        Fits a logistic regression classifier.

        Args:
        - X_train (pd.DataFrame): Training features.
        - y_train (pd.Series): Training labels.

        Returns:
        - LogisticRegression: The fitted classifier.
        """
        self.clf = LogisticRegression(class_weight='balanced')
        self.clf.fit(X_train, y_train)
        return self.clf

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluates the classifier against a majority-class baseline.

        Args:
        - X_train, y_train: Training data (used to compute baseline).
        - X_test, y_test: Test data.

        Returns:
        - np.ndarray: Predicted labels for the test set.
        """
        baseline_pred = [y_train.mode()[0]] * len(y_test)
        preds = self.clf.predict(X_test)

        print("Baseline accuracy:", accuracy_score(y_test, baseline_pred))
        print("Model accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        return preds


if __name__ == '__main__':
    # Example Usage:
    # dm = DirectionModel()
    # X_train, X_test, y_train, y_test, split = dm.train_test_split_by_time(merged)
    # dm.train(X_train, y_train)
    # preds = dm.evaluate(X_train, y_train, X_test, y_test)
    pass