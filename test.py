import unittest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib

class TestModelIntegrity(unittest.TestCase):

    def setUp(self):
        # Setup: Load the trained model before each test
        self.model = joblib.load('model/model.pkl')  

        # Load the real test data from a CSV file
        test_data = pd.read_csv('data/interim/test.csv')  
        self.X_test = test_data.drop('Approved_Flag', axis=1)  
        
        # Label encode the target variable 'Approved_Flag'
        self.le = LabelEncoder()
        self.y_true = self.le.fit_transform(test_data['Approved_Flag'])  # Encode the labels as numbers

    def test_input_shape(self):
        """Test the model's input shape."""
        # Get the expected number of features from the input data shape
        expected_n_features = self.X_test.shape[1]  # Automatically gets the number of features from X_test

        self.assertEqual(self.X_test.shape[1], expected_n_features, 
                         f"Expected input shape (n_samples, {expected_n_features}), but got {self.X_test.shape[1]} features.")

    def test_accuracy(self):
        """Test the accuracy of the model."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_true, y_pred)
        self.assertGreaterEqual(accuracy, 0.90, f"Accuracy is below 90%. Current accuracy: {accuracy}")

    def test_precision(self):
        """Test the precision of the model."""
        y_pred = self.model.predict(self.X_test)
        precision = precision_score(self.y_true, y_pred, average='weighted')
        self.assertGreaterEqual(precision, 0.90, f"Precision is below 90%. Current precision: {precision}")

    def test_recall(self):
        """Test the recall of the model."""
        y_pred = self.model.predict(self.X_test)
        recall = recall_score(self.y_true, y_pred, average='weighted')
        self.assertGreaterEqual(recall, 0.90, f"Recall is below 90%. Current recall: {recall}")

    def test_full_model(self):
        """Test the full model (all checks in one test)."""
        self.test_input_shape()
        self.test_accuracy()
        self.test_precision()
        self.test_recall()

if __name__ == '__main__':
    unittest.main()
