import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, validation_curve
import pandas as pd


class ModelEvaluator:
    """Advanced model evaluation and visualization"""

    def __init__(self, tutorial_instance):
        self.tutorial = tutorial_instance
        self.setup_plotting()

    def setup_plotting(self):
        """Setup matplotlib for pretty plots"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_confusion_matrix(self):
        """Visualize classification results"""
        print("📊 Creating Confusion Matrix...")

        # Get predictions
        X_test_scaled = self.tutorial.feature_scaler.transform(
            self.tutorial.X_test)
        predictions = self.tutorial.device_classifier.predict(X_test_scaled)

        # Create confusion matrix
        cm = confusion_matrix(self.tutorial.y_device_test, predictions)
        device_names = [
            self.tutorial.data_generator.get_device_name(i) for i in range(6)]

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=device_names, yticklabels=device_names)
        plt.title('Device Classification Confusion Matrix')
        plt.ylabel('Actual Device Type')
        plt.xlabel('Predicted Device Type')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """Visualize which features are most important"""
        print("🔥 Plotting Feature Importance...")

        feature_names = [
            "Total Ports", "Has SSH", "Has HTTP", "Has HTTPS",
            "Has Telnet", "Has RDP", "Has SMB", "Has FTP",
            "Port Spread", "High Ports"
        ]

        importances = self.tutorial.device_classifier.feature_importances_

        # Create plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]  # Sort by importance

        plt.bar(range(len(importances)), importances[indices])
        plt.title('Feature Importance in Device Classification')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(importances)), [
                   feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_risk_predictions(self):
        """Visualize risk prediction accuracy"""
        print("⚡ Plotting Risk Prediction Results...")

        X_test_scaled = self.tutorial.feature_scaler.transform(
            self.tutorial.X_test)
        risk_predictions = self.tutorial.risk_predictor.predict(X_test_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.tutorial.y_risk_test, risk_predictions, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)  # Perfect prediction line
        plt.xlabel('Actual Risk Score')
        plt.ylabel('Predicted Risk Score')
        plt.title('Risk Prediction Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self):
        """Show how model performance improves with more data"""
        print("📈 Creating Learning Curves...")

        X_train_scaled = self.tutorial.feature_scaler.transform(
            self.tutorial.X_train)

        train_sizes, train_scores, val_scores = learning_curve(
            self.tutorial.device_classifier, X_train_scaled, self.tutorial.y_device_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1),
                 'o-', label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1),
                 'o-', label='Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves - Device Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_comprehensive_report(self):
        """Create a complete analysis report"""
        print("\n📋 COMPREHENSIVE MODEL ANALYSIS REPORT")
        print("=" * 60)

        # Run all visualizations
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_risk_predictions()
        self.plot_learning_curves()

        print("✅ All visualizations complete!")
        print("💡 Key insights to look for:")
        print("   🎯 Confusion Matrix: Which device types get confused?")
        print("   🔥 Feature Importance: Which network patterns matter most?")
        print("   ⚡ Risk Predictions: How accurate are our risk assessments?")
        print("   📈 Learning Curves: Do we need more training data?")


# Usage example
if __name__ == "__main__":
    from model_trainer import NetworkSecurityMLTutorial

    # Run the tutorial first
    tutorial = NetworkSecurityMLTutorial()
    tutorial.run_complete_tutorial()

    # Then create advanced analysis
    evaluator = ModelEvaluator(tutorial)
    evaluator.create_comprehensive_report()
