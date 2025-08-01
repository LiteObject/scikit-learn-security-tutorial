#!/usr/bin/env python3
"""
Simple examples to test individual components of the tutorial
"""


def test_data_generation():
    """Test the data generation component"""
    print("🧪 Testing Data Generation...")

    from data_generator import NetworkDataGenerator

    generator = NetworkDataGenerator()

    # Test feature extraction
    test_cases = [
        [22, 80, 443],        # Web server
        [23, 21, 3389, 445],  # Vulnerable device
        [80],                 # Simple IoT
        [1337, 31337, 8080]   # Unusual ports
    ]

    for i, ports in enumerate(test_cases, 1):
        features = generator.extract_features(ports)
        # Show first 5 features
        print(f"   Test {i}: {ports} -> Features: {features[:5]}...")

    print("✅ Data generation test complete!\n")


def test_model_training():
    """Test model training with a small dataset"""
    print("🧪 Testing Model Training...")

    from model_trainer import NetworkSecurityMLTutorial

    # Create tutorial instance
    tutorial = NetworkSecurityMLTutorial()

    # Generate small dataset
    X, y_device, y_risk = tutorial.data_generator.generate_dataset(
        samples_per_class=20)

    # Split data
    from sklearn.model_selection import train_test_split
    tutorial.X_train, tutorial.X_test, tutorial.y_device_train, tutorial.y_device_test, \
        tutorial.y_risk_train, tutorial.y_risk_test = train_test_split(
            X, y_device, y_risk, test_size=0.2, random_state=42, stratify=y_device
        )

    # Test device classifier
    accuracy = tutorial.step_2_build_device_classifier()
    print(f"   Device classifier accuracy: {accuracy:.3f}")

    # Test anomaly detector
    num_anomalies, anomaly_rate = tutorial.step_3_build_anomaly_detector()
    print(
        f"   Anomaly detection: {num_anomalies} anomalies ({anomaly_rate:.1f}%)")

    # Test risk predictor
    mae, r2 = tutorial.step_4_build_risk_predictor()
    print(f"   Risk prediction R²: {r2:.3f}")

    print("✅ Model training test complete!\n")


def test_custom_prediction():
    """Test prediction on custom data"""
    print("🧪 Testing Custom Predictions...")

    from model_trainer import NetworkSecurityMLTutorial
    import numpy as np

    # Quick training
    tutorial = NetworkSecurityMLTutorial()
    X, y_device, y_risk = tutorial.data_generator.generate_dataset(
        samples_per_class=30)

    from sklearn.model_selection import train_test_split
    tutorial.X_train, tutorial.X_test, tutorial.y_device_train, tutorial.y_device_test, \
        tutorial.y_risk_train, tutorial.y_risk_test = train_test_split(
            X, y_device, y_risk, test_size=0.2, random_state=42, stratify=y_device
        )

    # Train models quickly
    tutorial.step_2_build_device_classifier()
    tutorial.step_3_build_anomaly_detector()
    tutorial.step_4_build_risk_predictor()

    # Test custom devices
    test_devices = [
        ("Web Server", [22, 80, 443]),
        ("Suspicious Device", [23, 21, 3389, 445]),
        ("IoT Device", [80])
    ]

    for name, ports in test_devices:
        features = tutorial.data_generator.extract_features(ports)
        features_array = np.array([features])
        features_scaled = tutorial.feature_scaler.transform(features_array)

        device_pred = tutorial.device_classifier.predict(features_scaled)[0]
        device_name = tutorial.data_generator.get_device_name(device_pred)
        risk_score = tutorial.risk_predictor.predict(features_scaled)[0]

        print(f"   {name}: Predicted as {device_name}, Risk: {risk_score:.3f}")

    print("✅ Custom prediction test complete!\n")


def main():
    """Run all tests"""
    print("🎓 SCIKIT-LEARN TUTORIAL COMPONENT TESTS")
    print("=" * 50)

    try:
        test_data_generation()
        test_model_training()
        test_custom_prediction()

        print("🎉 All tests passed! Tutorial is ready to use.")
        print("\n📚 Run the full tutorial with:")
        print("   python network_security_ml.py")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please run 'python setup.py' first to install dependencies.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Please check your installation.")


if __name__ == "__main__":
    main()
