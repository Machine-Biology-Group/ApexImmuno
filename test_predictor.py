#!/usr/bin/env python3
"""
Test script for anti-inflammatory peptide predictor
Tests the predictor with known sequences from the filtered list
"""

import sys
import numpy as np

# Test sequences from the filtered list (Anti_inflammatory_filtered_list.csv)
# These sequences passed the filtering criteria (Median MIC ≤ 32, Anti-inflammatory prob ≥ ~0.86)
test_sequences = [
    ("SIIKSLAALLTKLAIIVK", 0.872),  # Expected prob from filtered list
    ("KYLLNYTAKLIIKKLAKI", 0.870),
    ("PKLIYKIGALIKAVKKI", 0.864),
    ("GIVIIIILKAL", 0.885),
    ("NLLRIISKITLL", 0.865)
]

def test_predictor():
    """Test the anti-inflammatory predictor"""
    print("="*70)
    print("Testing Anti-inflammatory Peptide Predictor")
    print("="*70)

    try:
        from predict_anti_inflammatory import predict_anti_inflammatory
        print("✓ Successfully imported predict_anti_inflammatory")
    except Exception as e:
        print(f"✗ Failed to import predictor: {e}")
        sys.exit(1)

    # Extract sequences
    sequences = [seq for seq, _ in test_sequences]
    expected_probs = [prob for _, prob in test_sequences]

    print(f"\nTesting with {len(sequences)} sequences...")
    print("-"*70)

    try:
        # Make predictions
        predicted_probs = predict_anti_inflammatory(sequences)
        print("✓ Prediction completed successfully")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Display results
    print("\nResults:")
    print("-"*70)
    print(f"{'Sequence':<20} {'Expected':<12} {'Predicted':<12} {'Diff':<10} {'Status'}")
    print("-"*70)

    total_error = 0
    passed = 0
    tolerance = 0.05  # Allow 5% difference

    for seq, expected, predicted in zip(sequences, expected_probs, predicted_probs):
        diff = abs(predicted - expected)
        total_error += diff

        # Check if within tolerance
        if diff <= tolerance:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"

        print(f"{seq[:20]:<20} {expected:<12.4f} {predicted:<12.4f} {diff:<10.4f} {status}")

    print("-"*70)
    print(f"\nTest Summary:")
    print(f"  Total tests: {len(sequences)}")
    print(f"  Passed: {passed}/{len(sequences)}")
    print(f"  Mean absolute error: {total_error/len(sequences):.4f}")

    # Additional checks
    print(f"\nAdditional Checks:")
    print(f"  All predictions in [0,1]: {all(0 <= p <= 1 for p in predicted_probs)}")
    print(f"  Mean prediction: {np.mean(predicted_probs):.4f}")
    print(f"  Std prediction: {np.std(predicted_probs):.4f}")
    print(f"  Min prediction: {np.min(predicted_probs):.4f}")
    print(f"  Max prediction: {np.max(predicted_probs):.4f}")

    # Overall pass/fail
    print("\n" + "="*70)
    if passed == len(sequences):
        print("✓ ALL TESTS PASSED")
        print("="*70)
        return 0
    else:
        print(f"✗ {len(sequences)-passed} TEST(S) FAILED")
        print("="*70)
        print("\nNote: Some deviation is expected due to:")
        print("  - Model version differences")
        print("  - Random seed variations")
        print("  - Numerical precision")
        print("\nIf predictions are reasonable (>0.80 for these sequences), the predictor is working.")
        return 1

if __name__ == "__main__":
    exit_code = test_predictor()
    sys.exit(exit_code)
