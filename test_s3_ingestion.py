#!/usr/bin/env python3
"""
Test script for S3 ingestion functionality.
This script demonstrates how to use the new S3 ingestion feature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.ingestion_data import (
    main_s3_ingestion,
    validate_json_data,
    create_product_text_from_json,
    create_product_metadata_from_json
)

def test_json_validation():
    """Test JSON validation functionality."""
    print("=== Testing JSON Validation ===")

    # Valid JSON data
    valid_data = {
        "document_id": "test_product.json",
        "product": "FORMULADO",
        "class": "Regulador de crescimento",
        "formulation_type": "Concentrado emulsionável (EC)",
        "dose": "0,8 a 1,2 L/ha",
        "usage_mode": "Aplicar via pulverização terrestre..."
    }

    # Invalid JSON data (missing required fields)
    invalid_data = {
        "class": "Regulador de crescimento",
        "formulation_type": "Concentrado emulsionável (EC)"
    }

    print(f"Valid data validation: {validate_json_data(valid_data)}")
    print(f"Invalid data validation: {validate_json_data(invalid_data)}")

def test_text_creation():
    """Test text creation from JSON data."""
    print("\n=== Testing Text Creation ===")

    test_data = {
        "document_id": "- Maxapac 250 SC; (Bula).json",
        "product": "FORMULADO",
        "class": "Regulador de crescimento",
        "formulation_type": "Concentrado emulsionável (EC)",
        "dose": "0,8 a 1,2 L/ha para cana-de-açúcar",
        "usage_mode": "MAXAPAC 250 EC é um regulador de crescimento sistêmico..."
    }

    text = create_product_text_from_json(test_data)
    metadata = create_product_metadata_from_json(test_data, 0)

    print(f"Generated text:\n{text}")
    print(f"\nGenerated metadata:\n{metadata}")

def main():
    """Run tests and demonstrate S3 ingestion usage."""
    print("=== S3 Ingestion Test Script ===\n")

    # Run validation tests
    test_json_validation()
    test_text_creation()

    print("\n=== S3 Ingestion Usage Examples ===")
    print("To run S3 ingestion, use one of these commands:")
    print("1. python src/ingestion/ingestion_data.py s3 your-bucket-name")
    print("2. python src/ingestion/ingestion_data.py s3 your-bucket-name your/prefix/")
    print("\nMake sure you have AWS credentials configured (AWS CLI, environment variables, or IAM role)")

if __name__ == "__main__":
    main()