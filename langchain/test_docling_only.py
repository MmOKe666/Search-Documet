#!/usr/bin/env python3
"""
Simple test script to verify Docling installation and basic functionality.
"""

import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption


def test_docling_import():
    """Test if Docling can be imported successfully."""
    try:
        print("✅ Docling imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Docling: {e}")
        return False


def test_docling_converter():
    """Test Docling converter initialization."""
    try:
        # Initialize Docling converter with PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        print("✅ Docling converter initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Docling converter: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Testing Docling Installation")
    print("=" * 40)
    
    # Test 1: Import
    if not test_docling_import():
        return
    
    # Test 2: Converter initialization
    if not test_docling_converter():
        return
    
    print("\n🎉 All tests passed!")
    print("Docling is ready to process PDF files.")
    
    # Check for PDF files
    pdf_folder = Path("../Files/pdf")
    if pdf_folder.exists():
        pdf_files = list(pdf_folder.glob("*.pdf"))
        if pdf_files:
            print(f"\n📁 Found {len(pdf_files)} PDF files in {pdf_folder}:")
            for pdf_file in pdf_files:
                print(f"  - {pdf_file.name}")
        else:
            print(f"\n⚠️ No PDF files found in {pdf_folder}")
            print("Add some PDF files to test the full functionality.")
    else:
        print(f"\n⚠️ PDF folder {pdf_folder} does not exist")


if __name__ == "__main__":
    main()