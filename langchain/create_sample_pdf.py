#!/usr/bin/env python3
"""
Script to create a sample PDF file for testing the PDF folder loader.
"""

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import os
    
    def create_sample_pdf():
        """Create a sample PDF file."""
        pdf_path = "Files/pdf/sample_document.pdf"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        # Create PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 100, "Sample PDF Document")
        
        # Add content
        c.setFont("Helvetica", 12)
        y_position = height - 150
        
        content = [
            "This is a sample PDF document created for testing the PDF folder loader.",
            "",
            "Features of the PDF Loader:",
            "• Processes PDF files using Docling library",
            "• Extracts text content with OCR support",
            "• Generates vector embeddings",
            "• Stores documents in Weaviate database",
            "",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
            "",
            "This document will be processed into chunks and stored as embeddings",
            "for efficient search and retrieval operations."
        ]
        
        for line in content:
            c.drawString(100, y_position, line)
            y_position -= 20
        
        # Save PDF
        c.save()
        print(f"✅ Sample PDF created: {os.path.abspath(pdf_path)}")
        return pdf_path
    
    if __name__ == "__main__":
        create_sample_pdf()
        
except ImportError:
    print("❌ reportlab not installed. Installing...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        print("✅ reportlab installed successfully")
        print("Please run this script again to create the sample PDF.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install reportlab: {e}")
        print("You can install it manually with: pip install reportlab")