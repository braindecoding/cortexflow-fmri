#!/usr/bin/env python3
"""
Alternative PDF Converter for Methodology
Uses weasyprint with better error handling
"""

import markdown
import weasyprint
from datetime import datetime
import re

def convert_to_pdf():
    """Convert methodology to PDF using weasyprint"""
    
    print("📄 CONVERTING METODOLOGI TO PDF")
    print("=" * 50)
    
    try:
        # Read markdown file
        with open('METODOLOGI.md', 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        print("✅ Markdown file loaded")
        
        # Convert markdown to HTML
        md = markdown.Markdown(extensions=['tables', 'toc', 'codehilite', 'fenced_code'])
        html_body = md.convert(md_content)
        
        print("✅ Markdown converted to HTML")
        
        # Create complete HTML document with Indonesian styling
        html_document = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metodologi Penelitian CortexFlow</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm;
            @bottom-right {{
                content: "Halaman " counter(page);
                font-size: 10pt;
                color: #666;
            }}
        }}
        
        body {{
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #333;
            text-align: justify;
        }}
        
        .title-page {{
            text-align: center;
            margin-top: 5cm;
            page-break-after: always;
        }}
        
        .title-page h1 {{
            font-size: 20pt;
            font-weight: bold;
            margin-bottom: 1cm;
            color: #2c3e50;
        }}
        
        .title-page h2 {{
            font-size: 16pt;
            font-weight: normal;
            font-style: italic;
            margin-bottom: 2cm;
            color: #34495e;
        }}
        
        .title-page .date {{
            font-size: 14pt;
            margin-top: 3cm;
        }}
        
        h1 {{
            font-size: 18pt;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 2em;
            margin-bottom: 1em;
            page-break-before: auto;
        }}
        
        h2 {{
            font-size: 16pt;
            font-weight: bold;
            color: #34495e;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }}
        
        h3 {{
            font-size: 14pt;
            font-weight: bold;
            color: #34495e;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
        }}
        
        h4 {{
            font-size: 13pt;
            font-weight: bold;
            color: #34495e;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }}
        
        p {{
            margin-bottom: 0.8em;
            text-indent: 0.5cm;
        }}
        
        ul, ol {{
            margin-bottom: 1em;
            padding-left: 1.5cm;
        }}
        
        li {{
            margin-bottom: 0.3em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            font-size: 11pt;
        }}
        
        th, td {{
            border: 1px solid #333;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            text-align: center;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        code {{
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        pre {{
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            background-color: #f4f4f4;
            padding: 1em;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin: 1em 0;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 1em;
            margin: 1em 0;
            font-style: italic;
            color: #555;
        }}
        
        .figure-caption {{
            text-align: center;
            font-style: italic;
            margin: 0.5em 0;
            font-size: 11pt;
        }}
        
        .table-caption {{
            text-align: center;
            font-weight: bold;
            margin-bottom: 0.5em;
            font-size: 12pt;
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        .no-break {{
            page-break-inside: avoid;
        }}
        
        strong {{
            font-weight: bold;
        }}
        
        em {{
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="title-page">
        <h1>METODOLOGI PENELITIAN CORTEXFLOW</h1>
        <h2>Kerangka Kerja Dekoding Neural dengan<br>Validasi Silang yang Ditingkatkan</h2>
        <div class="date">
            <strong>Tanggal: {datetime.now().strftime("%d %B %Y")}</strong>
        </div>
    </div>
    
    <div class="content">
        {html_body}
    </div>
</body>
</html>
"""
        
        print("✅ HTML document created with styling")
        
        # Convert HTML to PDF
        print("🔄 Converting HTML to PDF...")
        
        # Create HTML object
        html_obj = weasyprint.HTML(string=html_document, base_url='.')
        
        # Generate PDF
        pdf_file = 'METODOLOGI_CortexFlow.pdf'
        html_obj.write_pdf(pdf_file)
        
        print(f"✅ PDF successfully created: {pdf_file}")
        return True
        
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def main():
    """Main function"""
    
    print("📄 METODOLOGI PDF CONVERTER")
    print("=" * 40)
    
    # Check if source file exists
    import os
    if not os.path.exists('METODOLOGI.md'):
        print("❌ METODOLOGI.md not found!")
        return
    
    # Convert to PDF
    success = convert_to_pdf()
    
    if success:
        print("\n🎉 CONVERSION COMPLETED!")
        print("📄 Output: METODOLOGI_CortexFlow.pdf")
        
        # Check file size
        import os
        file_size = os.path.getsize('METODOLOGI_CortexFlow.pdf') / (1024 * 1024)
        print(f"📊 File size: {file_size:.2f} MB")
    else:
        print("\n❌ CONVERSION FAILED!")

if __name__ == "__main__":
    main()
