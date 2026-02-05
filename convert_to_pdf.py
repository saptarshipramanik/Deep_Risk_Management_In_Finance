"""
Convert Markdown technical report to PDF.

This script converts the technical report from Markdown to PDF format.
"""

import markdown
from pathlib import Path

def md_to_html(md_file, output_file):
    """Convert Markdown to HTML with styling."""
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Add CSS styling
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Deep Hedging - Technical Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
            margin-top: 25px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        @media print {{
            body {{
                max-width: 100%;
                margin: 0;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Write HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"HTML created: {output_file}")
    return output_file


if __name__ == "__main__":
    # Paths
    md_file = Path("docs/technical_report.md")
    html_file = Path("docs/technical_report.html")
    
    # Convert
    print("Converting Markdown to HTML...")
    html_path = md_to_html(md_file, html_file)
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"\nHTML file: {html_file}")
    print("\nTo convert to PDF:")
    print("1. Open the HTML file in your browser")
    print("2. Press Ctrl+P (Print)")
    print("3. Select 'Save as PDF'")
    print("4. Save as 'technical_report.pdf'")
    print("\nAlternatively, use a tool like wkhtmltopdf or pandoc.")

