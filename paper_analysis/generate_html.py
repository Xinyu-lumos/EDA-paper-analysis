#!/usr/bin/env python3
"""Convert all day N markdown files to HTML with MathJax and Mermaid support."""
import os
import glob
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.toc import TocExtension

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:'default'}});</script>
<style>
body {{
  max-width: 900px;
  margin: 0 auto;
  padding: 20px 40px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.7;
  color: #333;
  background: #fafafa;
}}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 8px; }}
h2 {{ color: #16213e; border-bottom: 1px solid #e0e0e0; padding-bottom: 5px; margin-top: 2em; }}
h3 {{ color: #0f3460; }}
blockquote {{
  border-left: 4px solid #e94560;
  margin: 1em 0;
  padding: 0.5em 1em;
  background: #fff;
  border-radius: 0 4px 4px 0;
}}
code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
pre {{ background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 8px; overflow-x: auto; }}
pre code {{ background: none; padding: 0; color: inherit; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #16213e; color: white; }}
tr:nth-child(even) {{ background: #f5f5f5; }}
a {{ color: #e94560; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
img {{ max-width: 100%; }}
.mermaid {{ background: white; padding: 10px; border-radius: 8px; margin: 1em 0; }}
hr {{ border: none; border-top: 1px solid #e0e0e0; margin: 2em 0; }}
</style>
</head>
<body>
{content}
</body>
</html>
"""

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Extract title from first heading
    title = "EDA Paper Analysis"
    for line in md_text.split('\n'):
        if line.startswith('# '):
            title = line[2:].strip()
            break

    # Convert mermaid blocks to divs for mermaid.js rendering
    md_text = md_text.replace('```mermaid', '<div class="mermaid">')
    # Close mermaid divs: find lines that are just ``` after mermaid content
    lines = md_text.split('\n')
    in_mermaid = False
    new_lines = []
    for line in lines:
        if line.strip() == '<div class="mermaid">':
            in_mermaid = True
            new_lines.append(line)
        elif in_mermaid and line.strip() == '```':
            new_lines.append('</div>')
            in_mermaid = False
        else:
            new_lines.append(line)
    md_text = '\n'.join(new_lines)

    # Convert markdown to HTML
    extensions = [
        'tables',
        'fenced_code',
        'toc',
        'nl2br',
    ]
    html_body = markdown.markdown(md_text, extensions=extensions)

    # Wrap in template
    html_full = HTML_TEMPLATE.format(title=title, content=html_body)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_full)

    print(f"  Generated: {html_path}")

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    md_files = sorted(glob.glob(os.path.join(base, 'day*/*.md')))

    print(f"Found {len(md_files)} markdown files")
    for md_path in md_files:
        html_path = md_path.replace('.md', '.html')
        convert_md_to_html(md_path, html_path)

    # Generate index page
    index_content = '<h1>EDA Placement Paper Analysis</h1>\n<p>每日论文分析系列 — VLSI 布局领域近10年顶级会议/期刊论文</p>\n<ul>\n'
    for md_path in md_files:
        rel = os.path.relpath(md_path, base)
        day = os.path.basename(os.path.dirname(md_path))
        html_rel = rel.replace('.md', '.html')
        with open(md_path, 'r', encoding='utf-8') as f:
            title = next((l[2:].strip() for l in f if l.startswith('# ')), day)
        index_content += f'<li><a href="{html_rel}">{title}</a></li>\n'
    index_content += '</ul>'

    index_html = HTML_TEMPLATE.format(title="EDA Placement Paper Analysis", content=index_content)
    index_path = os.path.join(base, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    print(f"  Generated: {index_path}")
    print("Done!")

if __name__ == '__main__':
    main()
