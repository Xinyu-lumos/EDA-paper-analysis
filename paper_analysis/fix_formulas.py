#!/usr/bin/env python3
"""Fix LaTeX formula delimiters and regenerate HTML for all day notes."""
import os, glob, re

base = os.path.dirname(os.path.abspath(__file__))

def fix_formulas(content):
    """Replace \( \) with $ $ and \[ \] with $$ $$"""
    # Inline math: \( ... \) -> $ ... $
    content = content.replace('\\( ', '$')
    content = content.replace(' \\)', '$')
    # Display math: \[ on own line -> $$
    content = re.sub(r'^\\\[$', '$$', content, flags=re.MULTILINE)
    # Display math: \] on own line -> $$
    content = re.sub(r'^\\\]$', '$$', content, flags=re.MULTILINE)
    return content

# Fix all markdown files
md_files = sorted(glob.glob(os.path.join(base, 'day*', '*.md')))
for md_path in md_files:
    with open(md_path, 'r', encoding='utf-8') as f:
        original = f.read()
    fixed = fix_formulas(original)
    if fixed != original:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        inline_count = original.count('\\( ')
        display_count = len(re.findall(r'^\\\[$', original, re.MULTILINE))
        print(f'Fixed {os.path.basename(md_path)}: {inline_count} inline, {display_count} display')
    else:
        print(f'OK {os.path.basename(md_path)}')

# Verify no remaining problematic patterns
print('\n--- Verification ---')
for md_path in md_files:
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    remaining_inline = content.count('\\( ')
    remaining_display = len(re.findall(r'^\\\[$', content, re.MULTILINE))
    status = 'PASS' if remaining_inline == 0 and remaining_display == 0 else 'FAIL'
    print(f'  {os.path.basename(md_path)}: inline={remaining_inline}, display={remaining_display} [{status}]')
