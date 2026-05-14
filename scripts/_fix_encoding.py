content = open('scripts/generate_data_pdf.py', encoding='utf-8').read()
replacements = {
    '—': ' - ', '–': ' - ', '─': '-', '│': '|',
    '’': "'", '‘': "'", '“': '"', '”': '"',
    '→': '->', '←': '<-', '−': '-', '…': '...',
}
for src, dst in replacements.items():
    content = content.replace(src, dst)
result = ''.join(c if ord(c) < 256 else '?' for c in content)
open('scripts/generate_data_pdf.py', 'w', encoding='utf-8').write(result)
print('Done, non-latin1 chars replaced')
