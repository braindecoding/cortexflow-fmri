#!/usr/bin/env python3
"""
Clean Emoji from SOTA.md
========================

Remove all emoji/emoticons from SOTA.md to make it proper academic format.
"""

import re

def clean_emoji_from_file(file_path):
    """Remove emoji and emoticons from file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Common emoji patterns to remove
    emoji_patterns = [
        r'🎯', r'🚀', r'✅', r'❌', r'🏆', r'📊', r'🔄', r'💡',
        r'🎉', r'📈', r'🔍', r'⚠️', r'📁', r'🥇', r'🥈', r'🥉',
        r'🔬', r'📄', r'✨', r'🎊', r'📚', r'🌟', r'💯', r'🔥',
        r'👍', r'👎', r'💪', r'🧠', r'🎨', r'📝', r'🔧', r'⭐',
        r'🚨', r'⚡', r'🎪', r'🎭', r'🎬', r'🎮', r'🎲', r'🎯'
    ]
    
    # Remove emoji patterns
    for pattern in emoji_patterns:
        content = re.sub(pattern, '', content)
    
    # Remove any remaining emoji using Unicode ranges
    # Remove most common emoji ranges
    emoji_ranges = [
        r'[\U0001F600-\U0001F64F]',  # emoticons
        r'[\U0001F300-\U0001F5FF]',  # symbols & pictographs
        r'[\U0001F680-\U0001F6FF]',  # transport & map symbols
        r'[\U0001F1E0-\U0001F1FF]',  # flags (iOS)
        r'[\U00002702-\U000027B0]',  # dingbats
        r'[\U000024C2-\U0001F251]'   # enclosed characters
    ]
    
    for pattern in emoji_ranges:
        content = re.sub(pattern, '', content)
    
    # Clean up extra spaces
    content = re.sub(r' +', ' ', content)  # Multiple spaces to single
    content = re.sub(r'\n +', '\n', content)  # Spaces at line start
    content = re.sub(r' +\n', '\n', content)  # Spaces at line end
    
    # Write cleaned content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Cleaned emoji from {file_path}")

def main():
    """Main execution"""
    
    print("Cleaning emoji from SOTA.md...")
    clean_emoji_from_file('SOTA.md')
    print("Done!")

if __name__ == "__main__":
    main()
