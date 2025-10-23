import re
import string

class TextNormalizer:
    def __init__(self, prohibited_words_path):
        """Initialize with prohibited words list"""
        self.load_prohibited_words(prohibited_words_path)
        self.morph_patterns = self._create_morph_patterns()
    
    def load_prohibited_words(self, path):
        """Load prohibited words from file"""
        with open(path, 'r') as f:
            self.prohibited_words = [line.strip().lower() for line in f]
    
    def _create_morph_patterns(self):
        """Create regex patterns for common obfuscation tactics"""
        patterns = {}
        for word in self.prohibited_words:
            # Pattern 1: Letter substitution (e.g., "cur3" for "cure")
            pattern1 = word.replace('e', '[e3]').replace('a', '[a4@]').replace('o', '[o0]')
            
            # Pattern 2: Spacing (e.g., "c u r e")
            pattern2 = ' '.join(list(word))
            
            # Pattern 3: Special chars (e.g., "c*u*r*e")
            pattern3 = '[*_-]'.join(list(word))
            
            patterns[word] = [pattern1, pattern2, pattern3]
        
        return patterns
    
    def normalize(self, text):
        """Detect and normalize morphed language"""
        normalized = text.lower()
        
        # Remove excessive punctuation
        normalized = re.sub(r'[!]{2,}', '!', normalized)
        
        # Detect and replace common obfuscations
        for original, patterns in self.morph_patterns.items():
            for pattern in patterns:
                normalized = re.sub(pattern, original, normalized, flags=re.IGNORECASE)
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def detect_morphs(self, text):
        """Return detected morphed words"""
        detected = []
        text_lower = text.lower()
        
        for word in self.prohibited_words:
            # Check for various obfuscation patterns
            if any(re.search(pattern, text_lower) for pattern in self.morph_patterns[word]):
                detected.append(word)
        
        return detected

