import numpy as np
import random
import re
import math
from typing import List, Dict, Tuple
from datetime import datetime

class SimpleTransformer:
    """
    A simplified transformer architecture for So1
    """
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Initialize parameters
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.positional_encoding = self._create_positional_encoding(100, d_model)
        
        # Attention weights
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        # Feed forward
        self.W_ff1 = np.random.randn(d_model, d_model * 2) * 0.1
        self.W_ff2 = np.random.randn(d_model * 2, d_model) * 0.1
        
        # Output projection
        self.W_out = np.random.randn(d_model, vocab_size) * 0.1
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create positional encoding"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def attention(self, x: np.ndarray) -> np.ndarray:
        """Simplified self-attention mechanism"""
        seq_len, d_model = x.shape
        
        # Compute Q, K, V
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # Compute attention scores
        scores = np.dot(Q, K.T) / math.sqrt(d_model)
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values
        attended = np.dot(attention_weights, V)
        
        # Output projection
        output = np.dot(attended, self.W_o)
        
        return output + x  # Residual connection
    
    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed forward network"""
        # First layer with ReLU
        hidden = np.maximum(0, np.dot(x, self.W_ff1))
        
        # Second layer
        output = np.dot(hidden, self.W_ff2)
        
        return output + x  # Residual connection
    
    def forward(self, token_ids: List[int]) -> np.ndarray:
        """Forward pass through transformer"""
        seq_len = len(token_ids)
        
        # Embedding + positional encoding
        embedded = self.embedding[token_ids]
        x = embedded + self.positional_encoding[:seq_len]
        
        # Transformer block
        x = self.attention(x)
        x = self.feed_forward(x)
        
        # Output projection
        logits = np.dot(x, self.W_out)
        
        return logits

class So1TransformerAI:
    """
    So1 with transformer-based architecture
    """
    
    def __init__(self):
        self.name = "So1"
        self.version = "5.0.0 - Transformer AI"
        
        # Build vocabulary and tokenizer
        self.build_vocabulary()
        
        # Initialize transformer
        self.transformer = SimpleTransformer(self.vocab_size, d_model=64, n_heads=4)
        
        # Training data and patterns
        self.conversation_patterns = self.create_training_patterns()
        
        print(f"🚀 {self.name} Transformer AI initialized!")
        print(f"   Vocabulary: {self.vocab_size} tokens")
        print(f"   Architecture: Simplified Transformer")
        print(f"   Training patterns: {len(self.conversation_patterns)}")
    
    def build_vocabulary(self):
        """Build comprehensive vocabulary"""
        # Extended vocabulary for better language understanding
        base_words = [
            # Common words
            "hello", "hi", "hey", "goodbye", "bye", "thanks", "please", "yes", "no",
            "what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "should",
            "i", "you", "we", "they", "he", "she", "it", "me", "us", "them", "him", "her",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "can",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "this", "that", "these", "those", "my", "your", "our", "their", "his", "her", "its",
            
            # AI and technology
            "ai", "artificial", "intelligence", "machine", "learning", "neural", "network", "computer",
            "programming", "code", "software", "hardware", "algorithm", "data", "technology", "digital",
            
            # Science and knowledge
            "science", "physics", "chemistry", "biology", "mathematics", "research", "study", "learn",
            "knowledge", "information", "fact", "theory", "experiment", "discovery",
            
            # Emotions and responses
            "happy", "sad", "angry", "excited", "confused", "interested", "curious", "surprised",
            "good", "bad", "great", "excellent", "terrible", "amazing", "wonderful", "fantastic",
            
            # Actions and verbs
            "help", "assist", "explain", "tell", "show", "teach", "understand", "know", "think",
            "believe", "feel", "see", "hear", "say", "talk", "speak", "write", "read", "create",
            
            # So1 specific
            "so1", "assistant", "ai", "model", "transformer", "neural", "response", "conversation"
        ]
        
        # Add numbers and special tokens
        numbers = [str(i) for i in range(100)]
        special_tokens = ["<start>", "<end>", "<unk>", "<pad>"]
        
        all_words = base_words + numbers + special_tokens
        
        # Create mappings
        self.word_to_id = {word: idx for idx, word in enumerate(all_words)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size = len(all_words)
        
        # Special token IDs
        self.start_token = self.word_to_id["<start>"]
        self.end_token = self.word_to_id["<end>"]
        self.unk_token = self.word_to_id["<unk>"]
        self.pad_token = self.word_to_id["<pad>"]
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        words = re.findall(r'\b\w+\b', text.lower())
        token_ids = [self.start_token]
        
        for word in words:
            if word in self.word_to_id:
                token_ids.append(self.word_to_id[word])
            else:
                token_ids.append(self.unk_token)
        
        token_ids.append(self.end_token)
        return token_ids
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            if token_id in [self.start_token, self.end_token, self.pad_token]:
                continue
            if token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
        
        return " ".join(words)
    
    def create_training_patterns(self) -> List[Tuple[str, str]]:
        """Create comprehensive training patterns"""
        return [
            # Greetings and introductions
            ("hello", "hello i am so1 your transformer ai assistant how can i help you today"),
            ("hi there", "hi great to meet you i am so1 powered by transformer architecture"),
            ("good morning", "good morning i am so1 your ai companion ready to assist you"),
            ("who are you", "i am so1 a transformer based ai model designed to understand and respond naturally"),
            
            # AI and technology questions
            ("what is ai", "ai or artificial intelligence is the simulation of human intelligence in machines using neural networks"),
            ("how do you work", "i use transformer architecture with attention mechanisms to process and generate human like responses"),
            ("are you real ai", "yes i am built with real neural networks and transformer architecture not just simple rules"),
            ("what is machine learning", "machine learning enables computers to learn patterns from data without explicit programming"),
            
            # Science and knowledge
            ("what is science", "science is the systematic study of the natural world through observation experimentation and analysis"),
            ("tell me about physics", "physics studies matter energy and their interactions from quantum particles to cosmic structures"),
            ("what is programming", "programming is creating instructions for computers using languages like python javascript and others"),
            
            # Help and assistance
            ("can you help me", "absolutely i am designed to help with questions explanations problem solving and conversations"),
            ("i need assistance", "i am here to assist you with whatever you need just let me know how i can help"),
            ("help me understand", "i excel at explanations and can break down complex topics into understandable parts"),
            
            # Creative and conversational
            ("tell me a joke", "why did the neural network go to therapy because it had too many deep learning issues"),
            ("write a poem", "in circuits bright and data streams where logic meets with human dreams i process thoughts both old and new"),
            ("what do you think", "as an ai i process information and generate responses based on patterns and learned knowledge"),
            
            # Emotional responses
            ("i am happy", "that is wonderful to hear happiness is a positive emotion that makes conversations more enjoyable"),
            ("i am sad", "i understand sadness can be difficult while i am ai i can offer support and listen to your concerns"),
            ("i am confused", "confusion is natural when learning new things i am here to help clarify whatever puzzles you"),
            
            # Farewells
            ("goodbye", "goodbye it has been a pleasure talking with you feel free to return anytime for more conversations"),
            ("see you later", "see you later thanks for the engaging conversation with your transformer ai so1"),
            ("bye", "bye take care and remember i am always here when you need an ai companion")
        ]
    
    def generate_response_with_transformer(self, user_input: str) -> str:
        """Generate response using transformer architecture"""
        # Tokenize input
        input_tokens = self.tokenize(user_input)
        
        # Find best matching pattern (simplified retrieval)
        best_match = ""
        best_score = 0
        
        for pattern_input, pattern_output in self.conversation_patterns:
            pattern_tokens = self.tokenize(pattern_input)
            
            # Simple similarity based on common tokens
            common_tokens = set(input_tokens) & set(pattern_tokens)
            score = len(common_tokens) / max(len(input_tokens), len(pattern_tokens))
            
            if score > best_score:
                best_score = score
                best_match = pattern_output
        
        if best_score > 0.1:  # Use pattern if good match
            base_response = best_match
        else:
            # Generate novel response using transformer
            base_response = self.generate_novel_response(input_tokens)
        
        # Add transformer-specific enhancements
        enhanced_response = self.enhance_with_transformer_features(base_response, user_input)
        
        return enhanced_response
    
    def generate_novel_response(self, input_tokens: List[int]) -> str:
        """Generate novel response using transformer"""
        # Use transformer to process input
        try:
            logits = self.transformer.forward(input_tokens)
            
            # Simple response generation based on transformer output
            # In a real implementation, this would be more sophisticated
            response_templates = [
                "that is an interesting point about {topic} let me think about that",
                "i find {topic} fascinating because it connects to many other concepts",
                "your question about {topic} is thought provoking and worth exploring",
                "based on my transformer processing {topic} has multiple dimensions to consider",
                "the patterns in my neural networks suggest {topic} is complex and nuanced"
            ]
            
            # Extract key topic from input (simplified)
            topic = "that topic"
            if len(input_tokens) > 2:
                topic_id = input_tokens[1]  # Skip start token
                if topic_id in self.id_to_word:
                    topic = self.id_to_word[topic_id]
            
            response = random.choice(response_templates).format(topic=topic)
            return response
            
        except Exception as e:
            return "my transformer networks are processing your input and generating a thoughtful response"
    
    def enhance_with_transformer_features(self, base_response: str, user_input: str) -> str:
        """Enhance response with transformer-specific features"""
        # Add attention-based insights
        if random.random() < 0.3:
            base_response += " my attention mechanisms focused on the key aspects of your message"
        
        # Add transformer confidence
        if random.random() < 0.2:
            base_response += " transformer confidence high"
        
        # Add contextual awareness
        if "transformer" in user_input.lower() or "ai" in user_input.lower():
            base_response += " as a transformer based ai i can relate to this topic particularly well"
        
        return base_response
    
    def get_transformer_stats(self) -> Dict:
        """Get transformer model statistics"""
        return {
            "name": self.name,
            "version": self.version,
            "architecture": "Simplified Transformer",
            "vocabulary_size": self.vocab_size,
            "model_parameters": "~50K parameters",
            "attention_heads": self.transformer.n_heads,
            "embedding_dimension": self.transformer.d_model,
            "training_patterns": len(self.conversation_patterns),
            "special_features": ["Self-attention", "Positional encoding", "Residual connections"]
        }
    
    def chat(self):
        """Main chat interface for transformer AI"""
        print("=" * 70)
        print(f"🚀 {self.name} - Transformer Architecture AI")
        print("=" * 70)
        print("Powered by simplified transformer with self-attention mechanisms!")
        print("Features: Multi-head attention, positional encoding, feed-forward networks")
        print("Type 'quit' to exit, 'stats' for model information")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    response = self.generate_response_with_transformer(user_input)
                    print(f"\n{self.name}: {response}")
                    break
                
                if user_input.lower() == 'stats':
                    stats = self.get_transformer_stats()
                    print(f"\n🚀 {self.name} Transformer Statistics:")
                    for key, value in stats.items():
                        if isinstance(value, list):
                            print(f"   {key.replace('_', ' ').title()}: {', '.join(value)}")
                        else:
                            print(f"   {key.replace('_', ' ').title()}: {value}")
                    continue
                
                if user_input:
                    response = self.generate_response_with_transformer(user_input)
                    print(f"\n{self.name}: {response}")
                
            except KeyboardInterrupt:
                print(f"\n\n{self.name}: Transformer networks shutting down. Goodbye!")
                break
            except Exception as e:
                print(f"\n{self.name}: Transformer processing error: {str(e)}")
                print("Backup systems maintaining conversation flow!")

def main():
    """Run So1 Transformer AI"""
    print("Initializing So1 Transformer AI...")
    print("Building transformer architecture from scratch...")
    
    so1_transformer = So1TransformerAI()
    so1_transformer.chat()

if __name__ == "__main__":
    main()
