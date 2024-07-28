import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

nltk.download('maxent_ne_chunker')
nltk.download('words')
# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def extract_entities_spacy(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_entities_nltk(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity = " ".join([token for token, pos in subtree.leaves()])
            entities.append((entity, subtree.label()))
    return entities

if __name__ == "__main__":
    text = "Barack Obama was born on August 4, 1961, in Honolulu, Hawaii."

    print("Entities extracted using SpaCy:")
    print(extract_entities_spacy(text))

    print("Entities extracted using NLTK:")
    print(extract_entities_nltk(text))
