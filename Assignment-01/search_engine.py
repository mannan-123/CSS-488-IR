import os
import re
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class NounExtractor:
    def __init__(self):
        # Extended sets for specific noun types
        self.collective_nouns = {
            "team", "family", "flock", "jury", "committee", "army", "crowd", "group",
            "class", "band", "company", "panel", "crew", "squad", "community", "faculty"
        }
        self.material_nouns = {
            "gold", "silver", "wood", "iron", "plastic", "steel", "water", "air",
            "glass", "oil", "sand", "grain", "copper", "aluminum", "cotton", "wool"
        }
        self.abstract_nouns = {
            "freedom", "love", "beauty", "intelligence", "honesty", "wisdom", "courage",
            "justice", "happiness", "truth", "faith", "strength", "loyalty", "peace",
            "compassion", "patience", "humility", "dignity", "friendship", "hope",
            "creativity", "charity", "grace", "empathy", "kindness", "trust",
            "ambition", "confidence", "passion", "resilience", "integrity", "curiosity",
            "gratitude", "discipline", "honor", "imagination", "respect", "generosity",
            "perseverance", "sincerity", "thoughtfulness", "unity", "wisdom"
        }

        self.possessive_suffix = "'s"
        self.determiners = {"a", "an", "the"}

    def extract_nouns(self, words):
        # Initialize a set to store unique nouns
        nouns = set()

        # Simple POS tagging simulation
        for i, word in enumerate(words):
            word = word.strip(",.!?")

            # Proper nouns: Start with a capital letter and are not the first word after a determiner
            if word[0].isupper() and (i == 0 or words[i-1].lower() not in self.determiners):
                nouns.add(word)

            # Collective nouns
            elif word.lower() in self.collective_nouns:
                nouns.add(word)

            # Material nouns
            elif word.lower() in self.material_nouns:
                nouns.add(word)

            # Possessive nouns
            elif word.endswith(self.possessive_suffix):
                nouns.add(word)

            # Compound nouns (simple heuristic: checks if the word has a hyphen)
            elif "-" in word:
                nouns.add(word)

            # Abstract nouns (hard-coded example; extendable for specific concepts)
            elif word.lower() in self.abstract_nouns:
                nouns.add(word)

        return nouns


class HashTable:
    def __init__(self, initial_size=16):
        """Initialize the hash table with a specific initial size."""
        self.table = [None] * initial_size  # Create a list of empty buckets
        self.size = initial_size
        self.count = 0

    def _hash(self, key):
        """Generate a hash code for the given key."""
        hash_value = 0
        for char in key:
            hash_value = (hash_value * 31 + ord(char)) % self.size
        return hash_value

    def _resize(self):
        """Resize the hash table when it's too full."""
        self.size *= 2
        new_table = [None] * self.size
        for bucket in self.table:
            if bucket:
                for key, value in bucket:
                    new_index = self._hash(key)
                    if new_table[new_index] is None:
                        new_table[new_index] = [(key, value)]
                    else:
                        new_table[new_index].append((key, value))
        self.table = new_table

    def insert(self, key, value):
        """Insert a key-value pair into the hash table."""
        index = self._hash(key)
        if self.table[index] is None:
            self.table[index] = [(key, value)]
        else:
            # Handle collision using chaining (linked list)
            for i, (existing_key, existing_value) in enumerate(self.table[index]):
                if existing_key == key:
                    # If the word already exists, update the word count for the doc_id
                    found = False
                    for doc in existing_value:
                        if doc['doc_id'] == value[0]['doc_id']:
                            # Increment word count for this doc
                            doc['count'] += 1
                            found = True
                            break
                    if not found:
                        # Append new doc_id if not found
                        self.table[index][i][1].append(value[0])
                    return
            self.table[index].append((key, value))  # Insert new key-value pair

        self.count += 1
        if self.count > self.size * 0.7:  # Load factor threshold for resizing
            self._resize()

    def lookup(self, key):
        """Lookup the value associated with the given key."""
        index = self._hash(key)
        if self.table[index]:
            for stored_key, stored_value in self.table[index]:
                if stored_key == key:
                    return stored_value  # Return all associated document details
        return None  # Return None if the key is not found

    def remove(self, key):
        """Remove a key-value pair from the hash table."""
        index = self._hash(key)
        if self.table[index]:
            for i, (stored_key, stored_value) in enumerate(self.table[index]):
                if stored_key == key:
                    del self.table[index][i]
                    self.count -= 1
                    return True
        return False  # Return False if the key was not found

    def __str__(self):
        """String representation for debugging."""
        return str(self.table)


class Indexer:
    def __init__(self):
        self.index = HashTable()

    def add_document(self, doc_id, words):
        """Add a document to the index."""
        for word in words:
            self.index.insert(
                word.lower(), [{'doc_id': doc_id, 'count': 1}])

    def search(self, query_words):
        """Search for documents by keyword."""
        results = []
        for word in query_words:
            search_result = self.index.lookup(word.lower())
            if search_result:
                # Flatten the list of results and get only the doc_id's
                for item in search_result:
                    # Each item is now a dict with doc_id and count
                    results.append(
                        f"Query Word: {word}, Doc ID: {item['doc_id']}, (Count: {item['count']})")
        return results

    def lookup(self, word):
        """Lookup a word in the index."""
        return self.index.lookup(word)


class SearchEngine:
    def __init__(self):
        self.indexer_title = Indexer()
        self.indexer_content = Indexer()
        self.doc_lengths = {}
        self.documents = {}
        self.no_of_docs = 1

    def add_document(self, title, content):

        preprocessed_title = self.preprocessing(title, is_query=False)
        preprocessed_content = self.preprocessing(content, is_query=False)

        self.indexer_title.add_document(self.no_of_docs, preprocessed_title)
        self.indexer_content.add_document(
            self.no_of_docs, preprocessed_content)

        self.documents[self.no_of_docs] = {'title': title, 'content': content}
        self.doc_lengths[self.no_of_docs] = len(content.split())

        self.no_of_docs += 1

    def preprocessing(self, text, is_query=False):
        """Process text: remove punctuation, convert to lowercase, and lemmatize words."""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)

        if not is_query:
            extractor = NounExtractor()
            word_tokens = extractor.extract_nouns(word_tokens)

        return [word for word in word_tokens if word.lower() not in stop_words]

    def search_by_title(self, query):
        """Search by title using index."""
        words = self.preprocessing(query, is_query=True)
        results = self.indexer_title.search(words)

        format_string = ""
        for item in results:

            item = item.split(',')
            doc_id = int(item[1].split(':')[1].strip())
            count = item[2].split(':')[1].strip()
            count = int(count[:-1])
            query_word = item[0].split(':')[1].strip()
            title = self.documents[doc_id].get('title')
            content = self.documents[doc_id].get('content')

            format_string += f"Document ID: {doc_id}, Title: {title}\n{content}\n\n"
        return format_string

    def search_by_content(self, query):
        """Search by title using index."""
        words = self.preprocessing(query, is_query=True)
        results = self.indexer_content.search(words)

        format_string = ""
        for item in results:

            item = item.split(',')
            doc_id = int(item[1].split(':')[1].strip())
            count = item[2].split(':')[1].strip()
            count = int(count[:-1])
            query_word = item[0].split(':')[1].strip()
            title = self.documents[doc_id].get('title')
            content = self.documents[doc_id].get('content')

            format_string += f"Document ID: {doc_id}, Title: {title}\n{content}\n\n"
        return format_string

    def search_by_tf_idf(self, query):
        """Ranked search based on TF-IDF scores."""
        words = self.preprocessing(query, is_query=True)

        # Initialize a dictionary to store the TF-IDF scores for each document
        scores = {doc_id: 0 for doc_id in self.documents}

        for word in words:
            # Get the document list for the word from the content index
            word_docs = self.indexer_content.lookup(word.lower())

            if not word_docs:
                continue

            # Calculate IDF for the word
            # IDF = log(total_docs / (docs_containing_word))
            docs_containing_word = len(word_docs)
            # Add 1 to avoid division by zero
            idf = math.log(self.no_of_docs / (docs_containing_word + 1))

            for item in word_docs:
                doc_id = item['doc_id']
                count = item['count']

                # Calculate TF for the word in the document
                doc_length = self.doc_lengths[doc_id]
                tf = count / doc_length

                # Update the TF-IDF score for the document
                scores[doc_id] += tf * idf

        # Rank the documents by their TF-IDF scores (highest to lowest)
        ranked_results = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)

        format_string = self.get_ranked_results_format_string(ranked_results)
        return format_string

    def get_ranked_results_format_string(self, ranked_results):

        format_string = ""
        """Display ranked search results with TF-IDF scores."""
        for doc_id, score in ranked_results:
            if score == 0:
                continue
            doc = self.documents[doc_id]

            format_string += f"\nDocument ID {doc_id} (Score: {score:.4f}): {doc['title']}\n{doc['content']}\n\n"
        return format_string


def load_documents(folder_path):
    """Load text documents from a folder into the search engine."""
    search_engine = SearchEngine()
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                title = filename.replace('.txt', '')
                content = file.read()
                search_engine.add_document(title, content)
    return search_engine


def main():
    folder_path = './documents'
    search_engine = load_documents(folder_path)

    while True:
        os.system('cls')
        print("\n--- Document Search Engine ---")
        print("1. Search by Title")
        print("2. Search by Content")
        print("3. Ranked Search (TF-IDF)")
        print("4. Exit")

        choice = input("Choose an option: ")

        if choice == '1':
            title_query = input("Enter the title to search for: ")
            results = search_engine.search_by_title(title_query)
            if results:
                print(results)
            else:
                print("No documents found with the given title.")
        elif choice == '2':
            content_query = input("Enter the content to search for: ")
            results = search_engine.search_by_content(content_query)
            if results:
                print(results)
            else:
                print("No documents found with the given content.")
        elif choice == '3':
            ranked_query = input("Enter the ranked query: ")
            ranked_results = search_engine.search_by_tf_idf(
                ranked_query)
            if ranked_results:
                print(ranked_results)
            else:
                print("No documents found for the ranked query.")
        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
