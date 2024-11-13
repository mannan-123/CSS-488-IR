import os
import re
import math
from nltk.stem import WordNetLemmatizer

# Stopwords to exclude from indexing
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of',
    'to', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does',
    'did', 'but', 'not', 'that', 'this', 'it', 'its', 'they', 'them', 'he', 'she', 'his', 'her',
    'you', 'we', 'i', 'me', 'my', 'mine', 'your', 'yours', 'their', 'our', 'us', 'can', 'will',
    'just', 'if', 'so', 'no', 'yes', 'all', 'any', 'some', 'which', 'who', 'whom', 'how', 'why',
    'then', 'than', 'now', 'up', 'down', 'out', 'when', 'where'
}

lemmatizer = WordNetLemmatizer()


class SearchEngine:
    def __init__(self):
        self.index = {}
        self.doc_lengths = {}
        self.documents = {}

    def lemmatize(self, word):
        """Lemmatize a word using WordNetLemmatizer."""
        original_word = word

        word = lemmatizer.lemmatize(word)
        print(f"Original Word: {original_word} | Lemmatized Word: {word}")
        return word

    def add_to_index(self, word, doc_id, is_title=False):
        """Add a word to the index under title or content."""
        if word not in self.index:
            self.index[word] = {
                'title': set(), 'content': set(), 'doc_freq': 0}
        if is_title:
            self.index[word]['title'].add(doc_id)
        else:
            self.index[word]['content'].add(doc_id)
        self.index[word]['doc_freq'] += 1

    def add_document(self, doc_id, title, content):
        """Add document's title and content to the index."""
        self.documents[doc_id] = {'title': title, 'content': content}
        self.doc_lengths[doc_id] = len(content.split())
        # Index title and content words
        for word in self.tokenize(title):
            self.add_to_index(word, doc_id, is_title=True)
        for word in self.tokenize(content):
            self.add_to_index(word, doc_id, is_title=False)

    def tokenize(self, text):
        """Process text: remove punctuation, convert to lowercase, and lemmatize words."""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        return [self.lemmatize(word) for word in text.split() if word not in STOPWORDS]

    def search_by_title(self, query):
        """Search by title using index."""
        words = self.tokenize(query)
        results = set(self.index.get(words[0], {}).get('title', []))
        for word in words[1:]:
            results |= self.index.get(word, {}).get('title', set())
        return results

    def search_by_content(self, query):
        """Search by content using index."""
        words = self.tokenize(query)
        results = set(self.index.get(words[0], {}).get('content', []))
        for word in words[1:]:
            results |= self.index.get(word, {}).get('content', set())
        return results

    def tf_idf_search(self, query):
        """Ranked search based on TF-IDF scores."""
        words = self.tokenize(query)
        scores = {}
        num_docs = len(self.documents)
        for word in words:
            if word in self.index:
                for doc_id in self.index[word]['content']:
                    tf = self.documents[doc_id]['content'].lower().count(
                        word) / self.doc_lengths[doc_id]
                    idf = math.log(
                        num_docs / (1 + self.index[word]['doc_freq']))
                    scores[doc_id] = scores.get(doc_id, 0) + (tf * idf)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def display_results(self, doc_ids):
        """Display search results with document title and content with line numbers."""
        for doc_id in doc_ids:
            doc = self.documents[doc_id]
            print(f"\nDocument ID {doc_id}: {doc['title']}")
            for i, line in enumerate(doc['content'].splitlines(), 1):
                print(f"Line {i}: {line.strip()}")

    def display_ranked_results(self, ranked_results):
        """Display ranked search results with TF-IDF scores."""
        for doc_id, score in ranked_results:
            doc = self.documents[doc_id]
            print(
                f"\nDocument ID {doc_id} (Score: {score:.4f}): {doc['title']}")
            for i, line in enumerate(doc['content'].splitlines(), 1):
                print(f"Line {i}: {line.strip()}")


def load_documents(folder_path):
    """Load text documents from a folder into the search engine."""
    search_engine = SearchEngine()
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                title = filename.replace('.txt', '')
                content = file.read()
                doc_id = len(search_engine.documents) + 1
                search_engine.add_document(doc_id, title, content)
    return search_engine


def main():
    folder_path = './documents'
    search_engine = load_documents(folder_path)

    while True:
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
                search_engine.display_results(results)
            else:
                print("No documents found with the given title.")
        elif choice == '2':
            content_query = input("Enter the content to search for: ")
            results = search_engine.search_by_content(content_query)
            if results:
                search_engine.display_results(results)
            else:
                print("No documents found with the given content.")

        elif choice == '3':
            ranked_query = input("Enter the ranked query: ")
            ranked_results = search_engine.tf_idf_search(ranked_query)
            if ranked_results:
                search_engine.display_ranked_results(ranked_results)
            else:
                print("No documents found for the ranked query.")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
