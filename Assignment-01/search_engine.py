import os
import re
import math
from nltk.stem import WordNetLemmatizer

# Define stopwords to be excluded from the index
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of',
    'to', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does',
    'did', 'but', 'not', 'that', 'this', 'it', 'its', 'they', 'them', 'he', 'she', 'his', 'her',
    'you', 'we', 'I', 'me', 'my', 'mine', 'your', 'yours', 'their', 'our', 'us', 'can', 'will',
    'just', 'if', 'so', 'no', 'yes', 'all', 'any', 'some', 'which', 'who', 'whom', 'how', 'why',
    'then', 'than', 'now', 'up', 'down', 'out', 'when', 'where'
}

# Initialize the NLTK WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


class SearchEngine:
    def __init__(self):
        # Dictionary to store the index of words and their document IDs
        self.index = {}
        # Store document lengths (for TF-IDF)
        self.doc_lengths = {}
        self.documents = {}  # Store document titles and content by ID

    def lemmatize(self, word):
        """Convert word to its base form using NLTK's WordNetLemmatizer."""
        return lemmatizer.lemmatize(word)

    def add_document(self, doc_id, title, content):
        """Add a document's title and content to the search engine."""
        self.documents[doc_id] = {'title': title, 'content': content}
        words = self.tokenize(content)
        self.doc_lengths[doc_id] = len(words)
        for word in words:
            if word not in STOPWORDS:
                word = self.lemmatize(word)  # Apply lemmatization
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(doc_id)

    def tokenize(self, text):
        """Convert text to lowercase, remove punctuation, split into words, and lemmatize."""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
        words = text.lower().split()
        return [self.lemmatize(word) for word in words if word not in STOPWORDS]

    def search_by_title(self, title_query):
        """Search for a document by title."""
        title_query = title_query.strip()  # Handle leading/trailing spaces
        print("\n--- Search by Title ---")
        results = []
        for doc_id, doc in self.documents.items():
            if title_query.lower() in doc['title'].lower():
                results.append(f"Document ID {doc_id}: {doc['title']}")

        if results:
            print("\n".join(results))
        else:
            print("No documents found with the given title.")

    def search_by_content(self, content_query):
        """Search for a document by content and show the line numbers where the query exists."""
        content_query = content_query.strip()  # Handle leading/trailing spaces
        print("\n--- Search by Content ---")
        results = []
        for doc_id, doc in self.documents.items():
            lines = doc['content'].split('\n')
            found_lines = [
                f"Line {i + 1}: {line.strip()}"
                for i, line in enumerate(lines)
                if content_query.lower() in line.strip().lower()  # Stripping lines
            ]
            if found_lines:
                results.append(
                    f"\nDocument ID {doc_id}: {self.documents[doc_id]['title']}\n" + "\n".join(found_lines))

        if results:
            print("\n".join(results))
        else:
            print("No documents found with the given content.")

    def boolean_search(self, query):
        """Process Boolean queries with AND, OR, and NOT."""
        query = query.strip()  # Handle leading/trailing spaces
        print("\n--- Boolean Search ---")
        operators = {'AND', 'OR', 'NOT'}
        query_terms = query.upper().split()
        terms = [self.lemmatize(word.lower())
                 for word in query_terms if word not in operators]

        # Process 'AND', 'OR', 'NOT' logic
        result_docs = set()
        current_operator = 'AND'
        for term in terms:
            if current_operator == 'AND':
                if term in self.index:
                    result_docs &= set(self.index[term])
            elif current_operator == 'OR':
                if term in self.index:
                    result_docs |= set(self.index[term])
            elif current_operator == 'NOT':
                if term in self.index:
                    result_docs -= set(self.index[term])
            current_operator = term  # Switch to the next operator

        if result_docs:
            print("\nDocuments found:")
            for doc_id in result_docs:
                print(
                    f"Document ID {doc_id}: {self.documents[doc_id]['title']}")
        else:
            print("No documents found with the Boolean query.")

    def ranking_search(self, query):
        """Rank documents by relevance to the query using TF-IDF."""
        query = query.strip()  # Handle leading/trailing spaces
        print("\n--- Ranked Search (TF-IDF) ---")
        query_terms = self.tokenize(query)
        doc_scores = {}
        total_documents = len(self.documents)

        for term in query_terms:
            if term in self.index:
                term_doc_freq = len(self.index[term])
                # IDF calculation
                idf = math.log(total_documents / (1 + term_doc_freq))
                for doc_id in self.index[term]:
                    term_freq = self.index[term].count(
                        doc_id)  # TF for term in document
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += term_freq * idf

        sorted_docs = sorted(doc_scores.items(),
                             key=lambda x: x[1], reverse=True)

        if sorted_docs:
            print(f"Ranking of documents for query '{query}':")
            for doc_id, score in sorted_docs:
                print(
                    f"Document ID {doc_id}: {self.documents[doc_id]['title']} (Score: {score:.4f})")
        else:
            print("No documents found with the ranked query.")


def load_documents(folder_path):
    """Load text documents from a folder and add them to the search engine."""
    search_engine = SearchEngine()
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                # Use filename as the title
                title = filename.replace('.txt', '')
                content = file.read()
                doc_id = len(search_engine.documents) + 1  # Assign a unique ID
                search_engine.add_document(doc_id, title, content)
    return search_engine


def main():
    folder_path = './documents'
    search_engine = load_documents(folder_path)

    while True:
        print("\n--- Document Search Engine ---")
        print("1. Search by Title")
        print("2. Search by Content")
        print("3. Boolean Search")
        print("4. Ranked Search (TF-IDF)")
        print("5. Exit")

        choice = input("Choose an option: ")

        if choice == '1':
            title_query = input("Enter the title to search for: ")
            search_engine.search_by_title(title_query)
        elif choice == '2':
            content_query = input("Enter the content to search for: ")
            search_engine.search_by_content(content_query)
        elif choice == '3':
            boolean_query = input("Enter a Boolean query (AND, OR, NOT): ")
            search_engine.boolean_search(boolean_query)
        elif choice == '4':
            ranked_query = input("Enter the content query for ranked search: ")
            search_engine.ranking_search(ranked_query)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
