import os
import re
from collections import defaultdict

# Step 1: Define Stopwords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of',
    'to', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does',
    'did', 'but', 'not', 'that', 'this', 'it', 'its', 'they', 'them', 'he', 'she', 'his', 'her',
    'you', 'we', 'I', 'me', 'my', 'mine', 'your', 'yours', 'their', 'our', 'us', 'can', 'will',
    'just', 'if', 'so', 'no', 'yes', 'all', 'any', 'some', 'which', 'who', 'whom', 'how', 'why',
    'then', 'than', 'now', 'up', 'down', 'out', 'when', 'where'
}

# Step 2: Define the SearchEngine Class


class SearchEngine:
    def __init__(self):
        self.index = defaultdict(list)  # Dictionary to store the index
        self.documents = {}  # Store document titles and content by ID

    def add_document(self, doc_id, title, content):
        """Add a document's title and content to the search engine."""
        self.documents[doc_id] = {'title': title, 'content': content}
        words = self.tokenize(content)
        for word in words:
            if word not in STOPWORDS:
                if doc_id not in self.index[word]:
                    self.index[word].append(doc_id)

    def tokenize(self, text):
        """Convert text to lowercase, remove punctuation, split into words, and remove stopwords."""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
        words = text.lower().split()  # Split into words and lowercase
        return [word for word in words if word not in STOPWORDS]

    def search_by_title(self, title_query):
        """Search for a document by title."""
        for doc_id, doc in self.documents.items():
            if title_query.lower() in doc['title'].lower():
                print(f"Found in document ID {doc_id}: {doc['title']}")

    def search_by_content(self, content_query):
        """Search for documents by content."""
        query_words = self.tokenize(content_query)
        result_docs = set()

        for word in query_words:
            if word in self.index:
                result_docs.update(self.index[word])

        if result_docs:
            print(f"Documents containing '{content_query}':")
            for doc_id in result_docs:
                print(
                    f"Document ID {doc_id}: {self.documents[doc_id]['title']}")
        else:
            print("No documents found containing the query.")

# Step 3: Load and Index Documents


def load_documents(folder_path):
    search_engine = SearchEngine()
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                first_line = file.readline().strip()  # Read the first line as part of the title
                # Combine filename and first line
                title = f"{filename.replace('.txt', '')} - {first_line}"
                content = file.read()  # Read the rest of the document content
                doc_id = len(search_engine.documents) + 1  # Assign a unique ID
                search_engine.add_document(doc_id, title, content)
    return search_engine

# Step 4: User Interaction Functions


def main():
    folder_path = './documents'  # Folder where the text documents are stored
    search_engine = load_documents(folder_path)

    while True:
        print("\n--- Simple Document Search Engine ---")
        print("1. Search by Title")
        print("2. Search by Content")
        print("3. Exit")

        choice = input("Choose an option: ")

        if choice == '1':
            title_query = input("Enter the title to search for: ")
            search_engine.search_by_title(title_query)
        elif choice == '2':
            content_query = input("Enter the content to search for: ")
            search_engine.search_by_content(content_query)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


# Step 5: Run the Program
if __name__ == "__main__":
    main()
