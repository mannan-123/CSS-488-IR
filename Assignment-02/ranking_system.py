import os
import math
import re
from collections import Counter

# Step 1: Gather Your Documents


def load_documents(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                documents[filename] = file.read()
    return documents

# Step 3: Query Function


def query_documents(query, documents, method='keyword_matching'):
    query = preprocess_text(query)
    if method == 'keyword_matching':
        return keyword_matching(query, documents)
    elif method == 'tf_idf':
        return tf_idf_ranking(query, documents)
    elif method == 'cosine_similarity':
        return cosine_similarity_ranking(query, documents)
    else:
        raise ValueError("Unknown method specified")

# Step 4: Keyword Matching


def preprocess_text(text):
    # Convert text to lowercase and remove non-alphabetical characters
    text = text.lower()
    return re.sub(r'[^a-z\s]', '', text)


def keyword_matching(query, documents):
    query_keywords = query.split()
    scores = {}
    for doc_name, doc_content in documents.items():
        doc_keywords = preprocess_text(doc_content).split()
        matched_keywords = set(query_keywords).intersection(set(doc_keywords))
        scores[doc_name] = len(matched_keywords)
    ranked_docs = sorted(
        scores.items(), key=lambda item: item[1], reverse=True)
    return ranked_docs

# Step 5: TF-IDF Scoring


def compute_tf(doc):
    tf = Counter(doc.split())
    total_terms = len(doc.split())
    for term in tf:
        tf[term] = tf[term] / total_terms
    return tf


def compute_idf(documents):
    idf = {}
    total_docs = len(documents)
    for doc in documents.values():
        terms_in_doc = set(doc.split())
        for term in terms_in_doc:
            idf[term] = idf.get(term, 0) + 1
    for term in idf:
        idf[term] = math.log(total_docs / 1 + idf[term])
    return idf


def tf_idf_ranking(query, documents):
    query_terms = preprocess_text(query).split()
    tfidf_scores = {}
    tf = compute_tf(' '.join(query_terms))
    idf = compute_idf(documents)

    for doc_name, doc_content in documents.items():
        doc_tf = compute_tf(doc_content)
        score = 0
        for term in query_terms:
            if term in doc_tf:
                # score += doc_tf[term] * idf.get(term, 0) * tf.get(term, 0)
                score += doc_tf[term] * idf.get(term, 0)
        tfidf_scores[doc_name] = score
    ranked_docs = sorted(tfidf_scores.items(),
                         key=lambda item: item[1], reverse=True)
    return ranked_docs

# Step 6: Cosine Similarity


def cosine_similarity_ranking(query, documents):
    query_vector = build_vector(preprocess_text(query), documents)
    cosine_scores = {}
    for doc_name, doc_content in documents.items():
        doc_vector = build_vector(preprocess_text(doc_content), documents)
        cosine_scores[doc_name] = cosine_similarity(query_vector, doc_vector)
    ranked_docs = sorted(cosine_scores.items(),
                         key=lambda item: item[1], reverse=True)
    return ranked_docs


def build_vector(query, documents):
    vector = Counter(query.split())
    all_terms = set()
    for doc_content in documents.values():
        all_terms.update(doc_content.split())
    return {term: vector.get(term, 0) for term in all_terms}


def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1)
    magnitude1 = math.sqrt(sum(val**2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val**2 for val in vec2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# Step 7: Display Results


def display_ranked_documents(ranked_docs):
    for doc_name, score in ranked_docs:
        print(f"{doc_name} - Relevance Score: {score}")

# Step 8: User Interaction


def main():
    folder_path = "documents"
    documents = load_documents(folder_path)

    while True:
        print("\nSelect ranking method:")
        print("1. Keyword Matching")
        print("2. TF-IDF")
        print("3. Cosine Similarity")
        print("4. Quit")
        choice = input("Enter the method number: ").strip()

        if choice == '4':
            break

        query = input("Enter your query: ").strip()

        if choice == '1':
            ranked_docs = query_documents(
                query, documents, method='keyword_matching')
        elif choice == '2':
            ranked_docs = query_documents(query, documents, method='tf_idf')
        elif choice == '3':
            ranked_docs = query_documents(
                query, documents, method='cosine_similarity')
        else:
            print("Invalid choice. Try again.")
            continue

        display_ranked_documents(ranked_docs)


if __name__ == "__main__":
    main()
