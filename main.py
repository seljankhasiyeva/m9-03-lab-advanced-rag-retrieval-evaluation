import json
import chromadb
from sentence_transformers import SentenceTransformer
from ollama import chat
from rank_bm25 import BM25Okapi


def load_knowledge_base(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# The knowledge base is already split into meaningful chunks.
# If these were long documents (PDFs, books, etc.),
# they would need to be split into smaller chunks before creating embeddings.

def index_documents(collection, knowledge, embedding_model):

    for item in knowledge:

        embedding = embedding_model.encode(item["text"]).tolist()

        collection.add(
            ids=[item["id"]],
            documents=[item["text"]],
            embeddings=[embedding],
            metadatas=[
                {
                    "source": item["source"]
                }
            ]
        )

def retrieve_documents(collection, question, embedding_model, n_results=3):

    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    return results

def retrieve_documents_hybrid(
    collection,
    knowledge,
    bm25,
    question,
    embedding_model,
    n_results=3
):

    # -------- Dense Retrieval (Chroma) --------
    question_embedding = embedding_model.encode(question).tolist()

    dense_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    dense_ids = dense_results["ids"][0]

    # -------- BM25 Retrieval --------
    query_tokens = question.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranking = sorted(
        zip(knowledge, bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )

    bm25_ids = [doc["id"] for doc, _ in bm25_ranking[:5]]

    # -------- Merge Results --------
    merged_ids = []

    for doc_id in dense_ids + bm25_ids:
        if doc_id not in merged_ids:
            merged_ids.append(doc_id)

    merged_ids = merged_ids[:n_results]

    # -------- Return Same Format as Chroma --------
    documents = []
    metadatas = []
    ids = []

    for doc_id in merged_ids:

        item = next(doc for doc in knowledge if doc["id"] == doc_id)

        documents.append(item["text"])
        metadatas.append({"source": item["source"]})
        ids.append(item["id"])

    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas]
    }

def build_prompt(question, results):

    documents = results["documents"][0]
    sources = results["metadatas"][0]

    context = ""

    for doc, source in zip(documents, sources):
        context += f"Source: {source['source']}\n"
        context += f"{doc}\n\n"

    prompt = f"""
You are a helpful assistant.

Use ONLY the information provided in the context below.
Do NOT use your own knowledge.
If the answer is not explicitly stated in the context, reply exactly:

I don't know.

Context:
{context}

Question:
{question}

Provide a concise answer and include the source(s) you used.
"""

    return prompt

def generate_answer(prompt):

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content

def evaluate_faithfulness(
    collection,
    knowledge,
    bm25,
    embedding_model
):

    baseline_faithful = 0
    hybrid_faithful = 0

    print("\n" + "=" * 70)
    print("FAITHFULNESS EVALUATION")
    print("=" * 70)

    for sample in eval_set:

        question = sample["question"]

        systems = {
            "Baseline": retrieve_documents(
                collection,
                question,
                embedding_model
            ),
            "Hybrid": retrieve_documents_hybrid(
                collection,
                knowledge,
                bm25,
                question,
                embedding_model
            )
        }

        print(f"\nQuestion: {question}")

        for system_name, results in systems.items():

            prompt = build_prompt(question, results)
            answer = generate_answer(prompt)

            context = ""

            for doc in results["documents"][0]:
                context += doc + "\n"

            judge_prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Determine whether the generated answer is completely supported by the retrieved context.

Reply ONLY with:
YES
or
NO
"""

            judgment = chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "user",
                        "content": judge_prompt
                    }
                ]
            ).message.content.strip().upper()

            print(f"{system_name}: {judgment}")

            if judgment.startswith("YES"):
                if system_name == "Baseline":
                    baseline_faithful += 1
                else:
                    hybrid_faithful += 1

    print("\n" + "-" * 50)

    baseline_rate = baseline_faithful / len(eval_set)
    hybrid_rate = hybrid_faithful / len(eval_set)

    print(f"Baseline Faithfulness: {baseline_faithful}/{len(eval_set)} ({baseline_rate:.0%})")
    print(f"Hybrid Faithfulness: {hybrid_faithful}/{len(eval_set)} ({hybrid_rate:.0%})")

eval_set = [
    {"question": "How long do I have to get a full refund?", "expected_id": "kb-04"},
    {"question": "How do I cancel my subscription?", "expected_id": "kb-05"},
    {"question": "How do I reset my password?", "expected_id": "kb-07"},
    {"question": "What does error code 0x80070005 mean?", "expected_id": "kb-08"},
    {"question": "When is the office kitchen restocked?", "expected_id": "kb-10"},
]

def evaluate_retrieval(collection, knowledge, bm25, embedding_model):

    baseline_hits = 0
    hybrid_hits = 0

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    for sample in eval_set:

        question = sample["question"]
        expected_id = sample["expected_id"]

        # ---------- Baseline ----------
        baseline_results = retrieve_documents(
            collection,
            question,
            embedding_model
        )

        baseline_ids = baseline_results["ids"][0]

        if expected_id in baseline_ids:
            baseline_hits += 1

        # ---------- Hybrid ----------
        hybrid_results = retrieve_documents_hybrid(
            collection,
            knowledge,
            bm25,
            question,
            embedding_model
        )

        hybrid_ids = hybrid_results["ids"][0]

        if expected_id in hybrid_ids:
            hybrid_hits += 1

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_id}")
        print(f"Baseline: {baseline_ids}")
        print(f"Hybrid: {hybrid_ids}")

    print("\n" + "-" * 50)

    baseline_rate = baseline_hits / len(eval_set)
    hybrid_rate = hybrid_hits / len(eval_set)

    print(f"Baseline Hit Rate: {baseline_hits}/{len(eval_set)} ({baseline_rate:.0%})")
    print(f"Hybrid Hit Rate: {hybrid_hits}/{len(eval_set)} ({hybrid_rate:.0%})")

def main():

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()

    collection = client.create_collection(name="knowledge_base")

    knowledge = load_knowledge_base("knowledge_base.json")
    tokenized_docs = [item["text"].lower().split() for item in knowledge]
    bm25 = BM25Okapi(tokenized_docs)

    index_documents(collection, knowledge, embedding_model)

    print("Knowledge base indexed successfully!")

    questions = [
        "How long do I have to get a full refund?",
        "How do I reset my password?",
        "What is the company's stock price today?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print(f"Question: {question}\n")

        results = retrieve_documents_hybrid(
            collection,
            knowledge,
            bm25,
            question,
            embedding_model
        )
        print("Retrieved Sources:")

        documents = results["documents"][0]
        sources = results["metadatas"][0]

        for i, (doc, source) in enumerate(zip(documents, sources), start=1):
            print(f"\nResult {i}")
            print(f"Source: {source['source']}")
            print(doc)

        prompt = build_prompt(question, results)

        answer = generate_answer(prompt)

        print("\nGenerated Answer:")
        print(answer)

    evaluate_retrieval(
        collection,
        knowledge,
        bm25,
        embedding_model
    )

    evaluate_faithfulness(
        collection,
        knowledge,
        bm25,
        embedding_model
    )


if __name__ == "__main__":
    main()