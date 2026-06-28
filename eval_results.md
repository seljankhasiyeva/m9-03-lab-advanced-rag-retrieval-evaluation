# Evaluation Results

## Retrieval Upgrade

For this lab, I implemented **hybrid retrieval** by combining semantic vector search (ChromaDB embeddings) with BM25 keyword search. The final retrieved documents are obtained by merging the top results from both retrieval methods.

## Evaluation Set

| Question                                 | Expected Passage |
| ---------------------------------------- | ---------------- |
| How long do I have to get a full refund? | kb-04            |
| How do I cancel my subscription?         | kb-05            |
| How do I reset my password?              | kb-07            |
| What does error code 0x80070005 mean?    | kb-08            |
| When is the office kitchen restocked?    | kb-10            |

## Results

| Metric             | Baseline   | Hybrid     |
| ------------------ | ---------- | ---------- |
| Retrieval Hit Rate | 5/5 (100%) | 5/5 (100%) |
| Faithfulness       | 1/5 (20%)  | 2/5 (40%)  |

## Conclusion

Both the baseline dense retriever and the hybrid retriever achieved a 100% retrieval hit rate on the evaluation set, meaning the expected passages were successfully retrieved for all questions. However, the LLM-as-judge evaluation showed an improvement in faithfulness, increasing from 20% to 40% after introducing hybrid retrieval. Although retrieval accuracy remained unchanged, the hybrid approach produced answers that were judged to be better supported by the retrieved context.
