## RAG enhanced LLM
The synergy of RAG enhances the LLM ability to generate responses that are not only coherent and contextually appropriate but also enriched with the latest information and data,
making it valuable for applications that require higher levels of accuracy and specificity, such as customer support, research assistance, and specialized chatbots.
### RAG vs Fine tuning & When
* RAG: there is a desire to user external data in supportive capacity, or to be used as the centerpiece of the response (RCG). Good at Dynamic or Evolving Content, Generalization over Specialization, has resource constraints.
* Fine-tuning: adjusting model's parameters on a domain-specific dataset. high performance, control over Data, not need for real-time updates
### How to combine RAG with LLM: step by step
1. creating index of vectorized documents: create vector store containing the embedding of docs
2. RAG system use semantic search to locate close documents for the query, (cosine similarity, Nearest Neighbor)
3. send matching documents with use's original prompt to the LLM

## Rag knowledge learning
* Embedding models: he lower the dimensionality of the underlying vectors, the more compact the representation is in embedding space, which can affect downstream task quality. Sentence Transformers (sbert) provides embedding models with a dimension *n* in the range of 384, 512 and 768, and the models are completely free and open-source. OpenAI and Cohere embeddings, which require a paid API call to generate them, can be considered higher quality due to a dimensionality of a few thousand. One reason it makes sense to use a paid API to generate embeddings is if your data is multilingual.
* Performance turnning: 1, try different chunk sizes, chunk overlap, and chunking strategies; 2, Adding meta-data for filtering, One commonly used metadata tag is the “date” because it facilitates filtering by recency; 3, Structured Retrieval for Larger Document Sets, to embed document summaries and establish a mapping to text chunks within each document. This enables retrieval at the document level initially, prioritizing the identification of relevant documents before delving into chunk-level analysis.
* chunking and optimization: https://dev.to/peterabel/what-chunk-size-and-chunk-overlap-should-you-use-4338
