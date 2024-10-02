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
