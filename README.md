# podcast-db-updater
Pipeline downloading podcasts, converting it from audio to text, chunk the text to store it in a chromadb.


# Technical choices
1. The chormadb embedding function
1.1. The key constraints:
1.1.1. Size of the data
around 23000 sentences
rationnal:
100 podcasts with a mean of 20 minutes 
A native french speaker is around 160 words/min ==> 3200 words per 20 min
A mean of 15-18 words per sentence ==> around 213-230 sentences per podcasts 
This gives a  23000 sentences

| Constraint | Requirement |
|------------|-------------|
| Data size | 10,000 - 100,000 French sentences |
| Data updates | Weekly (can be scheduled) |
| Query frequency | Hundreds/day (needs fast response) |
| Response time | < 15 seconds acceptable |
| Cost | Free or near-free |
| Containers | Must be very small for fast deployment |

Solution retained ==>
Use HuggingFace's free Inference API for embeddings instead of running the model locally.
Free Tier Limits:

~30,000 requests/month
Rate limited (not instant, but fast)
Supports sentence-transformers models


### Pros & Cons

| Pros | Cons |
|------|------|
| ✅ Container ~100MB (vs 7GB) | ⚠️ 30K requests/month free |
| ✅ Cold start ~5 seconds | ⚠️ Depends on external service |
| ✅ No GPU needed | ⚠️ Slight latency (~500ms/request) |
| ✅ Free | ⚠️ Need internet access |
