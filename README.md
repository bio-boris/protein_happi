# Protein Happi

AI-powered protein sequence similarity search API using GPU-accelerated vector search.

## API Endpoints

### Search Jobs (Async)

The API now supports asynchronous job-based search operations:

#### Submit Search Job
`POST /search`

Submits a search job and returns immediately with a job ID. The search runs in the background using cached GPU resources.

**Request Body:**
```json
{
  "query_sequences": [
    {
      "id": "query1", 
      "sequence": "ACDEFGHIKLMNPQRSTVWY"
    }
  ],
  "max_hits": 5,
  "similarity_threshold": 0.0,
  "best_hit_only": false,
  "return_query_embeddings": false,
  "return_hit_embeddings": false
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "submitted_at": 1234567890.123
}
```

#### Check Job Status
`GET /job/{job_id}`

Returns the current status and results (if completed) of a search job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "submitted_at": 1234567890.123,
  "started_at": 1234567890.456,
  "completed_at": 1234567890.789,
  "error_message": null,
  "results": {
    "hits": [/* search results */]
  }
}
```

**Job Status Values:**
- `pending`: Job queued but not started
- `running`: Job currently executing
- `completed`: Job finished successfully
- `failed`: Job encountered an error

# Notes

* install packages with `uv add <package-name>`
* python3.12 -m venv .venv ; source .venv/bin/activate; 
* uv sync (caveat, you may need to ensure your .python_version file is correct and you have brew dependencies installed)
* install protein_search_evals [install.sh](install.sh)


# Brew Requirements for sentencepiece
* python 3.12 as of 03/14/2025
* brew install procs cmake coreutils protobuf sentencepiece


## To Do:

- [x] Update this README.md with info about your repository (async API endpoints)
- [ ] Modify `Dockerfile` with needed steps (assuming repo produces a Docker image)
- [ ] Ensure all [branch rules](https://github.com/kbase/.github/blob/develop/guide/enable-branch-rules.md) & [status checks](https://github.com/kbase/.github/blob/develop/guide/enable-branch-rules.md#require-status-checks) are enabled
