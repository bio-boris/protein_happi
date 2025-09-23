


class SearchJobResponse(BaseModel):
    job_id: str


class SearchResultRequest(BaseModel):
    job_id: str


class SearchResultResponse(BaseModel):
    status: str
    result: SearchResponse | None = None
    error: str | None = None


class SequenceModel(BaseModel):
    """The model for a sequence."""

    id: str = Field(..., description="The identifier of the sequence.")
    sequence: str = Field(..., description="The sequence itself.")


class HitModel(BaseModel):
    """The model for a single hit sequence."""

    id: str = Field(..., description="The hit Uniprot ID.")
    score: float = Field(..., description="The similarity score.")
    embedding: list[float] = Field(
        default_factory=list, description="The normalized hit embedding."
    )


class HitsModel(BaseModel):
    """The model for all hits."""

    query_id: str = Field(..., description="The query sequence ID.")
    best_hit: HitModel = Field(..., description="The best hit.")
    hits: list[HitModel] = Field(
        default_factory=list, description="The hits sorted by relevance."
    )
    query_embedding: list[float] = Field(
        default_factory=list, description="The normalized query embedding."
    )
    total_hits: int = Field(..., description="The total number of hits.")


class SearchRequest(BaseModel):
    """The request model for the search endpoint."""

    query_sequences: list[SequenceModel] = Field(
        ..., description="The query sequences."
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="The similarity threshold to use (by default returns max_hits number of hits).",
    )
    best_hit_only: bool = Field(
        default=False,
        description="Whether to return only the best hit for each sequence.",
    )
    max_hits: int = Field(
        default=5, ge=1, le=100, description="The maximum number of hits to return."
    )
    return_query_embeddings: bool = Field(
        default=False, description="Whether to return the normalized query embeddings."
    )
    return_hit_embeddings: bool = Field(
        default=False, description="Whether to return the normalized hit embeddings."
    )

    # Add a validator if best_hit_only is True, then set max_hits to 1
    @model_validator(mode="after")
    def validate_best_hit_only(self) -> Self:
        """Validate the best_hit_only field."""
        if self.best_hit_only:
            self.max_hits = 1
        return self


class SearchResponse(BaseModel):
    """The response model for the search endpoint."""

    hits: list[HitsModel] = Field(..., description="The hits for each query sequence.")
