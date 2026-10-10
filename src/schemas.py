from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int = 0
    score: float = 0.0

    def to_source(self, snippet_chars: int = 240) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 3),
            "snippet": self.text[:snippet_chars],
        }


@dataclass
class GradeResult:
    relevant: bool
    reason: str = ""


@dataclass
class VerifyResult:
    supported_ratio: float
    answers_question: bool
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass
class Result:
    question: str
    answer: str
    mode: str = "self_rag"  # "self_rag" | "baseline"
    sources: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    attempts: int = 0
    refused: bool = False
    used_retrieval: bool = True
    trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
