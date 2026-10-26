# Memory Layer Pro pipeline

```mermaid
flowchart TD
    A[Select synthetic demo user & session] --> B[Incoming user message]
    B --> C[Local PII / secret screening]
    C -->|masked message| D[Read bounded Redis working context]
    D --> E[Screen context before prompt use]
    E --> F[Embed protected query locally]

    F --> G{Retrieval enabled?}
    G -->|Yes| H[Retrieve top-k user-scoped durable memories from pgvector]
    H --> I[Screen retrieved memories]
    G -->|No| J[Exclude durable memories from prompt]

    I --> K[Build bounded, redacted prompt]
    J --> K
    K --> L[One hosted Nemotron call]
    L --> M[Structured result:<br/>answer + memory proposals]
    M --> N[Python parses, screens & validates]
    N --> O{Policy accepts proposal?}

    O -->|Accepted| P[Embed accepted proposal locally]
    O -->|Contradicts stored memory| W[Create SUPERSEDE link:<br/>old memory → new proposal]
    W --> P
    P --> Q[Write durable memory to PostgreSQL<br/>with created_at / updated_at / valid_from]
    O -->|Rejected| R[Record decision & reason]
    Q --> Q2[Append memory event:<br/>ADD / UPDATE / SUPERSEDE]
    R --> R2[Append memory event:<br/>REJECT + reason]
    Q2 --> S{Database confirms write?}
    R2 --> T
    S -->|Yes| T[Append protected user / assistant turns to Redis with TTL]
    S -->|No| U[Show answer; flag memory as unsaved]
    T --> V[Return answer, retrievals, decisions,<br/>events & timed trace]
    U --> V

    C -. privacy failure .-> X[Stop safely; show redacted error trace]
    D -. context failure .-> X
    F -. embedding failure .-> X
    H -. retrieval failure .-> X
    L -. provider / invalid output .-> X
    X --> V

    classDef local fill:#E8F3FF,stroke:#2563EB,color:#111827;
    classDef external fill:#FFF7E6,stroke:#D97706,color:#111827;
    classDef decision fill:#F3E8FF,stroke:#9333EA,color:#111827;
    classDef safe fill:#ECFDF5,stroke:#059669,color:#111827;
    class C,D,E,F,I,K,N,O,P,R,T,W,Q2,R2 local;
    class H,Q,L external;
    class G,O,S decision;
    class V,X,U safe;
```

**Legend:** blue = application/local processing, amber = external service,
purple = branch or policy decision, green = learner-visible outcome.

The privacy screen runs before content can reach Redis, pgvector, logs/traces,
event rows, or the hosted model. The model is called exactly once for a successful turn;
Python makes the memory-persistence decisions. Every decision appends one redacted
event row, and contradictions preserve history through SUPERSEDE links instead of
deleting it.
