# Jev in this benchmark

- **Credential:** `TYPESAFE_API_KEY` in the environment (console.typesafe.ai). Never in a file here.
- **SDK and model:** `typesafe-sdk` (Python, 0.7.4 when measured), `jev-latest`, which served `jev-1.13.0` on 2026-10-14.
- **Judgment:** one Choice per position, ID `p<t>`: "Written so far: "…". Which symbol comes next?", 28 options (`end`, `space`, a–z). The shared state is the task description in `questions.py`. One request per name.
- **Stays in code:** the loss, the floor, top-pick accuracy.
- **Measured (300 names, 3,125 predictions, 2026-10-14):** probabilities come rounded to 2 decimals; 159 real answers got 0 (154 of them the first letter). Loss 2.9849 with 0 read as 0.005, 3.4183 read as 1e-6. Top pick right 17.6%. 2,712 input tokens per name, $0.034 for 300 names ($0.114 per 1,000), median 307 ms per name over the network.
