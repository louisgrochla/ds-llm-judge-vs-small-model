# Build-vs-buy decision table

For Banking77-style intent classification, comparing self-hosted fine-tuned DistilBERT against Claude Sonnet 4.6 (zero-shot via API).

Assumptions:
- API per-query cost: $0.0053 (no prompt caching). With caching: $0.0015.
- Self-host fixed cost: $360.0/month (Hugging Face Inference Endpoint, T4).
- Labeling cost: $0.5/example (internal labeller, conservative).
- Quality threshold: macro-F1 ≥ Sonnet baseline (0.8913) at n ≥ 2,500 (ensemble).

| Scenario | n labels | queries/mo | DistilBERT macro-F1 | Meets Sonnet? | API $/mo | Self-host $/mo | Recommendation |
|---|---|---|---|---|---|---|---|
| Hobby / prototype | 100 | 1,000 | 0.384 | ✗ | $5 | $360 | API (quality not yet competitive at this n) |
| Small B2B SaaS | 1,000 | 50,000 | 0.866 | ✗ | $262 | $360 | API (quality not yet competitive at this n) |
| Mid-market fintech | 2,500 | 500,000 | 0.91 | ✓ | $2,625 | $360 | Self-host (saves $2,265/mo; labels pay back in 0.6 mo) |
| Scaled product (Monzo/Revolut tier) | 5,000 | 5,000,000 | 0.927 | ✓ | $26,250 | $360 | Self-host (saves $25,890/mo; labels pay back in 0.1 mo) |
| Public-cloud SaaS | 9,000 | 50,000,000 | 0.934 | ✓ | $262,500 | $360 | Self-host (saves $262,140/mo; labels pay back in 0.0 mo) |