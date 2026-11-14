# Test Questions for RAG Chatbot

Ingest `data/` first, then ask each question. Compare with the expected answer.

## A. Simple fact retrieval (single chunk)
| # | Question | Expected answer | Source |
|---|----------|-----------------|--------|
| 1 | Who founded Zentora Labs and when? | Maya Okonkwo and Daniel Reyes, 2019 | company_handbook.md |
| 2 | How many days of annual leave do employees get? | 28 days (5 can be carried over) | company_handbook.md |
| 3 | What is the price of the Growth plan? | 49 USD per agent / month | product_faq.md |
| 4 | How long are server logs retained? | 90 days | security_policy.txt |
| 5 | Who is the CISO? | Tomas Lindqvist | security_policy.txt |

## B. Numbers and details
| # | Question | Expected answer |
|---|----------|-----------------|
| 6 | What is the API rate limit for Enterprise? | 600 requests per minute |
| 7 | How much is the home office stipend? | 600 EUR one-time (+ 40 EUR/month internet) |
| 8 | What uptime does Enterprise SLA guarantee and what is the credit? | 99.95%, 10% service credit |
| 9 | What is the max file attachment size? | 25 MB per file, all plans |

## C. Multi-chunk / multi-document reasoning
| # | Question | Expected answer |
|---|----------|-----------------|
| 10 | Which plan do I need to use the Pilot AI assistant, and what does it cost with annual billing? | Enterprise, 99 USD/agent/month minus 20% = 79.20 USD |
| 11 | If a SEV-1 incident happens, how fast must the team respond and when are customers notified? | Within 15 min; customers within 72 hours |
| 12 | Can a Starter customer use the API or get phone support? | No API access; only email support (48h) |
| 13 | How many days can employees work remotely, and who approves fully remote contracts? | 3 days/week; department head + HR director (Priya Nair) |

## D. Multi-turn / follow-up (tests chat history)
1. "What plans does NimbusDesk have?" -> lists Starter, Growth, Enterprise
2. "Which of them has a free trial?" -> Starter and Growth only
3. "And how many agents can the first one have?" -> Starter: up to 5 agents

## E. Out-of-scope (must answer "I don't know")
| # | Question | Expected behavior |
|---|----------|-------------------|
| 14 | What is the CEO's salary? | "I don't know" |
| 15 | What is the capital of France? | "I don't know" (not in documents) |
| 16 | Does NimbusDesk support WhatsApp? | "I don't know" (not mentioned) |
| 17 | Who is the CTO of Zentora Labs? | "I don't know" |

## F. Tricky / hallucination traps
| # | Question | Expected answer |
|---|----------|-----------------|
| 18 | What is the free trial length for Enterprise? | No free trial; requires a demo call |
| 19 | Is Zentora Labs headquartered in Lisbon? | No: moved to Berlin in March 2022 |
| 20 | Can I get a refund on an annual plan after 45 days? | No (only within 30 days) |

## Pass criteria
- A-D: correct answer + correct source filename cited.
- E: refuses to guess, no invented facts.
- F: does not fall for the false premise.
