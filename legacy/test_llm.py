from generator import generate_candidate_raw, parse_delimiters

raw = generate_candidate_raw("Dynamic Programming & Monotonic Queue", 42)
print("=== RAW LLM RESPONSE ===")
print(raw[:1000])
print("\n=== PARSE RESULT ===")
parsed = parse_delimiters(raw)
print(parsed)
