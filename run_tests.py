import json
import urllib.request
import urllib.error
import sys

def run_tests():
    url = "http://127.0.0.1:8000/ask"
    
    try:
        with open("self_test_cases.json", "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Error: self_test_cases.json not found.")
        sys.exit(1)
        
    print(f"Starting evaluation of {len(test_cases)} test cases...")
    
    results = []
    
    for case in test_cases:
        case_id = case["id"]
        q_type = case["type"]
        question = case["question"]
        expected_files = case["expected_source_files"]
        expected_facts = case["expected_facts"]
        
        print(f"\n[{case_id}/15] [{q_type}] Q: {question}")
        
        payload = json.dumps({"question": question}).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=payload, 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
            answer = res_data.get("answer", "")
            citations = res_data.get("citations", [])
            trace = res_data.get("trace", [])
            
            # Evaluate Citations
            # Check if all expected files are in citations
            citation_pass = True
            for f in expected_files:
                if f not in citations:
                    citation_pass = False
                    break
            
            # For out-of-corpus, citations should be empty
            if q_type == "out_of_corpus" and len(citations) > 0:
                citation_pass = False
                
            # Evaluate Facts
            fact_matches = []
            for fact in expected_facts:
                match = fact.lower() in answer.lower()
                fact_matches.append(match)
                
            facts_pass = all(fact_matches)
            
            is_pass = citation_pass and facts_pass
            status = "PASS" if is_pass else "FAIL"
            
            print(f"  Answer: {answer}")
            print(f"  Citations: {citations} (Expected: {expected_files})")
            print(f"  Status: {status} (Citations: {'OK' if citation_pass else 'BAD'}, Facts: {'OK' if facts_pass else 'BAD'})")
            
            results.append({
                "id": case_id,
                "type": q_type,
                "question": question,
                "answer": answer,
                "citations": citations,
                "expected_source_files": expected_files,
                "status": status,
                "notes": f"Matched {sum(fact_matches)}/{len(expected_facts)} key facts."
            })
            
        except urllib.error.URLError as e:
            print(f"  Error calling API: {e}. Is the server running on http://127.0.0.1:8000?")
            results.append({
                "id": case_id,
                "type": q_type,
                "question": question,
                "answer": "ERROR: Server unreachable",
                "citations": [],
                "expected_source_files": expected_files,
                "status": "FAIL",
                "notes": f"URLError: {e}"
            })
            
    # Write results to markdown
    md_content = "# Evaluation Results\n\n"
    md_content += "Automatically generated evaluation results from running `run_tests.py`.\n\n"
    md_content += "| ID | Question | Expected Citations | Actual Citations | Status | Notes |\n"
    md_content += "|---|---|---|---|---|---|\n"
    
    passed_count = 0
    for r in results:
        status_emoji = "✅ PASS" if r["status"] == "PASS" else "❌ FAIL"
        if r["status"] == "PASS":
            passed_count += 1
        md_content += f"| {r['id']} | {r['question']} | `{', '.join(r['expected_source_files']) if r['expected_source_files'] else 'None'}` | `{', '.join(r['citations']) if r['citations'] else 'None'}` | **{status_emoji}** | {r['notes']} |\n"
        
    md_content += f"\n\n**Total Score: {passed_count}/{len(results)} Passed**\n"
    
    with open("eval_results.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\nEvaluation finished. Results written to eval_results.md. Score: {passed_count}/{len(results)}")

if __name__ == "__main__":
    run_tests()
