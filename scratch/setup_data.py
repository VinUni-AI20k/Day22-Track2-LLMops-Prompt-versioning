
import sys
import os

# Define the input file and output files
input_file = "pseudocode/step3_ragas_evaluation.py"
qa_pairs_file = "qa_pairs.py"
kb_file = "data/knowledge_base.txt"

# Extract QA_PAIRS content
qa_pairs_content = []
start_marker = "QA_PAIRS = ["
end_marker = "]"
recording = False
depth = 0

with open(input_file, "r") as f:
    for line in f:
        if start_marker in line:
            recording = True
            qa_pairs_content.append(line)
            depth += line.count("[") - line.count("]")
            continue
        
        if recording:
            qa_pairs_content.append(line)
            depth += line.count("[") - line.count("]")
            if depth == 0:
                recording = False
                break

if not qa_pairs_content:
    print("Failed to find QA_PAIRS")
    sys.exit(1)

# Create qa_pairs.py
with open(qa_pairs_file, "w") as f:
    f.write("# Generated from step3_ragas_evaluation.py\n\n")
    f.write("".join(qa_pairs_content))
    f.write("\n\nSAMPLE_QUESTIONS = [qa['question'] for qa in QA_PAIRS]\n")

print(f"Created {qa_pairs_file}")

# Create data/knowledge_base.txt
# We need to safely evaluate the list or just regex it
import ast
qa_list_str = "".join(qa_pairs_content).strip()
# Adjust string for ast.literal_eval if needed
if qa_list_str.startswith("QA_PAIRS = "):
    qa_list_str = qa_list_str[len("QA_PAIRS = "):]

try:
    qa_list = ast.literal_eval(qa_list_str)
    with open(kb_file, "w") as f:
        for item in qa_list:
            f.write(item["reference"] + "\n\n")
    print(f"Created {kb_file}")
except Exception as e:
    print(f"Error parsing QA_PAIRS: {e}")
    # Fallback to regex if ast fails
    import re
    references = re.findall(r'"reference":\s*"(.*?)"', "".join(qa_pairs_content), re.DOTALL)
    with open(kb_file, "w") as f:
        for ref in references:
            f.write(ref.replace("\\n", "\n") + "\n\n")
    print(f"Created {kb_file} via regex fallback")
