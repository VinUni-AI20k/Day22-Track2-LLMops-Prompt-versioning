from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

print(f"faithfulness: {type(faithfulness)}")
print(f"answer_relevancy: {type(answer_relevancy)}")
print(f"context_recall: {type(context_recall)}")
print(f"context_precision: {type(context_precision)}")
