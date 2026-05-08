try:
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecision,
    )
    print("Successfully imported classes from ragas.metrics")
    print(f"Faithfulness: {Faithfulness}")
except ImportError as e:
    print(f"ImportError: {e}")

try:
    import ragas.metrics as rm
    print(f"ragas.metrics attributes: {dir(rm)}")
except Exception as e:
    print(f"Error: {e}")
