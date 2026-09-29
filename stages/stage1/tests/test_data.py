from veriserve.data import gold_answer, is_correct, predicted_answer


def test_numeric_scoring():
    assert gold_answer("work\n#### 1,234") == "1,234"
    assert predicted_answer("Step 1: work\nFinal answer: 1,234.\n") == "1,234"
    assert is_correct("1/2", "0.5")
    assert not is_correct(None, "2")
