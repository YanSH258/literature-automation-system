from lcr.validation.structural import structural_check

def test_valid_answer():
    answer = "This is a sentence [1]. Another sentence [2]."
    result = structural_check(answer, 2)
    assert result.passed is True
    assert result.issues == []
    assert result.uncited_sentences == []
    assert set(result.cited_indices) == {1, 2}

def test_out_of_range_indices():
    answer = "This sentence has an out of range index [5]."
    result = structural_check(answer, 3)
    assert result.passed is False
    assert any("out_of_range_indices" in issue for issue in result.issues)
    assert result.cited_indices == [5]

def test_forbidden_pattern():
    answer = "This sentence uses a forbidden format [1,2]."
    result = structural_check(answer, 5)
    assert result.passed is False
    assert any("forbidden_pattern" in issue for issue in result.issues)

def test_uncited_sentence():
    answer = "This sentence is cited [1]. This one is not."
    result = structural_check(answer, 2)
    assert result.passed is False
    assert "This one is not." in result.uncited_sentences
    assert result.cited_indices == [1]

def test_empty_answer():
    answer = ""
    result = structural_check(answer, 5)
    assert result.passed is True
    assert result.issues == []
    assert result.uncited_sentences == []
    assert result.cited_indices == []
