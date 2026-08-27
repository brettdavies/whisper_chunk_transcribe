import pytest
from whisper_chunk_transcribe.helper_classes import ExpTestCase

def test_exp_test_case_initialization():
    """Test basic initialization of ExpTestCase"""
    test_case = ExpTestCase(
        experiment_id=1,
        test_case_id=1,
        prompt_template="Simple prompt",
        prompt_tokens=5
    )
    
    assert test_case.experiment_id == 1
    assert test_case.test_case_id == 1
    assert test_case.prompt_template == "Simple prompt"
    assert test_case.prompt_tokens == 5
    assert not test_case.is_dynamic
    assert not test_case.use_prev_transcription

def test_exp_test_case_dynamic_detection():
    """Test detection of dynamic prompts"""
    test_case = ExpTestCase(
        experiment_id=1,
        test_case_id=1,
        prompt_template="Test game between {teamA} and {teamB}",
        prompt_tokens=10
    )
    
    assert test_case.is_dynamic
    assert not test_case.use_prev_transcription

def test_exp_test_case_prev_transcription_detection():
    """Test detection of previous transcription usage"""
    test_case = ExpTestCase(
        experiment_id=1,
        test_case_id=1,
        prompt_template="Previous: {previous_transcription}",
        prompt_tokens=10
    )
    
    assert test_case.is_dynamic
    assert test_case.use_prev_transcription

def test_exp_test_case_invalid_prompt():
    """Test handling of invalid prompt templates"""
    with pytest.raises(ValueError):
        ExpTestCase(
            experiment_id=1,
            test_case_id=1,
            prompt_template=123,  # Invalid type
            prompt_tokens=10
        ) 