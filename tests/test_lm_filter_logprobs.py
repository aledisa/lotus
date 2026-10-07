import math
from types import SimpleNamespace

import pytest

from lotus.models import LM


def _top(token: str, probability: float):
    """Minimal implementation of a TopLogprobs object."""
    return SimpleNamespace(token=token, logprob=math.log(probability))

def _generated_token(token: str, probability: float, top_logprobs: list[tuple[str, float]]):
    """Minimal implementation of a ChatCompletionTokenLogprob object."""
    return SimpleNamespace(
        token=token,
        logprob=math.log(probability),
        top_logprobs=[_top(t, tp) for t, tp in top_logprobs],
    )

@pytest.fixture
def lm():
    """Fixture for an empty LM object."""
    return LM.__new__(LM)

def test_filter_cascade_uses_correct_position(lm):
    """
    Test that the filter cascade mechanism uses the proper position.

    The proper position is the index of the token which is actually True or False (if any).

    Simulated output: "Answer", ":", " False". The first token has True and False in top_logprobs but
    should not be used as the position for the score computation.
    """
    # Simulated output: "Answer", ":", "False"
    logprobs = [[
        _generated_token(
            token="Answer",
            probability=0.90,
            top_logprobs=[
                ("Answer", 0.90),
                ("True", 0.09),
                ("False", 0.01),
            ],
        ),
        _generated_token(
            token=":",
            probability=0.99,
            top_logprobs=[
                (":", 0.99),
                (":\n", 0.01),
            ],
        ),
        _generated_token(
            token="False",
            probability=0.80,
            top_logprobs=[
                ("True", 0.20),
                ("False", 0.80),
            ],
        ),
    ]]

    # Use the function from LOTUS
    result = lm.format_logprobs_for_filter_cascade(logprobs)

    # The score must come from the last token and should be 0.20 
    # P(True) / (P(True) + P(False)) = 0.20 / (0.20 + 0.80) = 0.20
    assert result.positive_probs[0] == pytest.approx(0.20)


def test_filter_cascade_sums_equivalent_true_false_variants(lm):
    """
    Test that the filter cascade mechanism sums equivalent True/False variants.

    E.g., "True", " True" and "TRUE" should all be considered equivalent and their probabilities summed.
    """

    logprobs = [[
        _generated_token(
            token=" True",
            probability=0.40,
            top_logprobs=[
                ("True", 0.40),
                (" true", 0.30),
                ("False", 0.20),
                (" false", 0.10),
            ],
        ),
    ]]

    result = lm.format_logprobs_for_filter_cascade(logprobs)

    # The score should be
    # positive mass = 0.40 + 0.30 = 0.70
    # negative mass = 0.20 + 0.10 = 0.30
    # P(True) / (P(True) + P(False)) = 0.70
    assert result.positive_probs[0] == pytest.approx(0.70)

@pytest.mark.parametrize(
    ("generated_tokens", "expected"),
    [
        (
            ["Answer", ":", " Tr", "ue"],
            1.0,  # P(True) = 1.0
        ),
        (
            ["Answer", ":", " Fa", "lse"],
            0.0,  # P(True) = 0.0
        ),
        (
            ["Answer", ":", " Pingu"],
            0.5,  # P(True) = 0.5 since it is not True or False, we assume equal probability

        ),
    ],
)

def test_filter_cascade_handles_partial_tokens(lm, generated_tokens, expected):
    """
    Test that the filter cascade mechanism handles partial tokens correctly.

    E.g., "Tr" and "ue" should be combined to form "True", and "Fa" and "lse" should be combined to form "False".
    """

    response_logprobs = []

    for token in generated_tokens:
        response_logprobs.append(
            _generated_token(
                token=token,
                probability=0.9,
                top_logprobs=[
                    (token, 0.9),
                    ("other", 0.1),
                ],
            )
        )

    logprobs = [response_logprobs]

    result = lm.format_logprobs_for_filter_cascade(logprobs)

    assert result.positive_probs[0] == pytest.approx(expected)