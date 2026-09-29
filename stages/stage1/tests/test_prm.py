from veriserve.prm import split_steps


def test_split_steps_keeps_exact_step_numbers_across_checkpoint():
    accepted = "Step 1: 12 / 3 = 4.\nStep 2: Add 2 to get 6."
    new = "Step 3: Multiply by 5.\nFinal answer: 30."
    assert split_steps(accepted) == [(1, "12 / 3 = 4."), (2, "Add 2 to get 6.")]
    assert split_steps(new, start=3) == [(3, "Multiply by 5.\nFinal answer: 30.")]


def test_split_steps_handles_unnumbered_final_after_checkpoint():
    assert split_steps("Final answer: 30.", start=3) == [(3, "Final answer: 30.")]
