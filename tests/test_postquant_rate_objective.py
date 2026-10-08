from core.ts.core.quantizemodel import postquant_rate_objective_bits


def test_arm_selection_uses_arm_and_latent_rate_only():
    objective = postquant_rate_objective_bits(
        "arm",
        candidate_module_bits=11,
        bits_per_module={"arm": 99, "upsampling": 20, "synthesis": 30},
        latent_bits=101,
        soft_label_bits=1000,
    )
    assert objective == 112


def test_decoder_selection_uses_decoder_and_soft_label_rates_only():
    objective = postquant_rate_objective_bits(
        "synthesis",
        candidate_module_bits=13,
        bits_per_module={"arm": 99, "upsampling": 17, "synthesis": 30},
        latent_bits=101,
        soft_label_bits=211,
    )
    assert objective == 241


def test_decoder_without_label_scorer_preserves_legacy_total_rate():
    objective = postquant_rate_objective_bits(
        "upsampling",
        candidate_module_bits=7,
        bits_per_module={"arm": 11, "upsampling": 20, "synthesis": 13},
        latent_bits=19,
    )
    assert objective == 50
