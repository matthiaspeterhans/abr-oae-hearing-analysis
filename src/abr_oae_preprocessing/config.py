from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    # Epoching
    epoch_ms: float = 41.5

    # ABR bandpass; OAE remains unfiltered.
    filter_abr: bool = True
    abr_low_hz: float = 60.0
    abr_high_hz: float = 3000.0
    filter_order: int = 4

    # Averaging
    average_mode: str = "block_weighted"  # "mean" or "block_weighted"
    split_within_each_record: bool = True
    block_size_trials: int | None = 75

    # ABR artifacts: fixed 50-uV threshold or robust MAD threshold only.
    artifact_threshold_uv: float = 50.0
    artifact_policy_by_dataset: dict = field(default_factory=lambda: {
        "earndb": "threshold_uv",
        "earh": "threshold_uv",
    })
    mad_k: float = 3.0

    # EARH trigger detection from the trigger channel.
    earh_trigger_threshold_fraction: float = 0.30
    earh_trigger_min_distance_ms: float = 30.0

    # Header and scaling corrections.
    correct_oae_gain: bool = True
    gain_overrides: dict = field(default_factory=dict)
    gain_multipliers_by_dataset_role: dict = field(default_factory=lambda: {
        "earh": {"abr": 256.0, "oae": 256.0},
    })

    eps: float = 1e-12
