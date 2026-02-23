import torch


def test_extract_family_flow_features_bidirectional(test_model):
    model = test_model.eval()
    aa_seq = "ACDEFGHIK"

    with torch.no_grad():
        outputs = model.extract_family_flow_features(aa_seq=aa_seq, wt_name="WT")

    e_fwd = outputs["E_fwd"]
    e_rev = outputs["E_rev"]
    e_bwd = outputs["E_bwd"]
    features = outputs["features"]
    z_concat = outputs["Zconcat"]

    assert outputs["wt_name"] == "WT"
    assert e_fwd.shape[:2] == (1, len(aa_seq))
    assert e_rev.shape == e_fwd.shape
    assert e_bwd.shape == e_fwd.shape
    assert features.shape == (1, len(aa_seq), e_fwd.shape[-1] * 2)
    assert torch.allclose(e_bwd, torch.flip(e_rev, dims=(1,)))
    assert torch.allclose(features[..., : e_fwd.shape[-1]], e_fwd)
    assert torch.allclose(features[..., e_fwd.shape[-1] :], e_bwd)
    assert torch.allclose(z_concat, features)


def test_extract_family_flow_features_with_projection(test_model):
    model = test_model.eval()
    aa_seq = "MNPQRSTVW"
    projection_layer = torch.nn.Linear(model.model.config.hidden_size * 2, 1280)

    with torch.no_grad():
        outputs = model.extract_family_flow_features(
            aa_seq=aa_seq,
            wt_name="WT2",
            projection_layer=projection_layer,
        )

    assert "projected_features" in outputs
    assert "Zfam" in outputs
    assert outputs["projected_features"].shape == (1, len(aa_seq), 1280)
    assert torch.allclose(outputs["projected_features"], outputs["Zfam"])


def test_extract_family_flow_features_batched_and_mask(test_model):
    model = test_model.eval()
    seqs = ["ACDEFG", "MNPQ"]

    with torch.no_grad():
        outputs = model.extract_family_flow_features(
            aa_seq=seqs,
            wt_name=["WT_A", "WT_B"],
            projection_dim=1280,
        )

    assert outputs["wt_name"] == ["WT_A", "WT_B"]
    assert outputs["seq_lengths"] == [len(seqs[0]), len(seqs[1])]
    assert outputs["features"].shape == (2, len(seqs[0]), model.model.config.hidden_size * 2)
    assert outputs["projected_features"].shape == (2, len(seqs[0]), 1280)
    assert torch.allclose(outputs["projected_features"], outputs["Zfam"])

    mask = outputs["seq_mask"]
    assert mask.shape == (2, len(seqs[0]))
    assert mask[0].sum().item() == len(seqs[0])
    assert mask[1].sum().item() == len(seqs[1])


def test_extract_family_flow_features_trainable_projection(test_model):
    model = test_model.eval()
    outputs = model.extract_family_flow_features(
        aa_seq="ACDEFG",
        projection_dim=1280,
        inference_mode=False,
    )
    assert outputs["projected_features"].requires_grad


def test_extract_family_flow_features_projection_args_conflict(test_model):
    model = test_model.eval()
    projection_layer = torch.nn.Linear(model.model.config.hidden_size * 2, 1280)
    try:
        model.extract_family_flow_features(
            aa_seq="ACDEFG",
            projection_layer=projection_layer,
            projection_dim=1280,
        )
    except ValueError as exc:
        assert "either projection_layer or projection_dim" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting projection args")
