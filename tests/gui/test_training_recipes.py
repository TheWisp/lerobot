def test_every_policy_offers_the_saved_mask_escape_hatch():
    """Whether a run trains on a dataset's stored masks is a property of the
    DATASET, not of the policy, and both trainers honour it — the same argument
    that puts the camera picker on every recipe. It was HVLA-only, so an ACT,
    Diffusion or SmolVLA run had no way to compare against raw pixels except a
    CLI flag the form cannot express.
    """
    from lerobot.gui.api.training import list_policies

    catalog = list_policies()
    assert catalog, "no recipes in the catalog"
    for entry in catalog:
        names = [f["name"] for f in entry.get("fields", [])]
        assert names.count("ignore_saved_masks") == 1, (
            f"{entry.get('type_name') or entry.get('policy_type')}: expected exactly one "
            f"saved-mask field, found {names.count('ignore_saved_masks')}"
        )


def test_the_two_trainers_spell_the_mask_flag_oppositely():
    """HVLA takes `--ignore-saved-masks` (store_true); `lerobot-train` reads
    `DatasetConfig.apply_saved_masks`, which defaults to True. One box, so the
    draccus side declares `negate` — and sending the ticked value straight
    through there would train WITH masks on a run asking to ignore them.
    """
    from lerobot.gui.api.training import list_policies

    for entry in list_policies():
        field = next(f for f in entry["fields"] if f["name"] == "ignore_saved_masks")
        if entry.get("recipe"):  # HVLA: bare key, store_true, no inversion
            assert "arg_key" not in field, f"HVLA must keep the bare key: {field}"
            assert not field.get("negate"), "HVLA's flag already means 'ignore'"
        else:  # draccus: the opposite dataclass field, so it must invert
            assert field.get("arg_key") == "dataset.apply_saved_masks", field
            assert field.get("negate") is True, (
                "the draccus field is `apply_saved_masks` but the box asks to IGNORE; "
                "without negate a ticked box trains WITH masks"
            )
