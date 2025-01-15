from typing import Optional


def validate_conditions(all_conditions: list[str], conditions: Optional[list[str]]):
    # check if user-defined their own conditions
    if conditions is None:
        # default to all conditions
        return all_conditions
    elif not set(conditions).issubset(all_conditions):
        raise ValueError(
            f"Unknown conditions: {conditions}. Available conditions are: {all_conditions}"
        )
    else:
        return conditions