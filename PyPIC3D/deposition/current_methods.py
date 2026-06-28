CURRENT_J_FROM_RHOV = 0
CURRENT_ESIRKEPOV = 1


def encode_current_calculation(current_calculation):
    current_codes = {
        "j_from_rhov": CURRENT_J_FROM_RHOV,
        "esirkepov": CURRENT_ESIRKEPOV,
    }
    if current_calculation not in current_codes:
        raise ValueError("Unsupported current_calculation. Use 'j_from_rhov' or 'esirkepov'.")
    return current_codes[current_calculation]
