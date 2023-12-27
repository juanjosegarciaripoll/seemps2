if False:
    from .oldcore import (
        Strategy,
        DEFAULT_TOLERANCE,
        DEFAULT_STRATEGY,
        NO_TRUNCATION,
        truncate_vector,
        Truncation,
        Simplification,
        MAX_BOND_DIMENSION,
        _contract_last_and_first,
        _contract_nrjl_ijk_klm,
    )
else:
    from .newcore import (
        Strategy,
        DEFAULT_TOLERANCE,
        DEFAULT_STRATEGY,
        NO_TRUNCATION,
        truncate_vector,
        _contract_last_and_first,
        Truncation,
        Simplification,
        MAX_BOND_DIMENSION,
        _contract_last_and_first,
        _contract_nrjl_ijk_klm,
    )
