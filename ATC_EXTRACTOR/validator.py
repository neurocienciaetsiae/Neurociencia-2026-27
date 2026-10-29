from pathlib import Path
from collections import defaultdict
import ast
import math

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("raw_csv")
OUTPUT_DIR = Path("output")

SUMMARY_REPORT = Path("rt_validation_summary.csv")
DIFFERENCES_REPORT = Path("rt_validation_differences.csv")

NUMERIC_TOLERANCE = 1e-9


TRIAL_RT_COLUMNS = {
    ("manual", "change"): (
        "manual.change_manual.timesOn"
    ),
    ("manual", "maintain"): (
        "manual.maintain_manual.timesOn"
    ),
    ("high_1", "change"): (
        "high_1.change_high.timesOn"
    ),
    ("high_1", "maintain"): (
        "high_1.maintain_high.timesOn"
    ),
    ("high_2", "change"): (
        "high_2.change_high.timesOn"
    ),
    ("high_2", "maintain"): (
        "high_2.maintain_high.timesOn"
    ),
    ("low", "change"): (
        "low.change_low.timesOn"
    ),
    ("low", "maintain"): (
        "low.maintain_low.timesOn"
    ),
    ("high_3", "change"): (
        "high_3.change_high.timesOn"
    ),
    ("high_3", "maintain"): (
        "high_3.maintain_high.timesOn"
    ),
}


QUESTIONNAIRE_CONFIGURATIONS = [
    {
        "response_column": "answ_personality",
        "questionnaire": "personality",
        "item_columns": [
            "personality.thisIndex",
        ],
        "start_column": (
            "test_personality.started"
        ),
        "stop_column": (
            "test_personality.stopped"
        ),
    },
    {
        "response_column": "answ_willingness",
        "questionnaire": "willingness",
        "item_columns": [
            "willingness.thisIndex",
        ],
        "start_column": (
            "test_willingness.started"
        ),
        "stop_column": (
            "test_willingness.stopped"
        ),
    },
    {
        "response_column": (
            "answ_SATI_short_high"
        ),
        "questionnaire": "SATI_HIGH",
        "item_columns": [
            "SATI_high_1.thisIndex",
            "SATI_high_2.thisIndex",
            "SATI_alta_3.thisIndex",
        ],
        "start_column": (
            "SATI_short_high.started"
        ),
        "stop_column": (
            "SATI_short_high.stopped"
        ),
    },
    {
        "response_column": (
            "answ_SATI_short_low"
        ),
        "questionnaire": "SATI_LOW",
        "item_columns": [
            "SATI_low.thisIndex",
        ],
        "start_column": (
            "SATI_short_low.started"
        ),
        "stop_column": (
            "SATI_short_low.stopped"
        ),
    },
    {
        "response_column": "answ_test_final",
        "questionnaire": "final_trust",
        "item_columns": [
            "final.thisIndex",
        ],
        "start_column": (
            "test_final.started"
        ),
        "stop_column": (
            "test_final.stopped"
        ),
    },
]


DIFFERENCE_COLUMNS = [
    "participant",
    "session",
    "phase",
    "source_csv",
    "output_xlsx",
    "sheet",
    "row",
    "rt_type",
    "block",
    "trial_or_item",
    "response",
    "source_column",
    "source_start",
    "source_stop",
    "expected_rt",
    "actual_rt",
    "difference",
    "issue",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def is_missing(value):
    """Return True for missing or empty values."""

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    return (
        isinstance(value, str)
        and value.strip() == ""
    )


def to_number(value):
    """Convert one value to float."""

    if is_missing(value):
        return None

    number = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(number):
        return None

    return float(number)


def normalize_text(value):
    """Normalize one text value."""

    if is_missing(value):
        return None

    return str(value).strip().lower()


def extract_times_on(value):
    """
    Extract the first timesOn value.

    Example:
        "[1.5326]" -> 1.5326
    """

    if is_missing(value):
        return None

    try:
        parsed_value = ast.literal_eval(
            str(value)
        )
    except (ValueError, SyntaxError, TypeError):
        return None

    if not isinstance(parsed_value, list):
        return None

    if not parsed_value:
        return None

    return to_number(parsed_value[0])


def calculate_duration(
    row,
    start_column,
    stop_column,
):
    """Calculate stopped minus started."""

    start = to_number(row.get(start_column))
    stop = to_number(row.get(stop_column))

    if start is None or stop is None:
        return None, start, stop

    rt = stop - start

    if rt < 0:
        return None, start, stop

    return rt, start, stop


def rt_values_equal(actual, expected):
    """Compare two RT values."""

    if is_missing(actual) and expected is None:
        return True

    if is_missing(actual) or expected is None:
        return False

    actual_number = to_number(actual)

    if actual_number is None:
        return False

    return math.isclose(
        actual_number,
        expected,
        rel_tol=NUMERIC_TOLERANCE,
        abs_tol=NUMERIC_TOLERANCE,
    )


def numeric_difference(actual, expected):
    """Return actual minus expected."""

    actual_number = to_number(actual)

    if actual_number is None:
        return None

    if expected is None:
        return None

    return actual_number - expected


def normalize_participant_id(value):
    """Return a three-digit participant ID."""

    digits = "".join(
        character
        for character in str(value)
        if character.isdigit()
    )

    if not digits:
        return None

    return digits.zfill(3)


def get_session_information(file_position):
    """Assign session and phase from file order."""

    if file_position == 0:
        return "01", "NEUTRAL"

    return "02", "ECOLOGICAL"


def make_one_based_value(value):
    """
    Convert an original zero-based item to one-based.

    PsychoPy thisIndex values are zero-based.
    """

    number = to_number(value)

    if number is None:
        return None

    if number.is_integer():
        return int(number) + 1

    return number + 1


def make_one_based_series(series):
    """
    Convert a sequence to one-based only if it starts at zero.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        return numeric

    if valid.min() == 0:
        numeric = numeric + 1

    return numeric


def append_difference(
    differences,
    *,
    participant,
    session,
    phase,
    csv_file,
    excel_file,
    sheet,
    row,
    rt_type,
    block,
    trial_or_item,
    response,
    source_column,
    source_start,
    source_stop,
    expected_rt,
    actual_rt,
    issue,
):
    """Add one discrepancy to the report."""

    differences.append(
        {
            "participant": participant,
            "session": session,
            "phase": phase,
            "source_csv": csv_file.name,
            "output_xlsx": excel_file.name,
            "sheet": sheet,
            "row": row,
            "rt_type": rt_type,
            "block": block,
            "trial_or_item": trial_or_item,
            "response": response,
            "source_column": source_column,
            "source_start": source_start,
            "source_stop": source_stop,
            "expected_rt": expected_rt,
            "actual_rt": actual_rt,
            "difference": numeric_difference(
                actual_rt,
                expected_rt,
            ),
            "issue": issue,
        }
    )


# ============================================================
# FILE DISCOVERY
# ============================================================

def group_csv_files():
    """Group CSV files by participant."""

    grouped = defaultdict(list)

    for csv_file in RAW_DIR.glob("*.csv"):
        participant_prefix = (
            csv_file.name.split("_")[0]
        )

        grouped[participant_prefix].append(
            csv_file
        )

    for participant_prefix in grouped:
        grouped[participant_prefix] = sorted(
            grouped[participant_prefix]
        )

    return grouped


def get_excel_file(
    participant,
    session,
    phase,
):
    """Return the expected Excel path."""

    return (
        OUTPUT_DIR
        / f"P{participant}_"
          f"S{session}_"
          f"{phase}.xlsx"
    )


# ============================================================
# TRIAL RT VALIDATION
# ============================================================

def get_trial_condition(row):
    """
    Identify which trial routine produced the row.
    """

    condition_columns = [
        (
            "manual",
            [
                "manual.change_manual.timesOn",
                "manual.maintain_manual.timesOn",
            ],
        ),
        (
            "high_1",
            [
                "high_1.change_high.timesOn",
                "high_1.maintain_high.timesOn",
            ],
        ),
        (
            "high_2",
            [
                "high_2.change_high.timesOn",
                "high_2.maintain_high.timesOn",
            ],
        ),
        (
            "low",
            [
                "low.change_low.timesOn",
                "low.maintain_low.timesOn",
            ],
        ),
        (
            "high_3",
            [
                "high_3.change_high.timesOn",
                "high_3.maintain_high.timesOn",
            ],
        ),
    ]

    for condition, columns in condition_columns:
        for column in columns:
            if extract_times_on(
                row.get(column)
            ) is not None:
                return condition

    block = to_number(row.get("block"))

    block_mapping = {
        1: "manual",
        2: "high_1",
        3: "high_2",
        4: "low",
        5: "high_3",
    }

    if block is None:
        return None

    return block_mapping.get(int(block))


def get_expected_trial_rt(row):
    """
    Get trial RT from the timesOn column that corresponds
    to the recorded answer.
    """

    answer = normalize_text(row.get("answer"))

    if answer == "no_response":
        return None, None

    if answer not in {"change", "maintain"}:
        return None, None

    condition = get_trial_condition(row)

    if condition is None:
        return None, None

    source_column = TRIAL_RT_COLUMNS.get(
        (condition, answer)
    )

    if source_column is None:
        return None, None

    expected_rt = extract_times_on(
        row.get(source_column)
    )

    return expected_rt, source_column


def validate_trial_rts(
    source,
    actual_trials,
    participant,
    session,
    phase,
    csv_file,
    excel_file,
):
    """Validate trial_rt and trust_rt."""

    differences = []
    checks = 0

    if "block" not in source.columns:
        return checks, differences

    source_trials = source.loc[
        source["block"].notna()
    ].copy()

    source_trials["_block"] = (
        make_one_based_series(
            source_trials["block"]
        )
    )

    source_trials["_trial"] = (
        source_trials.groupby(
            "_block",
            dropna=False,
        )["block_trial"]
        .transform(make_one_based_series)
    )

    if len(source_trials) != len(actual_trials):
        append_difference(
            differences,
            participant=participant,
            session=session,
            phase=phase,
            csv_file=csv_file,
            excel_file=excel_file,
            sheet="TRIALS",
            row=None,
            rt_type="row_count",
            block=None,
            trial_or_item=None,
            response=None,
            source_column=None,
            source_start=len(source_trials),
            source_stop=None,
            expected_rt=len(source_trials),
            actual_rt=len(actual_trials),
            issue="ROW_COUNT_MISMATCH",
        )

    rows_to_compare = min(
        len(source_trials),
        len(actual_trials),
    )

    for position in range(rows_to_compare):
        source_row = source_trials.iloc[position]
        actual_row = actual_trials.iloc[position]
        excel_row = position + 2

        block = source_row.get("_block")
        trial = source_row.get("_trial")
        answer = source_row.get("answer")

        expected_trial_rt, source_column = (
            get_expected_trial_rt(source_row)
        )

        actual_trial_rt = actual_row.get(
            "trial_rt"
        )

        checks += 1

        if not rt_values_equal(
            actual_trial_rt,
            expected_trial_rt,
        ):
            append_difference(
                differences,
                participant=participant,
                session=session,
                phase=phase,
                csv_file=csv_file,
                excel_file=excel_file,
                sheet="TRIALS",
                row=excel_row,
                rt_type="trial_rt",
                block=block,
                trial_or_item=trial,
                response=answer,
                source_column=source_column,
                source_start=None,
                source_stop=None,
                expected_rt=expected_trial_rt,
                actual_rt=actual_trial_rt,
                issue="RT_MISMATCH",
            )

        high_rt, high_start, high_stop = (
            calculate_duration(
                source_row,
                "question_high.started",
                "question_high.stopped",
            )
        )

        low_rt, low_start, low_stop = (
            calculate_duration(
                source_row,
                "question_low.started",
                "question_low.stopped",
            )
        )

        if high_rt is not None:
            expected_trust_rt = high_rt
            trust_source = (
                "question_high.stopped - "
                "question_high.started"
            )
            trust_start = high_start
            trust_stop = high_stop
        else:
            expected_trust_rt = low_rt
            trust_source = (
                "question_low.stopped - "
                "question_low.started"
            )
            trust_start = low_start
            trust_stop = low_stop

        actual_trust_rt = actual_row.get(
            "trust_rt"
        )

        checks += 1

        if not rt_values_equal(
            actual_trust_rt,
            expected_trust_rt,
        ):
            append_difference(
                differences,
                participant=participant,
                session=session,
                phase=phase,
                csv_file=csv_file,
                excel_file=excel_file,
                sheet="TRIALS",
                row=excel_row,
                rt_type="trust_rt",
                block=block,
                trial_or_item=trial,
                response=source_row.get("trust"),
                source_column=trust_source,
                source_start=trust_start,
                source_stop=trust_stop,
                expected_rt=expected_trust_rt,
                actual_rt=actual_trust_rt,
                issue="RT_MISMATCH",
            )

    return checks, differences


# ============================================================
# QUESTIONNAIRE RT VALIDATION
# ============================================================

def get_sati_block(row):
    """Reconstruct the SATI block."""

    mapping = [
        ("SATI_high_1.thisIndex", 2),
        ("SATI_high_2.thisIndex", 3),
        ("SATI_low.thisIndex", 4),
        ("SATI_alta_3.thisIndex", 5),
    ]

    for column, block in mapping:
        if not is_missing(row.get(column)):
            return block

    return None


def get_questionnaire_item(row, columns):
    """Return a one-based questionnaire item."""

    for column in columns:
        value = row.get(column)

        if not is_missing(value):
            return make_one_based_value(value)

    return None


def build_expected_questionnaire_rts(source):
    """Build expected questionnaire RT records."""

    records = []

    for configuration in (
        QUESTIONNAIRE_CONFIGURATIONS
    ):
        response_column = configuration[
            "response_column"
        ]

        if response_column not in source.columns:
            continue

        response_rows = source.loc[
            source[response_column].notna()
        ]

        questionnaire = configuration[
            "questionnaire"
        ]

        for _, row in response_rows.iterrows():
            expected_rt, start, stop = (
                calculate_duration(
                    row,
                    configuration["start_column"],
                    configuration["stop_column"],
                )
            )

            block = None

            if questionnaire in {
                "SATI_HIGH",
                "SATI_LOW",
            }:
                block = get_sati_block(row)

            records.append(
                {
                    "questionnaire": questionnaire,
                    "block": block,
                    "item": get_questionnaire_item(
                        row,
                        configuration["item_columns"],
                    ),
                    "response": row.get(
                        response_column
                    ),
                    "expected_rt": expected_rt,
                    "source_start": start,
                    "source_stop": stop,
                    "source_column": (
                        f"{configuration['stop_column']} - "
                        f"{configuration['start_column']}"
                    ),
                }
            )

    order = {
        "personality": 1,
        "willingness": 2,
        "SATI_HIGH": 3,
        "SATI_LOW": 3,
        "final_trust": 4,
    }

    records.sort(
        key=lambda record: (
            order.get(
                record["questionnaire"],
                99,
            ),
            record["block"] or 0,
            record["item"] or 0,
        )
    )

    return records


def validate_questionnaire_rts(
    source,
    actual_questionnaires,
    participant,
    session,
    phase,
    csv_file,
    excel_file,
):
    """Validate all questionnaire RT values."""

    differences = []
    checks = 0

    expected_records = (
        build_expected_questionnaire_rts(
            source
        )
    )

    if len(expected_records) != len(
        actual_questionnaires
    ):
        append_difference(
            differences,
            participant=participant,
            session=session,
            phase=phase,
            csv_file=csv_file,
            excel_file=excel_file,
            sheet="QUESTIONNAIRES",
            row=None,
            rt_type="row_count",
            block=None,
            trial_or_item=None,
            response=None,
            source_column=None,
            source_start=len(expected_records),
            source_stop=None,
            expected_rt=len(expected_records),
            actual_rt=len(actual_questionnaires),
            issue="ROW_COUNT_MISMATCH",
        )

    rows_to_compare = min(
        len(expected_records),
        len(actual_questionnaires),
    )

    for position in range(rows_to_compare):
        expected = expected_records[position]
        actual = actual_questionnaires.iloc[
            position
        ]

        actual_questionnaire = normalize_text(
            actual.get("questionnaire")
        )

        expected_questionnaire = normalize_text(
            expected["questionnaire"]
        )

        actual_block = to_number(
            actual.get("block")
        )

        expected_block = to_number(
            expected["block"]
        )

        actual_item = to_number(
            actual.get("item")
        )

        expected_item = to_number(
            expected["item"]
        )

        identity_matches = (
            actual_questionnaire
            == expected_questionnaire
            and (
                (
                    actual_block is None
                    and expected_block is None
                )
                or (
                    actual_block is not None
                    and expected_block is not None
                    and math.isclose(
                        actual_block,
                        expected_block,
                    )
                )
            )
            and (
                actual_item is not None
                and expected_item is not None
                and math.isclose(
                    actual_item,
                    expected_item,
                )
            )
        )

        excel_row = position + 2

        if not identity_matches:
            append_difference(
                differences,
                participant=participant,
                session=session,
                phase=phase,
                csv_file=csv_file,
                excel_file=excel_file,
                sheet="QUESTIONNAIRES",
                row=excel_row,
                rt_type="questionnaire_identity",
                block=expected["block"],
                trial_or_item=expected["item"],
                response=expected["response"],
                source_column=None,
                source_start=None,
                source_stop=None,
                expected_rt=(
                    f"{expected['questionnaire']} | "
                    f"{expected['block']} | "
                    f"{expected['item']}"
                ),
                actual_rt=(
                    f"{actual.get('questionnaire')} | "
                    f"{actual.get('block')} | "
                    f"{actual.get('item')}"
                ),
                issue="ROW_IDENTITY_MISMATCH",
            )

        actual_rt = actual.get("rt")
        expected_rt = expected["expected_rt"]

        checks += 1

        if not rt_values_equal(
            actual_rt,
            expected_rt,
        ):
            append_difference(
                differences,
                participant=participant,
                session=session,
                phase=phase,
                csv_file=csv_file,
                excel_file=excel_file,
                sheet="QUESTIONNAIRES",
                row=excel_row,
                rt_type=(
                    f"{expected['questionnaire']}_rt"
                ),
                block=expected["block"],
                trial_or_item=expected["item"],
                response=expected["response"],
                source_column=expected[
                    "source_column"
                ],
                source_start=expected[
                    "source_start"
                ],
                source_stop=expected[
                    "source_stop"
                ],
                expected_rt=expected_rt,
                actual_rt=actual_rt,
                issue="RT_MISMATCH",
            )

    return checks, differences


# ============================================================
# EMOTION RT VALIDATION
# ============================================================

def get_emotion_condition(block):
    """Return high or low for one emotion block."""

    if block == 4:
        return "low"

    return "high"


def build_expected_emotion_rts(source):
    """Build the expected emotion RT records."""

    records = []

    if "answ_valence" not in source.columns:
        return records

    emotion_number = 0

    for position in range(len(source)):
        valence_row = source.iloc[position]
        valence = normalize_text(
            valence_row.get("answ_valence")
        )

        if valence is None:
            continue

        block = emotion_number + 2
        condition = get_emotion_condition(
            block
        )

        activation = None
        activation_row = None

        if valence == "pleasant":
            activation_position = position + 1

            if activation_position < len(source):
                possible_row = source.iloc[
                    activation_position
                ]

                possible_activation = (
                    normalize_text(
                        possible_row.get(
                            "answ_pleasant"
                        )
                    )
                )

                if possible_activation is not None:
                    activation = possible_activation
                    activation_row = possible_row

        elif valence == "unpleasant":
            activation_position = position + 2

            if activation_position < len(source):
                possible_row = source.iloc[
                    activation_position
                ]

                possible_activation = (
                    normalize_text(
                        possible_row.get(
                            "answ_unpleasant"
                        )
                    )
                )

                if possible_activation is not None:
                    activation = possible_activation
                    activation_row = possible_row

        valence_column = (
            f"{valence}_{condition}.timesOn"
        )

        valence_rt = extract_times_on(
            valence_row.get(valence_column)
        )

        activation_column = None
        activation_rt = None

        if (
            activation_row is not None
            and activation is not None
        ):
            activation_column = (
                f"{activation}_"
                f"{condition}.timesOn"
            )

            activation_rt = extract_times_on(
                activation_row.get(
                    activation_column
                )
            )

        records.append(
            {
                "block": block,
                "valence": valence,
                "activation": activation,
                "valence_rt": valence_rt,
                "activation_rt": activation_rt,
                "valence_column": valence_column,
                "activation_column": (
                    activation_column
                ),
            }
        )

        emotion_number += 1

    return records


def validate_emotion_rts(
    source,
    actual_emotions,
    participant,
    session,
    phase,
    csv_file,
    excel_file,
):
    """Validate valence_rt and activation_rt."""

    differences = []
    checks = 0

    expected_records = (
        build_expected_emotion_rts(source)
    )

    if len(expected_records) != len(
        actual_emotions
    ):
        append_difference(
            differences,
            participant=participant,
            session=session,
            phase=phase,
            csv_file=csv_file,
            excel_file=excel_file,
            sheet="EMOTIONS",
            row=None,
            rt_type="row_count",
            block=None,
            trial_or_item=None,
            response=None,
            source_column=None,
            source_start=len(expected_records),
            source_stop=None,
            expected_rt=len(expected_records),
            actual_rt=len(actual_emotions),
            issue="ROW_COUNT_MISMATCH",
        )

    rows_to_compare = min(
        len(expected_records),
        len(actual_emotions),
    )

    for position in range(rows_to_compare):
        expected = expected_records[position]
        actual = actual_emotions.iloc[position]
        excel_row = position + 2

        actual_valence_rt = actual.get(
            "valence_rt"
        )

        checks += 1

        if not rt_values_equal(
            actual_valence_rt,
            expected["valence_rt"],
        ):
            append_difference(
                differences,
                participant=participant,
                session=session,
                phase=phase,
                csv_file=csv_file,
                excel_file=excel_file,
                sheet="EMOTIONS",
                row=excel_row,
                rt_type="valence_rt",
                block=expected["block"],
                trial_or_item=None,
                response=expected["valence"],
                source_column=expected[
                    "valence_column"
                ],
                source_start=None,
                source_stop=None,
                expected_rt=expected[
                    "valence_rt"
                ],
                actual_rt=actual_valence_rt,
                issue="RT_MISMATCH",
            )

        actual_activation_rt = actual.get(
            "activation_rt"
        )

        checks += 1

        if not rt_values_equal(
            actual_activation_rt,
            expected["activation_rt"],
        ):
            append_difference(
                differences,
                participant=participant,
                session=session,
                phase=phase,
                csv_file=csv_file,
                excel_file=excel_file,
                sheet="EMOTIONS",
                row=excel_row,
                rt_type="activation_rt",
                block=expected["block"],
                trial_or_item=None,
                response=expected["activation"],
                source_column=expected[
                    "activation_column"
                ],
                source_start=None,
                source_stop=None,
                expected_rt=expected[
                    "activation_rt"
                ],
                actual_rt=actual_activation_rt,
                issue="RT_MISMATCH",
            )

    return checks, differences


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    """Validate all exported RT values."""

    if not RAW_DIR.exists():
        print(
            f"Missing directory: "
            f"{RAW_DIR.resolve()}"
        )
        return

    if not OUTPUT_DIR.exists():
        print(
            f"Missing directory: "
            f"{OUTPUT_DIR.resolve()}"
        )
        return

    grouped_files = group_csv_files()

    if not grouped_files:
        print(
            f"No CSV files found in: "
            f"{RAW_DIR.resolve()}"
        )
        return

    all_differences = []
    summary_rows = []

    files_checked = 0
    total_rt_checks = 0

    for raw_participant in sorted(
        grouped_files
    ):
        participant = normalize_participant_id(
            raw_participant
        )

        if participant is None:
            print(
                f"Invalid participant prefix: "
                f"{raw_participant}"
            )
            continue

        participant_files = grouped_files[
            raw_participant
        ]

        for file_position, csv_file in enumerate(
            participant_files
        ):
            session, phase = (
                get_session_information(
                    file_position
                )
            )

            excel_file = get_excel_file(
                participant,
                session,
                phase,
            )

            print(
                f"Validating participant "
                f"{participant}, "
                f"session {session}: "
                f"{excel_file.name}"
            )

            if not excel_file.exists():
                print(
                    f"  ERROR: missing Excel file"
                )

                summary_rows.append(
                    {
                        "participant": participant,
                        "session": session,
                        "phase": phase,
                        "source_csv": csv_file.name,
                        "output_xlsx": (
                            excel_file.name
                        ),
                        "rt_checks": 0,
                        "differences": 1,
                        "status": "FAILED",
                    }
                )
                continue

            try:
                source = pd.read_csv(csv_file)

                trials = pd.read_excel(
                    excel_file,
                    sheet_name="TRIALS",
                    engine="openpyxl",
                    dtype=object,
                )

                questionnaires = pd.read_excel(
                    excel_file,
                    sheet_name="QUESTIONNAIRES",
                    engine="openpyxl",
                    dtype=object,
                )

                emotions = pd.read_excel(
                    excel_file,
                    sheet_name="EMOTIONS",
                    engine="openpyxl",
                    dtype=object,
                )

            except Exception as error:
                print(
                    f"  ERROR: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                summary_rows.append(
                    {
                        "participant": participant,
                        "session": session,
                        "phase": phase,
                        "source_csv": csv_file.name,
                        "output_xlsx": (
                            excel_file.name
                        ),
                        "rt_checks": 0,
                        "differences": 1,
                        "status": "FAILED",
                    }
                )
                continue

            file_differences = []
            file_checks = 0

            checks, differences = (
                validate_trial_rts(
                    source=source,
                    actual_trials=trials,
                    participant=participant,
                    session=session,
                    phase=phase,
                    csv_file=csv_file,
                    excel_file=excel_file,
                )
            )

            file_checks += checks
            file_differences.extend(
                differences
            )

            checks, differences = (
                validate_questionnaire_rts(
                    source=source,
                    actual_questionnaires=(
                        questionnaires
                    ),
                    participant=participant,
                    session=session,
                    phase=phase,
                    csv_file=csv_file,
                    excel_file=excel_file,
                )
            )

            file_checks += checks
            file_differences.extend(
                differences
            )

            checks, differences = (
                validate_emotion_rts(
                    source=source,
                    actual_emotions=emotions,
                    participant=participant,
                    session=session,
                    phase=phase,
                    csv_file=csv_file,
                    excel_file=excel_file,
                )
            )

            file_checks += checks
            file_differences.extend(
                differences
            )

            all_differences.extend(
                file_differences
            )

            total_rt_checks += file_checks
            files_checked += 1

            status = (
                "PASSED"
                if not file_differences
                else "FAILED"
            )

            summary_rows.append(
                {
                    "participant": participant,
                    "session": session,
                    "phase": phase,
                    "source_csv": csv_file.name,
                    "output_xlsx": (
                        excel_file.name
                    ),
                    "rt_checks": file_checks,
                    "differences": len(
                        file_differences
                    ),
                    "status": status,
                }
            )

            print(
                f"  RT checks: {file_checks}"
            )

            print(
                f"  Differences: "
                f"{len(file_differences)}"
            )

            print(
                f"  Status: {status}"
            )

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "participant",
            "session",
            "phase",
            "source_csv",
            "output_xlsx",
            "rt_checks",
            "differences",
            "status",
        ],
    )

    differences = pd.DataFrame(
        all_differences,
        columns=DIFFERENCE_COLUMNS,
    )

    summary.to_csv(
        SUMMARY_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    differences.to_csv(
        DIFFERENCES_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    passed_files = 0
    failed_files = 0

    if not summary.empty:
        passed_files = int(
            (
                summary["status"] == "PASSED"
            ).sum()
        )

        failed_files = int(
            (
                summary["status"] == "FAILED"
            ).sum()
        )

    print()
    print("RT VALIDATION COMPLETE")
    print(
        f"Files checked: "
        f"{files_checked}"
    )
    print(
        f"RT values checked: "
        f"{total_rt_checks}"
    )
    print(
        f"Passed files: "
        f"{passed_files}"
    )
    print(
        f"Failed files: "
        f"{failed_files}"
    )
    print(
        f"Differences found: "
        f"{len(all_differences)}"
    )
    print(
        f"Summary report: "
        f"{SUMMARY_REPORT}"
    )
    print(
        f"Differences report: "
        f"{DIFFERENCES_REPORT}"
    )

    if all_differences:
        print("RT VALIDATION: FAILED")
    else:
        print("RT VALIDATION: PASSED")


if __name__ == "__main__":
    main()