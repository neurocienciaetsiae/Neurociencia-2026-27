from pathlib import Path
from collections import defaultdict
import ast

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("raw_csv")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRIAL_COLUMNS = [
    "participant",
    "session",
    "phase",
    "block",
    "trial",
    "a1",
    "a2",
    "conflict",
    "automation_error",
    "answer",
    "correct",
    "trust",
    "trial_rt",
    "trust_rt",
]


QUESTIONNAIRE_COLUMNS = [
    "participant",
    "session",
    "phase",
    "questionnaire",
    "block",
    "item",
    "question",
    "clarification",
    "response",
    "rt",
]


EMOTION_COLUMNS = [
    "participant",
    "session",
    "phase",
    "block",
    "valence",
    "activation",
    "valence_rt",
    "activation_rt",
]


TRIAL_RT_COLUMNS = [
    "manual.change_manual.timesOn",
    "manual.maintain_manual.timesOn",
    "high_1.change_high.timesOn",
    "high_1.maintain_high.timesOn",
    "high_2.change_high.timesOn",
    "high_2.maintain_high.timesOn",
    "low.change_low.timesOn",
    "low.maintain_low.timesOn",
    "high_3.change_high.timesOn",
    "high_3.maintain_high.timesOn",
]


QUESTIONNAIRE_CONFIGURATIONS = [
    {
        "response_column": "answ_personality",
        "name": "personality",
        "item_columns": [
            "personality.thisIndex",
        ],
        "start_column": "test_personality.started",
        "stop_column": "test_personality.stopped",
    },
    {
        "response_column": "answ_willingness",
        "name": "willingness",
        "item_columns": [
            "willingness.thisIndex",
        ],
        "start_column": "test_willingness.started",
        "stop_column": "test_willingness.stopped",
    },
    {
        "response_column": "answ_SATI_short_high",
        "name": "SATI_HIGH",
        "item_columns": [
            "SATI_high_1.thisIndex",
            "SATI_high_2.thisIndex",
            "SATI_alta_3.thisIndex",
        ],
        "start_column": "SATI_short_high.started",
        "stop_column": "SATI_short_high.stopped",
    },
    {
        "response_column": "answ_SATI_short_low",
        "name": "SATI_LOW",
        "item_columns": [
            "SATI_low.thisIndex",
        ],
        "start_column": "SATI_short_low.started",
        "stop_column": "SATI_short_low.stopped",
    },
    {
        "response_column": "answ_test_final",
        "name": "final_trust",
        "item_columns": [
            "final.thisIndex",
        ],
        "start_column": "test_final.started",
        "stop_column": "test_final.stopped",
    },
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def is_missing(value):
    """Return True for None, NaN and empty strings."""

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
    """Convert one value to float without raising an error."""

    if is_missing(value):
        return None

    number = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(number):
        return None

    return float(number)


def extract_first_list_value(value):
    """
    Extract the first numeric value from a PsychoPy list.

    Example:
        "[1.5326]" -> 1.5326
    """

    if is_missing(value):
        return None

    try:
        parsed_value = ast.literal_eval(str(value))
    except (ValueError, SyntaxError, TypeError):
        return None

    if not isinstance(parsed_value, list):
        return None

    if not parsed_value:
        return None

    return to_number(parsed_value[0])


def calculate_rt(row, start_column, stop_column):
    """Calculate RT as stop time minus start time."""

    start_time = to_number(
        row.get(start_column)
    )

    stop_time = to_number(
        row.get(stop_column)
    )

    if start_time is None or stop_time is None:
        return None

    response_time = stop_time - start_time

    if response_time < 0:
        return None

    return float(response_time)


def get_optional_column(dataframe, column_name):
    """Return a column or an empty aligned Series."""

    if column_name in dataframe.columns:
        return dataframe[column_name]

    return pd.Series(
        index=dataframe.index,
        dtype="object",
    )


def get_item_number(row, possible_columns):
    """Return the first available questionnaire item index."""

    for column_name in possible_columns:
        value = row.get(column_name)

        if is_missing(value):
            continue

        number = to_number(value)

        if number is None:
            return value

        if number.is_integer():
            return int(number)

        return number

    return None


def make_one_based(series):
    """
    Convert a zero-based numeric sequence to a one-based sequence.

    Examples:
        0, 1, 2 -> 1, 2, 3
        1, 2, 3 -> 1, 2, 3

    The sequence is changed only when its minimum value is zero.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_values = numeric_series.dropna()

    if valid_values.empty:
        return series

    if valid_values.min() == 0:
        numeric_series = numeric_series + 1

    return numeric_series.astype("Int64")


# ============================================================
# FILE AND PARTICIPANT HELPERS
# ============================================================

def normalize_participant_id(raw_participant):
    """Convert a filename prefix to a three-digit ID."""

    digits = "".join(
        character
        for character in str(raw_participant)
        if character.isdigit()
    )

    if not digits:
        return None

    return digits.zfill(3)


def get_session_information(file_position):
    """
    Assign session using chronological file order.

    First CSV:
        session 01
        NEUTRAL

    Second CSV:
        session 02
        ECOLOGICAL
    """

    if file_position == 0:
        return "01", "NEUTRAL"

    return "02", "ECOLOGICAL"


def group_files_by_participant():
    """Group and sort CSV files by participant."""

    grouped_files = defaultdict(list)

    for csv_file in RAW_DIR.glob("*.csv"):
        participant_prefix = (
            csv_file.name.split("_")[0]
        )

        grouped_files[
            participant_prefix
        ].append(csv_file)

    for participant_prefix in grouped_files:
        grouped_files[
            participant_prefix
        ] = sorted(
            grouped_files[
                participant_prefix
            ]
        )

    return grouped_files


# ============================================================
# TRIAL EXTRACTION
# ============================================================

def get_trial_rt(row):
    """
    Extract trial RT from the selected ButtonStim timesOn.

    The first available timesOn value is used.
    """

    for column_name in TRIAL_RT_COLUMNS:
        response_time = extract_first_list_value(
            row.get(column_name)
        )

        if response_time is not None:
            return response_time

    return None


def calculate_trust_rt(trials):
    """
    Calculate trust RT from the trust routine timestamps.

    Trust uses a slider, so RT is calculated as:
        stopped - started
    """

    trust_rt = pd.Series(
        index=trials.index,
        dtype="float64",
    )

    if (
        "question_high.started" in trials.columns
        and "question_high.stopped" in trials.columns
    ):
        high_start = pd.to_numeric(
            trials["question_high.started"],
            errors="coerce",
        )

        high_stop = pd.to_numeric(
            trials["question_high.stopped"],
            errors="coerce",
        )

        high_rt = high_stop - high_start
        high_rt = high_rt.where(high_rt >= 0)

        trust_rt = trust_rt.fillna(high_rt)

    if (
        "question_low.started" in trials.columns
        and "question_low.stopped" in trials.columns
    ):
        low_start = pd.to_numeric(
            trials["question_low.started"],
            errors="coerce",
        )

        low_stop = pd.to_numeric(
            trials["question_low.stopped"],
            errors="coerce",
        )

        low_rt = low_stop - low_start
        low_rt = low_rt.where(low_rt >= 0)

        trust_rt = trust_rt.fillna(low_rt)

    return trust_rt


def extract_trials(
    dataframe,
    participant,
    session,
    phase,
):
    """Create the final TRIALS table."""

    if "block" not in dataframe.columns:
        return pd.DataFrame(
            columns=TRIAL_COLUMNS
        )

    trials = dataframe.loc[
        dataframe["block"].notna()
    ].copy()

    high_trust = get_optional_column(
        trials,
        "answ_high_question",
    )

    low_trust = get_optional_column(
        trials,
        "answ_low_question",
    )

    trials["trust"] = high_trust.combine_first(
        low_trust
    )

    trials["trial_rt"] = trials.apply(
        get_trial_rt,
        axis=1,
    )

    trials["trust_rt"] = calculate_trust_rt(
        trials
    )

    output = pd.DataFrame(index=trials.index)

    output["participant"] = participant
    output["session"] = session
    output["phase"] = phase

    column_mapping = {
        "block": "block",
        "block_trial": "trial",
        "a1": "a1",
        "a2": "a2",
        "conflict": "conflict",
        "automation_error": "automation_error",
        "answer": "answer",
        "correct": "correct",
    }

    for source_column, output_column in (
        column_mapping.items()
    ):
        output[output_column] = get_optional_column(
            trials,
            source_column,
        )

    output["trust"] = trials["trust"]
    output["trial_rt"] = trials["trial_rt"]
    output["trust_rt"] = trials["trust_rt"]

    # Ensure block numbering starts at 1.
    output["block"] = make_one_based(
        output["block"]
    )

    # Ensure trial numbering starts at 1 inside each block.
    output["trial"] = (
        output.groupby(
            "block",
            dropna=False,
        )["trial"]
        .transform(make_one_based)
    )

    return (
        output[TRIAL_COLUMNS]
        .reset_index(drop=True)
    )


# ============================================================
# QUESTIONNAIRE EXTRACTION
# ============================================================

def get_sati_block(row):
    """
    Reconstruct the SATI block.

    SATI high 1 -> block 2
    SATI high 2 -> block 3
    SATI low    -> block 4
    SATI high 3 -> block 5
    """

    mapping = [
        ("SATI_high_1.thisIndex", 2),
        ("SATI_high_2.thisIndex", 3),
        ("SATI_low.thisIndex", 4),
        ("SATI_alta_3.thisIndex", 5),
    ]

    for column_name, block in mapping:
        if not is_missing(row.get(column_name)):
            return block

    return None


def extract_one_questionnaire(
    dataframe,
    participant,
    session,
    phase,
    configuration,
):
    """Extract one questionnaire type."""

    response_column = configuration[
        "response_column"
    ]

    if response_column not in dataframe.columns:
        return []

    source_rows = dataframe.loc[
        dataframe[response_column].notna()
    ]

    questionnaire_name = configuration["name"]
    results = []

    for _, row in source_rows.iterrows():
        block = None

        if questionnaire_name in {
            "SATI_HIGH",
            "SATI_LOW",
        }:
            block = get_sati_block(row)

        results.append(
            {
                "participant": participant,
                "session": session,
                "phase": phase,
                "questionnaire": questionnaire_name,
                "block": block,
                "item": get_item_number(
                    row,
                    configuration["item_columns"],
                ),
                "question": row.get("Question"),
                "clarification": row.get(
                    "Clarification"
                ),
                "response": row.get(
                    response_column
                ),
                "rt": calculate_rt(
                    row,
                    configuration["start_column"],
                    configuration["stop_column"],
                ),
            }
        )

    return results


def normalize_questionnaire_items(questionnaires):
    """
    Ensure item numbers start at 1 inside each questionnaire.

    SATI items are normalized separately for every SATI block.
    """

    if questionnaires.empty:
        return questionnaires

    output = questionnaires.copy()

    output["item"] = (
        output.groupby(
            [
                "questionnaire",
                "block",
            ],
            dropna=False,
        )["item"]
        .transform(make_one_based)
    )

    return output


def sort_questionnaires(questionnaires):
    """Sort questionnaires in their intended order."""

    if questionnaires.empty:
        return questionnaires

    questionnaire_order = {
        "personality": 1,
        "willingness": 2,
        "SATI_HIGH": 3,
        "SATI_LOW": 3,
        "final_trust": 4,
    }

    output = questionnaires.copy()

    output["_questionnaire_order"] = (
        output["questionnaire"]
        .map(questionnaire_order)
        .fillna(99)
    )

    output["_block_order"] = pd.to_numeric(
        output["block"],
        errors="coerce",
    ).fillna(0)

    output["_item_order"] = pd.to_numeric(
        output["item"],
        errors="coerce",
    ).fillna(0)

    output = (
        output.sort_values(
            by=[
                "_questionnaire_order",
                "_block_order",
                "_item_order",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_questionnaire_order",
                "_block_order",
                "_item_order",
            ]
        )
        .reset_index(drop=True)
    )

    return output


def extract_questionnaires(
    dataframe,
    participant,
    session,
    phase,
):
    """Create the final QUESTIONNAIRES table."""

    questionnaire_rows = []

    for configuration in (
        QUESTIONNAIRE_CONFIGURATIONS
    ):
        questionnaire_rows.extend(
            extract_one_questionnaire(
                dataframe=dataframe,
                participant=participant,
                session=session,
                phase=phase,
                configuration=configuration,
            )
        )

    questionnaires = pd.DataFrame(
        questionnaire_rows,
        columns=QUESTIONNAIRE_COLUMNS,
    )

    questionnaires = normalize_questionnaire_items(
        questionnaires
    )

    return sort_questionnaires(
        questionnaires
    )


# ============================================================
# EMOTION EXTRACTION
# ============================================================

def get_emotion_condition(block):
    """
    Return the PsychoPy condition suffix.

    Blocks 2, 3 and 5 use high.
    Block 4 uses low.
    """

    if block == 4:
        return "low"

    return "high"


def get_valence_rt(
    valence_row,
    valence,
    condition,
):
    """Extract valence RT from the selected ButtonStim timesOn."""

    if valence not in {
        "pleasant",
        "unpleasant",
    }:
        return None

    column_name = (
        f"{valence}_{condition}.timesOn"
    )

    return extract_first_list_value(
        valence_row.get(column_name)
    )


def get_activation_rt(
    activation_row,
    activation,
    condition,
):
    """Extract activation RT from the selected ButtonStim timesOn."""

    if activation not in {
        "relaxed",
        "excited",
        "stressed",
        "bored",
    }:
        return None

    column_name = (
        f"{activation}_{condition}.timesOn"
    )

    return extract_first_list_value(
        activation_row.get(column_name)
    )


def extract_emotions(
    dataframe,
    participant,
    session,
    phase,
):
    """
    Create the final EMOTIONS table.

    Valence RT:
        First timesOn value of pleasant or unpleasant.

    Activation RT:
        First timesOn value of relaxed, excited,
        stressed or bored.
    """

    if "answ_valence" not in dataframe.columns:
        return pd.DataFrame(
            columns=EMOTION_COLUMNS
        )

    emotion_rows = []
    emotion_number = 0

    for position in range(len(dataframe)):
        valence_row = dataframe.iloc[position]

        valence = valence_row.get(
            "answ_valence"
        )

        if is_missing(valence):
            continue

        # Emotion assessments correspond to blocks 2, 3, 4 and 5.
        block = emotion_number + 2
        condition = get_emotion_condition(block)

        activation = None
        activation_row = None

        if valence == "pleasant":
            activation_position = position + 1

            if activation_position < len(dataframe):
                possible_row = dataframe.iloc[
                    activation_position
                ]

                possible_activation = (
                    possible_row.get(
                        "answ_pleasant"
                    )
                )

                if not is_missing(
                    possible_activation
                ):
                    activation = possible_activation
                    activation_row = possible_row

        elif valence == "unpleasant":
            activation_position = position + 2

            if activation_position < len(dataframe):
                possible_row = dataframe.iloc[
                    activation_position
                ]

                possible_activation = (
                    possible_row.get(
                        "answ_unpleasant"
                    )
                )

                if not is_missing(
                    possible_activation
                ):
                    activation = possible_activation
                    activation_row = possible_row

        valence_rt = get_valence_rt(
            valence_row=valence_row,
            valence=valence,
            condition=condition,
        )

        activation_rt = None

        if activation_row is not None:
            activation_rt = get_activation_rt(
                activation_row=activation_row,
                activation=activation,
                condition=condition,
            )

        emotion_rows.append(
            {
                "participant": participant,
                "session": session,
                "phase": phase,
                "block": block,
                "valence": valence,
                "activation": activation,
                "valence_rt": valence_rt,
                "activation_rt": activation_rt,
            }
        )

        emotion_number += 1

    emotions = pd.DataFrame(
        emotion_rows,
        columns=EMOTION_COLUMNS,
    )

    if not emotions.empty:
        emotions["block"] = make_one_based(
            emotions["block"]
        )

    return emotions


# ============================================================
# EXCEL EXPORT
# ============================================================

def export_participant_file(
    participant,
    session,
    phase,
    trials,
    questionnaires,
    emotions,
):
    """Write the three final worksheets to one Excel file."""

    output_file = (
        OUTPUT_DIR
        / f"P{participant}_"
          f"S{session}_"
          f"{phase}.xlsx"
    )

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:

        trials.to_excel(
            writer,
            sheet_name="TRIALS",
            index=False,
        )

        questionnaires.to_excel(
            writer,
            sheet_name="QUESTIONNAIRES",
            index=False,
        )

        emotions.to_excel(
            writer,
            sheet_name="EMOTIONS",
            index=False,
        )

    return output_file


# ============================================================
# MAIN PROCESS
# ============================================================

def main():
    """Process every participant and session."""

    grouped_files = group_files_by_participant()

    if not grouped_files:
        print(
            f"No CSV files were found in: "
            f"{RAW_DIR.resolve()}"
        )
        return

    generated_files = 0
    errors = []

    for raw_participant in sorted(
        grouped_files
    ):
        participant = normalize_participant_id(
            raw_participant
        )

        participant_files = grouped_files[
            raw_participant
        ]

        if participant is None:
            errors.append(
                f"Invalid participant prefix: "
                f"{raw_participant}"
            )
            continue

        if len(participant_files) != 2:
            print(
                f"WARNING: participant "
                f"{participant} has "
                f"{len(participant_files)} "
                f"CSV files."
            )

        for file_position, csv_file in enumerate(
            participant_files
        ):
            session, phase = (
                get_session_information(
                    file_position
                )
            )

            print(
                f"Processing participant "
                f"{participant}, "
                f"session {session}, "
                f"phase {phase}: "
                f"{csv_file.name}"
            )

            try:
                dataframe = pd.read_csv(
                    csv_file
                )

                trials = extract_trials(
                    dataframe,
                    participant,
                    session,
                    phase,
                )

                questionnaires = (
                    extract_questionnaires(
                        dataframe,
                        participant,
                        session,
                        phase,
                    )
                )

                emotions = extract_emotions(
                    dataframe,
                    participant,
                    session,
                    phase,
                )

                output_file = (
                    export_participant_file(
                        participant,
                        session,
                        phase,
                        trials,
                        questionnaires,
                        emotions,
                    )
                )

                generated_files += 1

                print(
                    f"Saved: {output_file}"
                )

            except Exception as error:
                error_message = (
                    f"{csv_file.name}: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                errors.append(error_message)

                print(
                    f"ERROR: {error_message}"
                )

    print()
    print("PROCESS COMPLETE")
    print(
        f"Generated files: "
        f"{generated_files}"
    )

    if errors:
        print(f"Errors: {len(errors)}")

        for error_message in errors:
            print(f"  - {error_message}")

    else:
        print("Errors: 0")


if __name__ == "__main__":
    main()