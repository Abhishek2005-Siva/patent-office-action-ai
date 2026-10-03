import os
import traceback

OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

from google.cloud import bigquery

client = bigquery.Client()
print("BigQuery client created. Project:", client.project)

PROJECT = "patents-public-data"
DATASET = "uspto_oce_office_actions"


def run(name, query):
    print(f"\n=== {name} ===")
    try:
        df = client.query(query).to_dataframe()
        print("rows:", len(df))
        print(df.head(5).to_string())
        df.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
        return df
    except Exception:
        print(f"FAILED: {name}")
        traceback.print_exc()
        return None


# diagnostic: how are the office_actions.rejection_10X flag columns encoded?
run(
    "rejection_flag_distinct_values",
    f"""
    SELECT rejection_101, rejection_102, rejection_103, rejection_112, COUNT(*) AS n
    FROM `{PROJECT}.{DATASET}.office_actions`
    GROUP BY rejection_101, rejection_102, rejection_103, rejection_112
    ORDER BY n DESC
    LIMIT 20
    """,
)

run(
    "rejections_action_type_full",
    f"""
    SELECT action_type, action_subtype, COUNT(*) AS n
    FROM `{PROJECT}.{DATASET}.rejections`
    GROUP BY action_type, action_subtype
    ORDER BY n DESC
    """,
)

# real, deterministic sample of individual statutory rejections with real claim numbers
sample = run(
    "rejections_sample",
    f"""
    SELECT app_id, ifw_number, action_type, action_subtype, claim_numbers
    FROM `{PROJECT}.{DATASET}.rejections`
    WHERE action_type IN ('101', '102', '103', '112')
    ORDER BY app_id, ifw_number, action_type, action_subtype
    LIMIT 8000
    """,
)

run(
    "citations_for_sample_rejections",
    f"""
    SELECT c.app_id, c.ifw_number, c.action_type, c.action_subtype, c.citation_pat_pgpub_id, c.citation_in_oa
    FROM `{PROJECT}.{DATASET}.citations` c
    INNER JOIN (
        SELECT DISTINCT app_id, ifw_number
        FROM `{PROJECT}.{DATASET}.rejections`
        WHERE action_type IN ('101', '102', '103', '112')
        ORDER BY app_id, ifw_number
        LIMIT 8000
    ) r
    ON c.app_id = r.app_id AND c.ifw_number = r.ifw_number
    """,
)

run(
    "office_actions_for_sample",
    f"""
    SELECT app_id, ifw_number, mail_dt, art_unit, uspc_class, allowed_claims
    FROM `{PROJECT}.{DATASET}.office_actions`
    WHERE (app_id, ifw_number) IN (
        SELECT DISTINCT app_id, ifw_number
        FROM `{PROJECT}.{DATASET}.rejections`
        WHERE action_type IN ('101', '102', '103', '112')
        ORDER BY app_id, ifw_number
        LIMIT 8000
    )
    """,
)

# large distinct-value samples for regex/format stress-testing
run(
    "claim_numbers_large_sample",
    f"""
    SELECT DISTINCT claim_numbers
    FROM `{PROJECT}.{DATASET}.rejections`
    WHERE claim_numbers IS NOT NULL
    LIMIT 5000
    """,
)

run(
    "citation_id_large_sample",
    f"""
    SELECT DISTINCT citation_pat_pgpub_id, LENGTH(citation_pat_pgpub_id) AS len
    FROM `{PROJECT}.{DATASET}.citations`
    WHERE citation_pat_pgpub_id IS NOT NULL
    LIMIT 5000
    """,
)

print("\n=== DONE ===")
