"""List every stored figure behind one number, across all reports and tables.

    python -m scripts.inspect_fact HUL revenue_from_operations FY2023-24
    python -m scripts.inspect_fact "Nestle India" revenue_from_operations FY2024-25
"""
import sys

import duckdb

import config

if len(sys.argv) < 4:
    sys.exit("usage: python -m scripts.inspect_fact <company> <metric> <period>")
company, metric, period = sys.argv[1:4]
con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
rows = con.execute(
    "SELECT report_fy, basis, segment, value, unit, round(value_cr,1), substr(label,1,38), section, substr(table_id,1,8) "
    "FROM facts WHERE company=? AND metric=? AND period=? ORDER BY report_end DESC, basis, value_cr DESC NULLS LAST LIMIT 30",
    [company, metric, period]).fetchall()
print(f"{company} | {metric} | {period}: {len(rows)} stored figures")
print("report_fy | basis | segment | value | unit | value_cr | row label | section | table")
for r in rows:
    print("  ", " | ".join(str(x) for x in r))