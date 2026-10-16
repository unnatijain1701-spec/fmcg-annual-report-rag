"""Sanity check of the numbers database.   python -m scripts.check_db"""
import duckdb

import config

con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
print("Rows per company and basis (facts_latest):")
for r in con.execute("SELECT company, basis, count(*) AS rows, sum(unit_inferred::int) AS unit_inferred "
                     "FROM facts_latest GROUP BY 1,2 ORDER BY 1,2").fetchall():
    print("  ", r)
print("\nConsolidated revenue from operations (Rs crore), company totals:")
rows = con.execute("SELECT company, period, round(value_cr, 0) FROM facts_latest "
                   "WHERE metric='revenue_from_operations' AND segment='' AND basis='consolidated' ORDER BY 1,2").fetchall()
for r in rows:
    print("  ", r)
have = {r[0] for r in rows}
all_c = {r[0] for r in con.execute("SELECT DISTINCT company FROM facts_latest").fetchall()}
print("\nCompanies with NO consolidated revenue row:", sorted(all_c - have) or "none")

print("\nCoverage of key metrics on the preferred basis (number of fiscal years with a company-total value):")
KEY = ["revenue_from_operations", "net_profit", "ebitda", "profit_before_tax", "eps_basic", "dividend_per_share", "total_assets", "total_equity"]
for company in sorted(all_c):
    got = dict(con.execute("SELECT metric, count(DISTINCT period) FROM facts_latest WHERE company=? AND segment='' AND default_basis "
                           "AND metric IN (" + ",".join("?" * len(KEY)) + ") GROUP BY 1", [company] + KEY).fetchall())
    print(f"  {company:14}", ", ".join(f"{m.replace('_from_operations','').replace('_per_share','')}={got.get(m, 0)}" for m in KEY))