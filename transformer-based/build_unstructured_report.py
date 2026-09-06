"""Inventory and display all locally available text/event sources without fetching."""
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/unstructured-data'


def literals(path):
    result = {}
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                result[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def slug(value):
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = {}
    records = []
    def source(key, name=None, origin='', status='No historical text collected', url='', kind='text/event', detail=''):
        if key not in registry:
            registry[key] = dict(id=key, name=name or key, origins=[], urls=[], kind=kind, status=status, detail=detail)
        s = registry[key]
        if origin and origin not in s['origins']:
            s['origins'].append(origin)
        if url and url not in s['urls']:
            s['urls'].append(url)
        return s

    catalog = json.loads((ROOT / 'data/metadata/source_catalog.json').read_text())
    for s in catalog['sources']:
        source(s['id'], s['id'].replace('_', ' ').title(), 'data/metadata/source_catalog.json',
               s['status'], s.get('url') or '', s['modality'], s['feature_family'])

    data_rows = list(csv.DictReader((ROOT / 'data/Data Sources - Data.csv').open()))
    mappings = ['cbn_circulars', 'cbn_money_market', 'ngx_asi', 'dmo_fgn_bond_auction_results',
                'nbs_monthly_cpi', 'cbn_exchange_rates', 'cbn_daily_crude', 'cbn_reserves',
                'cbn_government_securities', 'wsj_dxy', 'fred_global_macro', 'fed_broad_dollar_index',
                'parallel_usdngn', 'cbn_money_credit', 'cbn_capital_importation', 'nbs_gdp',
                'wsj_vix', 'investing_gold', 'yahoo_brent']
    assert len(data_rows) == len(mappings)
    for row, key in zip(data_rows, mappings):
        source(key, row['Item'].strip(), 'Data Sources - Data.csv: ' + row['Item'].strip(),
               url=row['Link to source'], kind='quantitative reference', detail=json.dumps(row))

    event_path = ROOT / 'data/Data Sources - Events.csv'
    event_rows = list(csv.reader(event_path.open()))
    section = ''
    for row in event_rows:
        if row[0] in ['CBN MPC Meetings', 'FOMC Meetings', "BOE Meeting(UK's)", 'Chinese Holidays']:
            section = row[0]
        if len(row) > 2 and row[2].startswith('http'):
            key = {'CBN MPC Meetings': 'cbn_mpc_calendar' if 'calendar' in row[2] else 'cbn_mpc_decisions',
                   'FOMC Meetings': 'fomc_documents', "BOE Meeting(UK's)": 'boe_mpc_minutes'}[section]
            source(key, section + (' — ' + row[0] if row[0] else ''),
                   'Data Sources - Events.csv: ' + section, url=row[2])
    header_index = next(i for i, row in enumerate(event_rows) if row[0] == 'Holiday')
    years = event_rows[header_index][1:]
    for row in event_rows[header_index + 1:]:
        if not row[0]:
            continue
        key = 'chinese_holiday_' + slug(row[0])
        source(key, row[0], 'Data Sources - Events.csv', 'Manual year-level date ranges; unverified', kind='manual holiday')
        for year, value in zip(years, row[1:]):
            if value:
                records.append(dict(source_id=key, record_id=key+'_'+year, date=None, year=int(year),
                    date_basis='Year supplied in manual CSV; date range preserved verbatim',
                    kind='manual holiday entry', title=row[0], text=value, url='',
                    status='Unverified manual dates; may include future dates', origin='data/Data Sources - Events.csv', metadata=''))

    news = literals('app/services/news_aggregator.py')
    feed_keys = ['nairametrics','businessday_ng','vanguard_ng','thisday_live','premium_times','punch_ng','oilprice','imf_news']
    assert len(feed_keys) == len(news['DIRECT_RSS_FEEDS'])
    for key, (url, name, category) in zip(feed_keys, news['DIRECT_RSS_FEEDS']):
        source(key, name, 'app/services/news_aggregator.py: DIRECT_RSS_FEEDS', url=url, detail=category)
    source('cbn_press_releases', origin='app/services/news_aggregator.py: CBN scraper', url='https://www.cbn.gov.ng/Press/')
    for i, (query, category) in enumerate(news['GOOGLE_NEWS_QUERIES'], 1):
        source(f'google_news_query_{i:02d}', 'Google News: '+query, 'app/services/news_aggregator.py: GOOGLE_NEWS_QUERIES',
               url='https://news.google.com/rss/search?q='+quote(query), detail=category)
    legacy = literals('app/macro_calendar.py')
    for event in legacy['EVENTS']:
        key = 'legacy_event_' + slug(event['event'])
        source(key, event['event'], 'app/macro_calendar.py: EVENTS',
               'Definition and approximate schedule only; no dated historical occurrences', kind='legacy event definition',
               detail=json.dumps(dict(event, schedule=legacy['_SCHEDULES'].get(event['event'])), ensure_ascii=False))
        records.append(dict(source_id=key, record_id=key, date=None, year=None, date_basis='Undated recurring definition',
            kind='event definition', title=event['event'], text=json.dumps(event, ensure_ascii=False), url='',
            status='Heuristic context; not observed history', origin='app/macro_calendar.py', metadata=json.dumps(legacy['_SCHEDULES'].get(event['event']))))

    dataset = ds.dataset(ROOT / 'data/dataset/records', format='parquet', partitioning='hive')
    for key in dataset.to_table(columns=['source_id'])['source_id'].unique().to_pylist():
        source(key, origin='data/dataset/records', kind='dataset source')
    table = dataset.to_table(filter=ds.field('modality').isin(['unstructured','unstructured_index'])).to_pandas()
    for r in table.to_dict('records'):
        is_index = r['modality'] == 'unstructured_index'
        date = None if pd.isna(r['observed_at']) else r['observed_at'].strftime('%Y-%m-%d')
        year = None if pd.isna(r['period_year']) else int(r['period_year'])
        # Sitemap last-modified timestamps do not date the articles they may contain.
        records.append(dict(source_id=r['source_id'], record_id=r['record_id'],
            date=None if is_index else date, year=None if is_index else (int(date[:4]) if date else year),
            date_basis='Sitemap timestamp only; excluded from historical coverage' if is_index else ('Source-index date; publication not independently verified' if date else 'Archive year only'),
            kind='sitemap index' if is_index else 'document metadata', title=r['title'] or '', text=r['text_value'] or '',
            url=r['url'] or '', status=r['quality_status'], origin=r['source_path'],
            metadata=json.dumps(dict(source_metadata=r['metadata_json'], stored_observed_at=date, stored_period_year=year))))

    # This curated acquisition is newer than the canonical Parquet snapshot.
    nbs_path = ROOT / 'data/curated/unstructured/nbs_cpi_resources_index.csv'
    nbs = pd.read_csv(nbs_path).fillna('')
    source('nbs_cpi_resources', 'NBS CPI reports and resource packages', str(nbs_path.relative_to(ROOT)),
           'Resource metadata available; local package presence checked; text extraction pending', kind='document/resource')
    for r in nbs.to_dict('records'):
        date = r['publication_date'] or None
        period = str(r['period_date'])
        package = ROOT / 'data' / r['package_path']
        records.append(dict(source_id='nbs_cpi_resources', record_id='nbs_resource_'+str(r['resource_id']), date=date,
            year=int((date or period)[:4]) if (date or period) else None,
            date_basis='Publication date from curated index' if date else 'Reference-period year; publication date unknown',
            kind='document metadata' if r['extension']=='pdf' else 'resource package metadata', title=r['title'], text=r['title'],
            url=r['url'], status=('Local package present' if package.is_file() else 'Local package missing')+'; extracted body unavailable in this report',
            origin=str(nbs_path.relative_to(ROOT)), metadata=json.dumps(r, ensure_ascii=False)))

    records.sort(key=lambda r:(r['source_id'], r['date'] or str(r['year'] or '9999'), r['title']))
    totals = Counter(r['source_id'] for r in records)
    annual = defaultdict(Counter)
    daily = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r['year'] is not None and 2006 <= r['year'] <= 2026:
            annual[r['source_id']][str(r['year'])] += 1
        if r['date'] and '2006-01-01' <= r['date'] <= '2026-09-02':
            daily[r['date']][r['source_id']].append(r['record_id'])
    sources = sorted(registry.values(), key=lambda s:s['id'])
    for s in sources:
        s['records'] = totals[s['id']]
        s['annual'] = dict(annual[s['id']])
        s['dated_records'] = sum(len(v.get(s['id'], [])) for v in daily.values())
        s['record_kinds'] = dict(Counter(r['kind'] for r in records if r['source_id']==s['id']))
    payload = dict(start='2006-01-01', end='2026-09-02', sources=sources, records=records,
                   counts=dict(canonical_records=len(table), legacy_definitions=len(legacy['EVENTS']),
                               manual_holidays=7*len(years), nbs_resources=len(nbs)))
    (OUT/'source_inventory.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    pd.DataFrame([{**s, 'origins': ' | '.join(s['origins']), 'urls': ' | '.join(s['urls']),
                   'annual':json.dumps(s['annual']), 'record_kinds':json.dumps(s['record_kinds'])} for s in sources]).to_csv(OUT/'source_inventory.csv', index=False)
    pd.DataFrame(records).to_csv(OUT/'all_unstructured_records.csv', index=False)
    with (OUT/'daily_source_record_ids.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['date']+[s['id'] for s in sources])
        for date in pd.date_range('2006-01-01','2026-09-02').strftime('%Y-%m-%d'):
            writer.writerow([date]+[json.dumps(daily[date][s['id']]) if daily[date][s['id']] else '' for s in sources])
    pd.DataFrame([{'year':y, **{s['id']:annual[s['id']][str(y)] for s in sources}} for y in range(2006,2027)]).to_csv(OUT/'annual_source_record_counts.csv', index=False)
    template = (Path(__file__).with_name('unstructured_report_template.html')).read_text()
    (OUT/'all_unstructured_sources.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
    print(json.dumps(dict(sources=len(sources),records=len(records),counts=payload['counts'],output=str(OUT)), indent=2))


if __name__=='__main__':
    main()
