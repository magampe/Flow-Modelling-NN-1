import csv

METADATA_START = '#        '
DATA_START = 'agency_cd'


def get_parameters(line):
    divider = line.strip().rindex('  ')
    code = '_'.join(i for i in line[:divider].strip().split(' ') if i)
    name = line[divider:].strip()
    return {code: name, f'{code}_cd': f'QC {name}'}


def process_gages(fname):
    parameter_codes = []
    code_lookup = {'agency_cd': 'agency_cd', 'site_no': 'site_no', 'datetime': 'datetime'}
    cols = None
    data = []
    with open(fname) as f:
        tsv = (i.rstrip() for i in f)
        for line in tsv:
            if line.startswith(METADATA_START):
                line = line.lstrip(METADATA_START)
                if not (line.startswith('TS') or line.startswith('Data')):
                    code_lookup.update(get_parameters(line))
            elif line.startswith(DATA_START):
                cols = [code_lookup[c] for c in line.split('\t')]
                parameter_codes.extend(c for c in cols if c not in parameter_codes)
                tsv.__next__()
            elif not line.startswith('#'):
                values = line.split('\t')
                for i, v in enumerate(values):
                    if v in ('Dis', 'Eqp'):
                        values[i] = None
                data.append(dict(zip(cols, values)))
    return parameter_codes, data


def create_csv(fields, data, fname):
    with open(fname, 'w', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


gage_fname = 'dv.tsv'
gage_csv = 'TN_gage_data_22_23.csv'
parameters, gage_data = process_gages(gage_fname)
create_csv(parameters, gage_data, gage_csv)
