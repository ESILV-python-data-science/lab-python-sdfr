# -*- coding: utf-8 -*-
# Christopher Kermorvant 2017

import csv
from  datetime import datetime
import statistics


# Constants are in UPPERCASE
JFK_FILE = 'jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv'


### Question 1
# first try to parse with split
with open(JFK_FILE, 'r') as csv_file:
    # read the first line = header
    header_line = csv_file.readline()
    # fields are separated by ";"
    nb_fields = len(header_line.split(';'))
    # use format syntax (new in python3) https://pyformat.info/#simple
    print('Expects {} fields on each line'.format(nb_fields))
    for i, line in enumerate(csv_file.readlines()):
        # check that we have the expected number of fields
        nb_line_fields = len(line.strip().split(';'))
        if not nb_line_fields == nb_fields:
            print('Line {} is KO found {} fields'.format(i, nb_line_fields))
            print('{}'.format(line))
            break
        print("Checked {} lines".format(i))

# use CSV library : always better to use existing libraries
with open(JFK_FILE, 'r', newline='') as csv_file:
    print('Checking with CSV library', end='...')
    lines_reader = csv.reader(csv_file, delimiter=';')
    nb_fields = 17
    for line in lines_reader:
        if not len(line) == nb_fields:
            print('Bad line')
            print(line)
    print('Done')


### Question 2
with open(JFK_FILE, 'r', newline='') as csv_file:
    # use the reader which gives access to column names
    lines_reader = csv.DictReader(csv_file, delimiter=';')
    all_page_number = []
    not_a_number = 0
    for line in lines_reader:
        try:
            page_num = int(line['Num Pages'])
            if page_num == 0:
                print('Document with 0 page is {}'.format(line['File Name']))

            all_page_number.append(page_num)
        except ValueError:
            not_a_number += 1
    print('Find {} page numbers'.format(len(all_page_number)))
    print('Number of missing number of pages : {} '.format(not_a_number))
    print('Total number of fields is {}'.format(len(all_page_number)+not_a_number))

    # Compute the Mean
    sum_pages = 0.0 # force to float, not needed in python3, but to be safe
    for n in all_page_number:
        sum_pages += n
    mean = sum_pages / len(all_page_number)
    print('Mean is {}'.format(mean))
    print('Mean (from statistics module) is {}'.format(statistics.mean(all_page_number)))

# Document with 0 page :
# this is https://www.archives.gov/files/research/jfk/releases/docid-32333253.pdf
# on the document, it is said that this number is not associated to any document.
# Either it never existed or it has been lost.

### Question 3
with open(JFK_FILE, 'r', newline='') as csv_file:
    for field in ['Doc Type', 'Agency']:
        # use the reader which gives access to column names
        lines_reader = csv.DictReader(csv_file, delimiter=';')
        field_types = set()
        for line in lines_reader:
            field_types.add(line[field])
        print('Number of {} is {}'.format(field, len(field_types)))

        # create dictionary to store counts
        field_types_number = {ftype:0 for ftype in field_types}
        # return to the beginning of the file
        csv_file.seek(0)
        # read again
        lines_reader = csv.DictReader(csv_file, delimiter=';')
        for line in lines_reader:
            # update count
            field_types_number[line[field]] += 1
        print('Number of document per {}'.format(field))
        print(field_types_number)
        # go back to beginning for next loop
        csv_file.seek(0)


### Question 4
with open(JFK_FILE, 'r', newline='') as csv_file:
    # use the reader which gives access to column names
    lines_reader = csv.DictReader(csv_file, delimiter=';')
    # number of field which are not convertible to date
    not_a_date = 0
    # find min and max date
    min_date = datetime.max
    max_date = datetime.min
    # store number of doc per year
    doc_per_year = {}
    for line in lines_reader:
        try:
            doc_date = datetime.strptime(line['Doc Date'], '%m/%d/%Y')
            if doc_date > max_date:
                max_date = doc_date
            if doc_date < min_date:
                min_date = doc_date
            # use get for default value is year is not in the dictionary
            doc_per_year[doc_date.year] = doc_per_year.get(doc_date.year, 0) + 1
        except ValueError:
            not_a_date += 1
    print('Could not convert {} dates'.format(not_a_date))
    print('Min date is {}'.format(min_date))
    print('Max date is {}'.format(max_date))
    print('Document per years:')
    years = [y for y in doc_per_year]
    _ = [print("{}: {}".format(y, doc_per_year[y]), end=' ') for y in sorted(years)]
    print()
