import sys
from operator import add

from pyspark import SparkContext

sc = SparkContext(appName="LearningPySparkChapter02")

data = sc.parallelize(
    [('Amber', 22), ('Alfred', 23), ('Skye',4), ('Albert', 12),
     ('Amber', 9)])

data_from_file = sc.textFile('data/VS14MORT.txt.gz', 2)

data_heterogenous = sc.parallelize([('Ferrari', 'fast'), {'Porsche': 100000}, ['Spain','visited', 4504]]).collect()
print(data_heterogenous)
print(data_heterogenous[1]['Porsche'])
print(data_from_file.take(1))

def extractInformation(row):
  import re
  import numpy as np

  selected_indices = [
    2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 81, 82, 83, 84, 85, 87, 89
  ]

  '''
      Input record schema
      schema: n-m (o) -- xxx
          n - position from
          m - position to
          o - number of characters
          xxx - description
      1. 1-19 (19) -- reserved positions
      2. 20 (1) -- resident status
      3. 21-60 (40) -- reserved positions
      4. 61-62 (2) -- education code (1989 revision)
      5. 63 (1) -- education code (2003 revision)
      6. 64 (1) -- education reporting flag
      7. 65-66 (2) -- month of death
      8. 67-68 (2) -- reserved positions
      9. 69 (1) -- sex
      10. 70 (1) -- age: 1-years, 2-months, 4-days, 5-hours, 6-minutes, 9-not stated
      11. 71-73 (3) -- number of units (years, months etc)
      12. 74 (1) -- age substitution flag (if the age reported in positions 70-74 is calculated using dates of birth and death)
      13. 75-76 (2) -- age recoded into 52 categories
      14. 77-78 (2) -- age recoded into 27 categories
      15. 79-80 (2) -- age recoded into 12 categories
      16. 81-82 (2) -- infant age recoded into 22 categories
      17. 83 (1) -- place of death
      18. 84 (1) -- marital status
      19. 85 (1) -- day of the week of death
      20. 86-101 (16) -- reserved positions
      21. 102-105 (4) -- current year
      22. 106 (1) -- injury at work
      23. 107 (1) -- manner of death
      24. 108 (1) -- manner of disposition
      25. 109 (1) -- autopsy
      26. 110-143 (34) -- reserved positions
      27. 144 (1) -- activity code
      28. 145 (1) -- place of injury
      29. 146-149 (4) -- ICD code
      30. 150-152 (3) -- 358 cause recode
      31. 153 (1) -- reserved position
      32. 154-156 (3) -- 113 cause recode
      33. 157-159 (3) -- 130 infant cause recode
      34. 160-161 (2) -- 39 cause recode
      35. 162 (1) -- reserved position
      36. 163-164 (2) -- number of entity-axis conditions
      37-56. 165-304 (140) -- list of up to 20 conditions
      57. 305-340 (36) -- reserved positions
      58. 341-342 (2) -- number of record axis conditions
      59. 343 (1) -- reserved position
      60-79. 344-443 (100) -- record axis conditions
      80. 444 (1) -- reserve position
      81. 445-446 (2) -- race
      82. 447 (1) -- bridged race flag
      83. 448 (1) -- race imputation flag
      84. 449 (1) -- race recode (3 categories)
      85. 450 (1) -- race recode (5 categories)
      86. 461-483 (33) -- reserved positions
      87. 484-486 (3) -- Hispanic origin
      88. 487 (1) -- reserved
      89. 488 (1) -- Hispanic origin/race recode
   '''

  record_split = re \
    .compile(
    r'([\s]{19})([0-9]{1})([\s]{40})([0-9\s]{2})([0-9\s]{1})([0-9]{1})([0-9]{2})' +
    r'([\s]{2})([FM]{1})([0-9]{1})([0-9]{3})([0-9\s]{1})([0-9]{2})([0-9]{2})' +
    r'([0-9]{2})([0-9\s]{2})([0-9]{1})([SMWDU]{1})([0-9]{1})([\s]{16})([0-9]{4})' +
    r'([YNU]{1})([0-9\s]{1})([BCOU]{1})([YNU]{1})([\s]{34})([0-9\s]{1})([0-9\s]{1})' +
    r'([A-Z0-9\s]{4})([0-9]{3})([\s]{1})([0-9\s]{3})([0-9\s]{3})([0-9\s]{2})([\s]{1})' +
    r'([0-9\s]{2})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' +
    r'([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' +
    r'([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' +
    r'([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})([A-Z0-9\s]{7})' +
    r'([A-Z0-9\s]{7})([\s]{36})([A-Z0-9\s]{2})([\s]{1})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' +
    r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' +
    r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' +
    r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})' +
    r'([A-Z0-9\s]{5})([A-Z0-9\s]{5})([A-Z0-9\s]{5})([\s]{1})([0-9\s]{2})([0-9\s]{1})' +
    r'([0-9\s]{1})([0-9\s]{1})([0-9\s]{1})([\s]{33})([0-9\s]{3})([0-9\s]{1})([0-9\s]{1})')
  try:
    rs = np.array(record_split.split(row))[selected_indices]
  except:
    rs = np.array(['-99'] * len(selected_indices))
  return rs

#     return record_split.split(row)
data_from_file_conv = data_from_file.map(extractInformation)
data_from_file_conv.map(lambda row: row).take(1)

sc.stop()

#data_from_file = sc.\
#    textFile(
#        '/Users/drabast/Documents/PySpark_Data/VS14MORT.txt.gz',
#        4)
